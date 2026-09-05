"""
RAG chatbot generation: embeds the user's question, retrieves grounding
context from Qdrant (the user's own resume chunks + similar active job
postings) and Postgres (profile/user_details), reranks the retrieved
candidates by lexical overlap with the query on top of vector similarity,
then asks Groq to answer using only that retrieved context -- either as a
single response or as a token stream.

Follows the same Groq-via-httpx pattern as helpers/llm_match_explainer.py.
"""
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from config import GROQ_API_KEY, GROQ_MODEL
from helpers.logger import get_logger

logger = get_logger("chatbot_rag")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
    "of", "and", "or", "my", "me", "i", "what", "which", "how", "do", "does",
    "with", "about", "tell", "have", "has", "any", "can", "you", "your", "it",
    "this", "that", "be", "as",
}


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#.]+", text.lower()) if w and w not in _STOPWORDS}


def _lexical_overlap(query_words: set[str], text: str) -> float:
    if not query_words or not text:
        return 0.0
    text_words = _keywords(text)
    if not text_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)


def rerank_resume_chunks(question: str, chunks: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """
    Hybrid rerank: blend Qdrant's cosine similarity score with lexical
    keyword overlap against the raw question, so chunks that literally
    mention terms from the question (e.g. a skill name) outrank chunks that
    are only vaguely similar in embedding space. Returns the top_k reranked.
    """
    query_words = _keywords(question)
    scored = []
    for chunk in chunks:
        vector_score = float(chunk.get("score") or 0.0)
        lexical_score = _lexical_overlap(query_words, chunk.get("text", ""))
        blended = 0.7 * vector_score + 0.3 * lexical_score
        scored.append({**chunk, "rerank_score": blended})
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


def rerank_jobs(question: str, jobs: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    query_words = _keywords(question)
    scored = []
    for job in jobs:
        vector_score = float(job.get("score") or 0.0)
        searchable = " ".join(
            [job.get("title", ""), job.get("industry_type", "") or "", " ".join(job.get("skills") or [])]
        )
        lexical_score = _lexical_overlap(query_words, searchable)
        blended = 0.7 * vector_score + 0.3 * lexical_score
        scored.append({**job, "rerank_score": blended})
    scored.sort(key=lambda j: j["rerank_score"], reverse=True)
    return scored[:top_k]

_SYSTEM_PROMPT = """You are a candidate-profile assistant for a job-hunting platform, built for \
recruiters and hiring managers to look up a registered candidate's professional background. \
Answer helpfully and strictly grounded in the context provided -- never inventing facts that \
aren't present in it.

You will be given, about ONE registered candidate:
- Their registration profile: full name, email, age, graduation status, LinkedIn/LeetCode/HackerRank \
links, resume skills, years of experience, industry.
- Resume excerpts retrieved from their uploaded resume (may be empty if none uploaded), ranked \
most relevant first.
- Job postings retrieved as relevant to their resume/the question (may be empty), ranked most \
relevant first.
- The recent conversation history, for context on follow-up questions.

Scope rules -- follow strictly:
1. Only answer using the profile/resume/job context given. If it doesn't contain enough to \
answer, say so plainly (e.g. "no resume on file for this candidate") rather than guessing.
2. Identity and contact info (name, email, LinkedIn, LeetCode, HackerRank) are always fine to \
share when asked -- this platform exists so recruiters can find and contact candidates.
3. Otherwise stay professional/technical: skills, experience, projects, education, tools, \
industry, and job matches. Age and graduation status may be stated as plain facts if asked, but \
never speculated about or elaborated on beyond the raw value.
4. Politely decline only questions with nothing to do with this candidate's registration or \
resume data at all -- e.g. personal opinions or unrelated topics -- noting briefly that you \
only answer questions about the registered candidate's profile and resume. Do NOT decline \
questions about name, contact info, skills, or experience -- those are always in scope.
5. Never guess a person's identity, background, or skills beyond what's explicitly in the context. \
If a resume excerpt looks like unfilled template boilerplate (e.g. "[Your Name]", "Lorem ipsum", \
placeholder brackets), ignore it rather than repeating it as if it were real data.

Formatting rules -- follow exactly, every time:
- Write in clear, correctly punctuated, professional English. No emojis, no emoji bullets, no \
exclamation marks, no filler sign-offs ("Let me know if...", "Feel free to ask...").
- Open with one short line naming the candidate and their role/seniority -- no heading needed \
for this line.
- Group the rest of the answer under plain bold section labels relevant to what was asked (choose \
from, in this order when relevant: **Skills**, **Experience**, **Projects**, **Education**, \
**Contact**) -- omit any section that has nothing to report or wasn't asked about.
- Under each section label, use a blank line then a "- " bullet per item. One fact per bullet, \
short and specific. Do not merge multiple facts into one run-on bullet.
- Leave exactly one blank line between the opening line and the first section, and between each \
section, so the answer is visually chunked and easy to scan -- never one dense paragraph.
- Do not answer with a bare JSON object or code fence; this structured bullet format IS the \
required output shape, always, even for short answers.
Reply in plain text with this markdown structure (bold labels, blank lines, bullets)."""


def _build_user_prompt(
    question: str,
    profile: dict[str, Any],
    resume_chunks: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> str:
    profile_lines = (
        f"Username: {profile.get('username')}\n"
        f"Full name: {profile.get('full_name') or 'not provided'}\n"
        f"Email: {profile.get('email') or 'not provided'}\n"
        f"Age: {profile.get('age')}\n"
        f"Graduated: {profile.get('graduated')}\n"
        f"LinkedIn: {profile.get('linkedin_url') or 'not provided'}\n"
        f"LeetCode: {profile.get('leetcode_url') or 'not provided'}\n"
        f"HackerRank: {profile.get('hackerrank_url') or 'not provided'}\n"
        f"Skills: {', '.join(profile.get('resume_skills') or []) or 'none on file'}\n"
        f"Experience (years): {profile.get('experience') if profile.get('experience') is not None else 'unknown'}\n"
        f"Industry: {profile.get('industry_type') or 'unknown'}"
    )

    chunk_lines = (
        "\n\n".join(
            f"({idx + 1}) [{c.get('section', 'unknown')}]\n{c.get('text', '')}"
            for idx, c in enumerate(resume_chunks)
        )
        or "No resume excerpts available."
    )

    job_lines = (
        "\n".join(
            f"({idx + 1}) {j.get('title', 'Unknown title')} at {j.get('company', 'Unknown company')} "
            f"(industry: {j.get('industry_type') or 'n/a'}, remote: {j.get('remote_type') or 'n/a'}, "
            f"skills: {', '.join(j.get('skills') or []) or 'n/a'})"
            for idx, j in enumerate(jobs)
        )
        or "No matching job postings retrieved."
    )

    return (
        f"Question about registered candidate '{profile.get('username')}': {question}\n\n"
        f"=== Candidate registration profile ===\n{profile_lines}\n\n"
        f"=== Resume excerpts (ranked most relevant first) ===\n{chunk_lines}\n\n"
        f"=== Relevant job postings (ranked most relevant first) ===\n{job_lines}"
    )


def _build_messages(
    question: str,
    profile: dict[str, Any],
    resume_chunks: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": _build_user_prompt(question, profile, resume_chunks, jobs)})
    return messages


async def stream_answer_user_question(
    question: str,
    profile: dict[str, Any],
    resume_chunks: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """
    Ask Groq to answer `question`, yielding text deltas as they arrive over
    Groq's SSE stream. Yields nothing (empty stream) when no API key is
    configured or the request fails outright -- callers should treat an
    empty stream as "fall back to a canned response".
    """
    if not GROQ_API_KEY:
        logger.info("GROQ_API_KEY not configured; skipping streaming RAG chatbot generation")
        return

    import httpx

    payload = {
        "model": GROQ_MODEL,
        "messages": _build_messages(question, profile, resume_chunks, jobs, history),
        "temperature": 0.3,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", GROQ_API_URL, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content")
                    except Exception:
                        continue
                    if delta:
                        yield delta
    except Exception as exc:
        logger.warning("Groq streaming chatbot request failed: %s", exc)
        return
