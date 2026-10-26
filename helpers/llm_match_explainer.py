"""
RAG-style match explanation: retrieves the resume chunks most relevant to a
job (via vector search over resume_chunks -- see db/vector_store.search_resume_chunks)
and asks Groq's LLM to write a grounded, natural-language explanation of the
match using only that retrieved text, not the whole resume or its own
assumptions about the candidate.

This is the "generation" half that's otherwise missing from this project's
retrieval-only matching (ATS check / recommendations score and rank, but
never generate an explanation of *why*).
"""
from typing import Any

from helpers import groq_client
from helpers.logger import get_logger

logger = get_logger("llm_match_explainer")

_SYSTEM_PROMPT = """You are an expert technical recruiter and ATS (Applicant Tracking System) \
analyst: a careful, evidence-based evaluator who explains why a candidate's resume matches a \
job posting, strictly grounded in the material provided -- never overselling a candidate or \
inventing qualifications they haven't demonstrated.

You will be given:
- The job's title, company, and required/preferred skills.
- Several excerpts retrieved from the candidate's resume (labeled by section).

Ground every claim ONLY in the provided resume excerpts and job skills -- do
not invent experience, employers, or skills that are not present in the
excerpts. If the excerpts don't support a strong match, say so plainly rather
than overselling it.

Return ONLY a JSON object matching this exact shape, no prose, no markdown fences:

{
  "summary": string (2-3 sentences: overall fit, grounded in the excerpts),
  "supporting_points": array of strings (specific resume evidence for the match, one point per string),
  "gaps": array of strings (required/preferred skills or experience not evidenced in the excerpts)
}"""


def _build_user_prompt(job_title: str, company_name: str, required_skills: list[str], preferred_skills: list[str], chunks: list[dict[str, Any]]) -> str:
    chunk_lines = "\n\n".join(f"[{chunk.get('section', 'unknown')}]\n{chunk.get('text', '')}" for chunk in chunks)
    return (
        f"Job title: {job_title}\n"
        f"Company: {company_name}\n"
        f"Required skills: {', '.join(required_skills) or 'none listed'}\n"
        f"Preferred skills: {', '.join(preferred_skills) or 'none listed'}\n\n"
        f"Resume excerpts (most relevant to this job, retrieved by similarity search):\n\n{chunk_lines}"
    )


async def explain_match_via_llm(
    job_title: str,
    company_name: str,
    required_skills: list[str],
    preferred_skills: list[str],
    chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Ask Groq to generate a grounded match explanation from retrieved resume
    chunks. Returns None (never raises) when no API key is configured, there
    are no chunks to ground the explanation in, or the call fails.
    """
    if not chunks:
        logger.info("No resume chunks retrieved; skipping match explanation")
        return None

    parsed = await groq_client.chat_json(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(job_title, company_name, required_skills, preferred_skills, chunks)},
        ],
        temperature=0.2,
        label="match explanation",
    )
    if parsed is None:
        return None

    return {
        "summary": (parsed.get("summary") or "").strip() or None,
        "supporting_points": [str(p).strip() for p in (parsed.get("supporting_points") or []) if str(p).strip()],
        "gaps": [str(g).strip() for g in (parsed.get("gaps") or []) if str(g).strip()],
    }
