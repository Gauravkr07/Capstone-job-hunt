"""
ATS (Applicant Tracking System) match scoring: compares a user's resume
profile against a specific job posting using two independent signals --

  1. Vector similarity  -- cosine similarity between the resume profile
     embedding and the job embedding (same model/vector space for both,
     see helpers/embeddings.py). Captures semantic/contextual fit that
     exact keyword matching misses (e.g. "built REST APIs" vs "API development").

  2. Skill overlap       -- exact-match comparison between the resume's
     extracted skill set and the job's required/preferred skills (stored
     structured in Postgres via JobSkill). Captures precise, literal
     requirement fit that vector similarity can blur or miss entirely.

These are fused into one weighted score. Weighting required skills higher
than preferred, and skill-overlap higher than raw vector similarity, since a
resume that's semantically "similar" but missing hard requirements should not
outrank one that actually has them.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Job, JobMatch, User
from db.vector_store import (
    get_job_vector_point,
    get_job_vector_points,
    get_resume_vector_point,
    search_resume_chunks,
    search_resume_profiles,
    search_similar_jobs,
)
from helpers.embeddings import embed_text
from helpers.llm_match_explainer import explain_match_via_llm
from helpers.logger import get_logger
from services.job_service import fetch_job_skills, job_embedding_text

logger = get_logger("ats_service")

# How many resume chunks to retrieve as grounding context for a match
# explanation -- enough for a well-rounded answer without bloating the LLM prompt.
EXPLAIN_MATCH_CHUNK_LIMIT = 5

# Recommendations retrieval: how many candidates to pull from vector search
# before reranking by fused score. Wider than the final returned list so
# reranking has enough candidates to actually reorder.
RECOMMENDATION_RETRIEVAL_LIMIT = 200

# When a job is added, how many of the most similar CVs are scored against it.
CANDIDATE_POOL_LIMIT = 500

# Fusion weights -- must sum to 1.0. Skill overlap is weighted higher than
# raw vector similarity because it reflects literal, checkable requirements.
VECTOR_WEIGHT = 0.4
SKILL_WEIGHT = 0.6

# Within the skill-overlap score, required skills matter far more than preferred.
REQUIRED_SKILL_WEIGHT = 0.8
PREFERRED_SKILL_WEIGHT = 0.2


@dataclass
class AtsMatchResult:
    job_id: int
    vector_score: float
    skill_score: float
    fused_score: float
    has_skill_data: bool = True
    matched_required_skills: list[str] = field(default_factory=list)
    missing_required_skills: list[str] = field(default_factory=list)
    matched_preferred_skills: list[str] = field(default_factory=list)
    missing_preferred_skills: list[str] = field(default_factory=list)
    explanation: dict[str, Any] | None = None
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _skill_overlap_score(
    resume_skills: set[str],
    required_skills: list[str],
    preferred_skills: list[str],
) -> tuple[float, list[str], list[str], list[str], list[str]]:
    """
    Returns 0.0 (not 1.0) when a job has no extracted skills at all -- an
    empty requirement list means "no skill signal available" (e.g. the job's
    domain isn't covered by the skill taxonomy), not "trivially satisfied".
    Treating it as a perfect match would let skill-less jobs outscore jobs
    with genuine, verifiable overlap.
    """
    matched_required = [s for s in required_skills if s in resume_skills]
    missing_required = [s for s in required_skills if s not in resume_skills]
    matched_preferred = [s for s in preferred_skills if s in resume_skills]
    missing_preferred = [s for s in preferred_skills if s not in resume_skills]

    if not required_skills and not preferred_skills:
        return 0.0, matched_required, missing_required, matched_preferred, missing_preferred

    required_ratio = len(matched_required) / len(required_skills) if required_skills else 1.0
    preferred_ratio = len(matched_preferred) / len(preferred_skills) if preferred_skills else 1.0

    score = required_ratio * REQUIRED_SKILL_WEIGHT + preferred_ratio * PREFERRED_SKILL_WEIGHT
    return score, matched_required, missing_required, matched_preferred, missing_preferred


def _score_match(
    job_id: int,
    resume_vector: list[float],
    resume_skills: set[str],
    job_vector: list[float],
    required_skills: list[str],
    preferred_skills: list[str],
) -> AtsMatchResult:
    """Shared scoring core used by both the single-job ATS check and multi-job recommendations."""
    has_skill_data = bool(required_skills or preferred_skills)

    vector_score = _cosine_similarity(resume_vector, job_vector)
    skill_score, matched_req, missing_req, matched_pref, missing_pref = _skill_overlap_score(
        resume_skills, required_skills, preferred_skills
    )

    if has_skill_data:
        fused_score = vector_score * VECTOR_WEIGHT + skill_score * SKILL_WEIGHT
    else:
        # No skill signal to fuse with -- fall back to vector similarity alone
        # rather than diluting the score with a meaningless zero.
        fused_score = vector_score

    return AtsMatchResult(
        job_id=job_id,
        vector_score=round(vector_score, 4),
        skill_score=round(skill_score, 4),
        fused_score=round(fused_score, 4),
        has_skill_data=has_skill_data,
        matched_required_skills=matched_req,
        missing_required_skills=missing_req,
        matched_preferred_skills=matched_pref,
        missing_preferred_skills=missing_pref,
    )


async def _require_resume_point(username: str) -> dict[str, Any]:
    resume_point = await asyncio.to_thread(get_resume_vector_point, username)
    if not resume_point:
        raise ValueError(f"No resume vector found for user '{username}' -- submit a resume first")
    return resume_point


async def _load_job_context(
    session: AsyncSession, username: str, job_id: int
) -> tuple[Job, dict[str, Any], dict[str, Any], list[str], list[str]]:
    """
    Everything a single-job match needs: (job, job_point, resume_point,
    required_skills, preferred_skills). Raises ValueError if the job, its
    embedding, or the caller's resume is missing.
    """
    job = await session.get(Job, job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    required_skills, preferred_skills = (await fetch_job_skills(session, [job_id]))[job_id]

    job_point, resume_point = await asyncio.gather(
        asyncio.to_thread(get_job_vector_point, job_id), _require_resume_point(username)
    )
    if not job_point:
        raise ValueError(f"No embedding found for job {job_id}")

    return job, job_point, resume_point, required_skills, preferred_skills


async def _generate_match_explanation(
    job_vector: list[float],
    job_title: str,
    company_name: str,
    username: str,
    required_skills: list[str],
    preferred_skills: list[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Shared RAG core used by explain_match(), compute_ats_match(...,
    include_explanation=True) and analyze_job_description(): retrieve the
    resume chunks most relevant to this job (vector search over resume_chunks,
    scoped to this user), then ask Groq to generate a grounded natural-language
    explanation using only those retrieved chunks plus the job's skills.

    Returns (explanation_or_none, retrieved_chunks). explanation is None if
    no LLM is configured or the call failed -- retrieved_chunks are still
    returned so the caller has the raw evidence either way.
    """
    chunks = await asyncio.to_thread(search_resume_chunks, job_vector, EXPLAIN_MATCH_CHUNK_LIMIT, username)

    explanation = await explain_match_via_llm(job_title, company_name, required_skills, preferred_skills, chunks)

    retrieved_chunks = [
        {"section": chunk.get("section"), "text": chunk.get("text"), "score": chunk.get("score")}
        for chunk in chunks
    ]
    return explanation, retrieved_chunks


async def _explain_stored_job(job: Job, job_point: dict[str, Any], username: str, required: list[str], preferred: list[str]):
    payload = job_point["payload"]
    return await _generate_match_explanation(
        job_point["vector"], payload.get("title", job.title), payload.get("company", ""), username, required, preferred
    )


async def compute_ats_match(
    session: AsyncSession,
    username: str,
    job_id: int,
    include_explanation: bool = False,
) -> AtsMatchResult:
    """
    Compute a hybrid (vector + skill-overlap) ATS match score between a
    user's resume and one job posting. Raises ValueError if the job, its
    embedding, or the resume embedding is missing.

    Set include_explanation=True to also run the RAG-based match explanation
    (retrieves resume chunks + calls Groq) and attach it to the result --
    this adds a network round-trip and LLM latency, so it's opt-in and
    defaults to off. Callers that score many jobs at once (e.g.
    get_job_recommendations) should never set this, to avoid one LLM call
    per candidate.
    """
    job, job_point, resume_point, required_skills, preferred_skills = await _load_job_context(
        session, username, job_id
    )

    result = _score_match(
        job_id,
        resume_point["vector"],
        set(resume_point["payload"].get("skills") or []),
        job_point["vector"],
        required_skills,
        preferred_skills,
    )

    if include_explanation:
        result.explanation, result.retrieved_chunks = await _explain_stored_job(
            job, job_point, username, required_skills, preferred_skills
        )

    return result


async def explain_match(session: AsyncSession, username: str, job_id: int) -> dict[str, Any]:
    """
    Standalone RAG-based match explanation (no numeric score) -- see
    _generate_match_explanation for the retrieval+generation logic shared
    with compute_ats_match(..., include_explanation=True).

    Returns {"explanation": {...} or None, "retrieved_chunks": [...]}.

    Raises ValueError if the job doesn't exist or the caller has no resume.
    """
    job, job_point, _, required_skills, preferred_skills = await _load_job_context(session, username, job_id)

    explanation, retrieved_chunks = await _explain_stored_job(
        job, job_point, username, required_skills, preferred_skills
    )
    return {"explanation": explanation, "retrieved_chunks": retrieved_chunks}


def _to_match_row(username: str, result: AtsMatchResult) -> JobMatch:
    return JobMatch(
        username=username,
        job_id=result.job_id,
        vector_score=result.vector_score,
        skill_score=result.skill_score,
        fused_score=result.fused_score,
        has_skill_data=result.has_skill_data,
        matched_required_skills=result.matched_required_skills,
        missing_required_skills=result.missing_required_skills,
        matched_preferred_skills=result.matched_preferred_skills,
        missing_preferred_skills=result.missing_preferred_skills,
    )


def _from_match_row(row: JobMatch) -> AtsMatchResult:
    return AtsMatchResult(
        job_id=row.job_id,
        vector_score=row.vector_score,
        skill_score=row.skill_score,
        fused_score=row.fused_score,
        has_skill_data=row.has_skill_data,
        matched_required_skills=row.matched_required_skills or [],
        missing_required_skills=row.missing_required_skills or [],
        matched_preferred_skills=row.matched_preferred_skills or [],
        missing_preferred_skills=row.missing_preferred_skills or [],
    )


async def refresh_matches_for_user(session: AsyncSession, username: str) -> int:
    """
    Recompute and store (job_matches) how this user's CV scores against the
    active jobs most similar to it. Called whenever the user's CV changes, so
    old matches never outlive the CV they were computed from. Two-stage:
    vector search pulls the RECOMMENDATION_RETRIEVAL_LIMIT nearest jobs, then
    each is rescored with the hybrid vector + skill-overlap formula.
    """
    await session.execute(delete(JobMatch).where(JobMatch.username == username))

    resume_point = await asyncio.to_thread(get_resume_vector_point, username)
    if not resume_point:
        await session.commit()
        return 0
    resume_skills = set(resume_point["payload"].get("skills") or [])

    candidates = await asyncio.to_thread(
        search_similar_jobs, resume_point["vector"], RECOMMENDATION_RETRIEVAL_LIMIT, "ACTIVE"
    )
    job_ids = [c["job_id"] for c in candidates]
    # Only jobs that still exist in Postgres (the source of truth).
    job_ids = list((await session.execute(select(Job.id).where(Job.id.in_(job_ids)))).scalars())
    job_points = await asyncio.to_thread(get_job_vector_points, job_ids)
    skills = await fetch_job_skills(session, job_ids)

    rows = [
        _to_match_row(username, _score_match(job_id, resume_point["vector"], resume_skills, job_points[job_id]["vector"], *skills[job_id]))
        for job_id in job_ids
        if job_id in job_points
    ]
    session.add_all(rows)
    await session.commit()
    logger.info("Stored %d job matches for %s", len(rows), username)
    return len(rows)


async def refresh_matches_for_job(session: AsyncSession, job_id: int) -> int:
    """
    Score one (new or changed) job against every user's CV and store the
    results, so it appears in each matching user's recommendations straight
    away -- without waiting for them to re-upload their CV.
    """
    job_point = await asyncio.to_thread(get_job_vector_point, job_id)
    await session.execute(delete(JobMatch).where(JobMatch.job_id == job_id))
    if not job_point:
        await session.commit()
        return 0

    required, preferred = (await fetch_job_skills(session, [job_id]))[job_id]
    profiles = await asyncio.to_thread(search_resume_profiles, job_point["vector"], CANDIDATE_POOL_LIMIT)
    usernames = [p["payload"].get("username") for p in profiles]
    known = set((await session.execute(select(User.username).where(User.username.in_(usernames)))).scalars())

    rows = [
        _to_match_row(
            p["payload"]["username"],
            _score_match(job_id, p["vector"], set(p["payload"].get("skills") or []), job_point["vector"], required, preferred),
        )
        for p in profiles
        if p["payload"].get("username") in known
    ]
    session.add_all(rows)
    await session.commit()
    logger.info("Stored %d candidate matches for job %s", len(rows), job_id)
    return len(rows)


async def get_job_recommendations(
    session: AsyncSession, username: str, limit: int = 10, min_score: float = 0.0
) -> list[AtsMatchResult]:
    """
    The user's stored recommendations (job_matches) for ACTIVE jobs, best
    fused_score first. If none are stored yet (e.g. data from before matches
    were persisted), they're computed once and stored. Raises ValueError if
    the user has no CV.
    """

    async def read() -> list[AtsMatchResult]:
        rows = await session.execute(
            select(JobMatch)
            .join(Job, Job.id == JobMatch.job_id)
            .where(JobMatch.username == username, Job.status == "ACTIVE", JobMatch.fused_score >= min_score)
            .order_by(JobMatch.fused_score.desc())
            .limit(limit)
        )
        return [_from_match_row(r) for r in rows.scalars()]

    results = await read()
    if results:
        return results

    has_any = (await session.execute(select(JobMatch.id).where(JobMatch.username == username).limit(1))).first()
    if not has_any:
        await _require_resume_point(username)  # raises ValueError when there's no CV
        await refresh_matches_for_user(session, username)
        results = await read()
    return results


async def analyze_job_description(username: str, parsed: dict[str, Any], include_explanation: bool = True) -> AtsMatchResult:
    """
    Score the caller's resume against a job description that is NOT stored
    (the "check a JD" feature): same embedding text, scoring and explanation
    as a stored job, computed in memory. `parsed` is the output of
    helpers.jd_scraper.parse_raw_job_posting. job_id is 0 on the result.
    """
    resume_point = await _require_resume_point(username)
    required, preferred = parsed["required_skills"], parsed["preferred_skills"]
    job_vector = await asyncio.to_thread(
        embed_text,
        job_embedding_text(
            parsed["title"], parsed["company_name"], parsed["skills"], parsed["industry_type"], parsed["cleaned_description"]
        ),
    )

    result = _score_match(
        0, resume_point["vector"], set(resume_point["payload"].get("skills") or []), job_vector, required, preferred
    )
    if include_explanation:
        result.explanation, result.retrieved_chunks = await _generate_match_explanation(
            job_vector, parsed["title"], parsed["company_name"], username, required, preferred
        )
    return result
