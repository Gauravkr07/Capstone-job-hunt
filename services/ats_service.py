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
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Job, JobSkill, Skill
from db.vector_store import (
    get_job_vector_point,
    get_job_vector_points,
    get_resume_vector_point,
    search_resume_chunks,
    search_similar_jobs,
)
from helpers.llm_match_explainer import explain_match_via_llm

# How many resume chunks to retrieve as grounding context for a match
# explanation -- enough for a well-rounded answer without bloating the LLM prompt.
EXPLAIN_MATCH_CHUNK_LIMIT = 5

# Recommendations retrieval: how many candidates to pull from vector search
# before reranking by fused score. Wider than the final returned list so
# reranking has enough candidates to actually reorder.
RECOMMENDATION_RETRIEVAL_LIMIT = 50

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


async def _fetch_job_required_preferred_skills(session: AsyncSession, job_id: int) -> tuple[list[str], list[str]]:
    skill_rows = await session.execute(select(JobSkill, Skill).join(Skill).where(JobSkill.job_id == job_id))
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    for job_skill, skill in skill_rows.all():
        if job_skill.importance == "preferred":
            preferred_skills.append(skill.name)
        else:
            required_skills.append(skill.name)
    return required_skills, preferred_skills


async def _fetch_job_skills_batch(
    session: AsyncSession, job_ids: list[int]
) -> dict[int, tuple[list[str], list[str]]]:
    """
    Batched version of _fetch_job_required_preferred_skills -- one Postgres
    round-trip for many jobs instead of one per job. Returns
    {job_id: (required_skills, preferred_skills)}.
    """
    if not job_ids:
        return {}

    rows = await session.execute(
        select(JobSkill.job_id, JobSkill.importance, Skill.name)
        .join(Skill)
        .where(JobSkill.job_id.in_(job_ids))
    )

    result: dict[int, tuple[list[str], list[str]]] = {job_id: ([], []) for job_id in job_ids}
    for job_id, importance, skill_name in rows.all():
        required, preferred = result[job_id]
        (preferred if importance == "preferred" else required).append(skill_name)

    return result


def _score_match(
    resume_vector: list[float],
    resume_skills: set[str],
    job_vector: list[float],
    required_skills: list[str],
    preferred_skills: list[str],
) -> tuple[float, float, float, bool, list[str], list[str], list[str], list[str]]:
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

    return vector_score, skill_score, fused_score, has_skill_data, matched_req, missing_req, matched_pref, missing_pref


async def _generate_match_explanation(
    job: Job,
    job_point: dict[str, Any],
    username: str,
    required_skills: list[str],
    preferred_skills: list[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Shared RAG core used by both explain_match() and compute_ats_match(...,
    include_explanation=True): retrieve the resume chunks most relevant to
    this job (vector search over resume_chunks, scoped to this user), then
    ask Groq to generate a grounded natural-language explanation using only
    those retrieved chunks plus the job's required/preferred skills.

    Returns (explanation_or_none, retrieved_chunks). explanation is None if
    no LLM is configured or the call failed -- retrieved_chunks are still
    returned so the caller has the raw evidence either way.
    """
    chunks = search_resume_chunks(job_point["vector"], limit=EXPLAIN_MATCH_CHUNK_LIMIT, username=username)

    job_title = job_point["payload"].get("title", job.title)
    company_name = job_point["payload"].get("company", "")

    explanation = await explain_match_via_llm(job_title, company_name, required_skills, preferred_skills, chunks)

    retrieved_chunks = [
        {"section": chunk.get("section"), "text": chunk.get("text"), "score": chunk.get("score")}
        for chunk in chunks
    ]
    return explanation, retrieved_chunks


async def compute_ats_match(
    session: AsyncSession,
    username: str,
    job_id: int,
    include_explanation: bool = False,
) -> AtsMatchResult:
    """
    Compute a hybrid (vector + skill-overlap) ATS match score between a
    user's resume and one job posting. Raises ValueError if either the
    resume or job embedding is missing (caller should have already ingested
    both before checking a match).

    Set include_explanation=True to also run the RAG-based match explanation
    (retrieves resume chunks + calls Groq) and attach it to the result --
    this adds a network round-trip and LLM latency, so it's opt-in and
    defaults to off. Callers that score many jobs at once (e.g.
    get_job_recommendations) should never set this, to avoid one LLM call
    per candidate.
    """
    job = await session.get(Job, job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    required_skills, preferred_skills = await _fetch_job_required_preferred_skills(session, job_id)

    resume_point = get_resume_vector_point(username)
    if not resume_point:
        raise ValueError(f"No resume vector found for user '{username}' -- submit a resume first")

    job_point = get_job_vector_point(job_id)
    if not job_point:
        raise ValueError(f"No embedding found for job {job_id}")

    resume_skills = set(resume_point["payload"].get("skills") or [])

    vector_score, skill_score, fused_score, has_skill_data, matched_req, missing_req, matched_pref, missing_pref = (
        _score_match(resume_point["vector"], resume_skills, job_point["vector"], required_skills, preferred_skills)
    )

    explanation = None
    retrieved_chunks: list[dict[str, Any]] = []
    if include_explanation:
        explanation, retrieved_chunks = await _generate_match_explanation(
            job, job_point, username, required_skills, preferred_skills
        )

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
        explanation=explanation,
        retrieved_chunks=retrieved_chunks,
    )


async def explain_match(session: AsyncSession, username: str, job_id: int) -> dict[str, Any]:
    """
    Standalone RAG-based match explanation (no numeric score) -- see
    _generate_match_explanation for the retrieval+generation logic shared
    with compute_ats_match(..., include_explanation=True).

    Returns {"explanation": {...} or None, "retrieved_chunks": [...]}.

    Raises ValueError if the job doesn't exist or the caller has no resume.
    """
    job = await session.get(Job, job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    required_skills, preferred_skills = await _fetch_job_required_preferred_skills(session, job_id)

    job_point = get_job_vector_point(job_id)
    if not job_point:
        raise ValueError(f"No embedding found for job {job_id}")

    resume_point = get_resume_vector_point(username)
    if not resume_point:
        raise ValueError(f"No resume vector found for user '{username}' -- submit a resume first")

    explanation, retrieved_chunks = await _generate_match_explanation(
        job, job_point, username, required_skills, preferred_skills
    )

    return {"explanation": explanation, "retrieved_chunks": retrieved_chunks}


async def get_job_recommendations(session: AsyncSession, username: str, limit: int = 10) -> list[AtsMatchResult]:
    """
    Recommend jobs for a user's resume: retrieve a candidate pool via vector
    similarity search (fast, broad recall across job_postings), then rerank
    that pool by the same fused vector+skill-overlap score used for a single
    ATS check (precise, but too expensive to run against every job in the
    collection) -- so the final order reflects actual likely fit, not just
    raw semantic similarity.

    Returns results sorted by fused_score descending, highest-likelihood
    matches first. Raises ValueError if the caller has no resume yet.
    """
    resume_point = get_resume_vector_point(username)
    if not resume_point:
        raise ValueError(f"No resume vector found for user '{username}' -- submit a resume first")

    resume_skills = set(resume_point["payload"].get("skills") or [])

    candidates = search_similar_jobs(resume_point["vector"], limit=RECOMMENDATION_RETRIEVAL_LIMIT, status="ACTIVE")
    candidate_job_ids = [candidate["job_id"] for candidate in candidates]

    # Batched lookups: one Qdrant retrieve + one Postgres query for the whole
    # candidate pool, instead of two round-trips per candidate.
    job_vector_points = get_job_vector_points(candidate_job_ids)
    job_skills_by_id = await _fetch_job_skills_batch(session, candidate_job_ids)

    results: list[AtsMatchResult] = []
    for job_id in candidate_job_ids:
        job_point = job_vector_points.get(job_id)
        if not job_point:
            continue

        required_skills, preferred_skills = job_skills_by_id.get(job_id, ([], []))

        vector_score, skill_score, fused_score, has_skill_data, matched_req, missing_req, matched_pref, missing_pref = (
            _score_match(resume_point["vector"], resume_skills, job_point["vector"], required_skills, preferred_skills)
        )

        results.append(
            AtsMatchResult(
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
        )

    results.sort(key=lambda r: r.fused_score, reverse=True)
    return results[:limit]
