from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_session
from db.schemas import AtsCheckRequest, AtsCheckResponse, ExplainMatchRequest, ExplainMatchResponse, JobRecommendationsResponse
from helpers.auth_context import require_username
from services.ats_service import compute_ats_match, explain_match, get_job_recommendations
from services.job_service import build_job_response, get_job_by_id

router = APIRouter(prefix="/jobs", tags=["ats"])

security = HTTPBearer()


@router.post("/atsCheck", response_model=AtsCheckResponse, dependencies=[Depends(security)])
async def check_ats_match(
    payload: AtsCheckRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Score how well the caller's stored resume matches a specific job posting.

    Hybrid match score, two independent signals:
    - vector_score: cosine similarity between the resume profile embedding and
      the job embedding (semantic/contextual fit).
    - skill_score: exact-match overlap between the resume's skills and the
      job's required/preferred skills (weighted 80/20 required vs preferred).
    fused_score = 0.4 * vector_score + 0.6 * skill_score -- unless the job has
    no extracted skills at all (has_skill_data=False), in which case
    fused_score falls back to vector_score alone rather than being diluted by
    a meaningless zero.

    Set `include_explanation: true` to also run the RAG-based match
    explanation (retrieves resume chunks + calls Groq) and attach it as
    `explanation`/`retrieved_chunks` -- this adds a network round-trip and
    LLM latency on top of the otherwise-fast numeric scoring, so it defaults
    to false. For explanation-only use without the numeric score, see
    POST /jobs/explainMatch instead.

    Requires the caller to have already submitted a resume (POST /jobs/resume
    or /jobs/resume/upload) and the target job to already be ingested.
    """
    username = require_username(request)

    try:
        result = await compute_ats_match(
            session, username, payload.job_id, include_explanation=payload.include_explanation
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "job_id": result.job_id,
        "vector_score": result.vector_score,
        "skill_score": result.skill_score,
        "fused_score": result.fused_score,
        "has_skill_data": result.has_skill_data,
        "matched_required_skills": result.matched_required_skills,
        "missing_required_skills": result.missing_required_skills,
        "matched_preferred_skills": result.matched_preferred_skills,
        "missing_preferred_skills": result.missing_preferred_skills,
        "explanation": result.explanation,
        "retrieved_chunks": result.retrieved_chunks,
    }


@router.get("/recommendations", response_model=JobRecommendationsResponse, dependencies=[Depends(security)])
async def get_recommendations(
    request: Request,
    limit: int = Query(default=10, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
):
    """
    Recommend active job postings for the caller's resume, ordered by actual
    likely fit -- highest selection-likelihood first.

    Two-stage retrieval + rerank:
    1. Retrieval: vector similarity search across job_postings (broad recall,
       cheap) pulls a wider candidate pool than what's returned.
    2. Rerank: each candidate is rescored with the same hybrid
       vector+skill-overlap formula used by /jobs/atsCheck (precise, but too
       expensive to run against the whole collection), then sorted by
       fused_score descending.

    Requires the caller to have already submitted a resume.
    """
    username = require_username(request)

    try:
        results = await get_job_recommendations(session, username, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    recommendations = []
    for result in results:
        job = await get_job_by_id(session, result.job_id)
        if not job:
            continue

        job_response = await build_job_response(session, job)
        recommendations.append(
            {
                "job": job_response,
                "vector_score": result.vector_score,
                "skill_score": result.skill_score,
                "fused_score": result.fused_score,
                "has_skill_data": result.has_skill_data,
                "matched_required_skills": result.matched_required_skills,
                "missing_required_skills": result.missing_required_skills,
                "matched_preferred_skills": result.matched_preferred_skills,
                "missing_preferred_skills": result.missing_preferred_skills,
            }
        )

    return {"recommendations": recommendations}


@router.post("/explainMatch", response_model=ExplainMatchResponse, dependencies=[Depends(security)])
async def explain_job_match(
    payload: ExplainMatchRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Retrieval-augmented explanation of why the caller's resume matches a
    specific job -- genuine RAG, not just retrieval-for-ranking like
    /jobs/atsCheck and /jobs/recommendations.

    1. Retrieve: vector search over the caller's resume_chunks, scoped to
       this job's embedding, pulling the most relevant excerpts (e.g. the
       experience section that mentions the exact required skills).
    2. Generate: Groq is given only those retrieved excerpts plus the job's
       required/preferred skills, and asked to write a grounded explanation
       -- summary, supporting evidence, and gaps -- without inventing
       anything not present in the excerpts.

    `explanation` is null if no GROQ_API_KEY is configured or the LLM call
    fails; `retrieved_chunks` (the raw evidence) is still returned either way.

    Requires the caller to have already submitted a resume and the target
    job to already be ingested.
    """
    username = require_username(request)

    try:
        result = await explain_match(session, username, payload.job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return result
