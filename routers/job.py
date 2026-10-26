import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_session
from db.schemas import JobCreate, JobListResponse, JobRawTextCreate, JobResponse
from helpers.jd_scraper import parse_raw_job_posting
from helpers.job_board_fetch import fetch_and_filter
from helpers.logger import get_logger
from services.job_service import build_job_response, build_job_responses, ingest_job, list_jobs

logger = get_logger("job_router")

router = APIRouter(prefix="/jobs", tags=["job-posting"])

security = HTTPBearer()


@router.post("/exportJobDesc", response_model=JobResponse, dependencies=[Depends(security)])
async def insert_job_description(job_data: JobCreate, session: AsyncSession = Depends(get_session)):
    """
    Ingest a job posting from structured fields (jd, company_name, position, etc.).

    - Input: JobCreate -- caller supplies title/company/location explicitly.
    - Skills/industry/experience/remote-type/employment-type are extracted from
      `jd` via regex/keyword matching (helpers.jd_scraper.scrape_job_description).
    - Deduplicates by content hash: if an identical job already exists, returns
      the existing row with is_duplicate=True instead of inserting a new one.
    - Persists Company (get-or-create), Job, and JobSkill rows to Postgres, and
      upserts a compact embedding + payload to the Qdrant job_postings collection.
    """
    job, is_duplicate = await ingest_job(
        session,
        title=job_data.position.strip(),
        company_name=job_data.company_name.strip(),
        jd_text=job_data.jd.strip(),
        location=[loc.strip() for loc in job_data.location if loc.strip()] if job_data.location else None,
        source=job_data.source,
        source_url=job_data.source_url,
        external_job_id=job_data.external_job_id,
        employment_type_override=job_data.employment_type,
        salary_min=job_data.salary_min,
        salary_max=job_data.salary_max,
        salary_currency=job_data.salary_currency,
    )
    return await build_job_response(session, job, is_duplicate)


def _require_text(value: Any | None, field_name: str) -> str:
    if value is None:
        raise HTTPException(status_code=400, detail=f"Missing {field_name}")

    text = str(value).strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"Missing {field_name}")

    return text


@router.post("/scrapeJobPosting", response_model=JobResponse, dependencies=[Depends(security)])
async def scrape_job_posting(job_data: JobRawTextCreate, session: AsyncSession = Depends(get_session)):
    """
    Ingest a job posting from raw, unstructured text (e.g. pasted from a job
    board) -- no separate title/company/location fields required.

    - Input: JobRawTextCreate -- just `raw_text` (+ optional source metadata).
    - Extraction order: tries Groq LLM structured extraction first
      (helpers.llm_jd_extractor.extract_job_fields_via_llm); if no GROQ_API_KEY
      is configured or the call fails, falls back to regex/keyword heuristics
      (helpers.jd_scraper.parse_raw_job_posting) that scan for label lines
      ("Title:", "Company:", "Location:"), "<Company> is a/hiring..." phrasing,
      and "hiring a <Title> to/for..." phrasing.
    - Raises 400 if title or company_name cannot be determined by either path.
    - Same dedup/persist/embed pipeline as /exportJobDesc from that point on.
    """
    try:
        parsed = await parse_raw_job_posting(job_data.raw_text.strip())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    title = _require_text(parsed.get("title"), "title")
    company_name = _require_text(parsed.get("company_name"), "company_name")

    job, is_duplicate = await ingest_job(
        session,
        title=title,
        company_name=company_name,
        jd_text=job_data.raw_text.strip(),
        location=parsed.get("location", None),
        source=job_data.source,
        source_url=job_data.source_url,
        external_job_id=job_data.external_job_id,
        extracted=parsed,
    )
    return await build_job_response(session, job, is_duplicate)


@router.post("/fetchFromBoards", dependencies=[Depends(security)])
async def fetch_jobs_from_boards(
    limit: int = Query(default=25, ge=1, le=200, description="Max postings to ingest this run, as a cost/time guard"),
    session: AsyncSession = Depends(get_session),
):
    """
    Pull live postings from public ATS boards (Greenhouse/Lever/Ashby -- see
    helpers.job_board_fetch.DEFAULT_COMPANIES), run them through the same
    deterministic title/location/freshness prefilter as the standalone
    job-search agent this was adapted from, then ingest the survivors through
    the existing extract/dedupe/embed pipeline (same as /exportJobDesc).

    Never raises on a single bad posting or a dead board slug -- both are
    logged and skipped so one failure can't sink the whole batch.
    """
    # Sequential HTTP to ~20 boards takes several seconds -- keep it off the event loop.
    candidates = (await asyncio.to_thread(fetch_and_filter))[:limit]

    ingested, duplicates, failed = 0, 0, 0
    for job in candidates:
        try:
            _, is_duplicate = await ingest_job(
                session,
                title=job.title,
                company_name=job.company,
                jd_text=job.description,
                location=[job.location] if job.location else None,
                source=job.ats,
                source_url=job.url,
                external_job_id=job.job_id,
            )
            duplicates += is_duplicate
            ingested += not is_duplicate
        except Exception as exc:
            # A failed flush/commit leaves the session unusable until rolled back,
            # which would otherwise make every later posting in the batch fail too.
            await session.rollback()
            failed += 1
            logger.warning("Failed to ingest %s (%s): %s", job.job_id, job.title, exc)

    return {
        "fetched_and_filtered": len(candidates),
        "ingested": ingested,
        "duplicates": duplicates,
        "failed": failed,
    }


@router.get("", response_model=JobListResponse, dependencies=[Depends(security)])
async def get_jobs(
    status: str | None = Query(default=None, description="Filter by job status, e.g. ACTIVE"),
    industry_type: str | None = Query(default=None, description="Filter by industry, e.g. fintech"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """
    List job postings from Postgres, newest first, with optional status/industry
    filters and pagination.
    """
    jobs, total = await list_jobs(session, status=status, industry_type=industry_type, limit=limit, offset=offset)
    return {"total": total, "limit": limit, "offset": offset, "jobs": await build_job_responses(session, jobs)}

