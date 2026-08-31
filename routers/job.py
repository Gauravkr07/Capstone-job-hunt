from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_session
from db.schemas import JobCreate, JobListResponse, JobRawTextCreate, JobResponse
from helpers.jd_scraper import parse_raw_job_posting
from services.job_service import build_job_response, ingest_job, list_jobs

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

    job, is_duplicate = await ingest_job(
        session,
        title=parsed.get("title", None), # type: ignore
        company_name=parsed.get("company_name", None), # type: ignore
        jd_text=job_data.raw_text.strip(),
        location=parsed.get("location", None),
        source=job_data.source,
        source_url=job_data.source_url,
        external_job_id=job_data.external_job_id,
        extracted=parsed,
    )
    return await build_job_response(session, job, is_duplicate)


@router.get("/user", dependencies=[Depends(security)])
async def get_job(request: Request):
    """Debug endpoint: echoes the authenticated JWT payload for the caller. Not job-related; kept for auth troubleshooting."""
    return {"message": f"Details of job {request.state.user}"}


@router.get("", response_model=JobListResponse,dependencies=[Depends(security)])
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
    job_responses = [await build_job_response(session, job) for job in jobs]

    return {"total": total, "limit": limit, "offset": offset, "jobs": job_responses}


