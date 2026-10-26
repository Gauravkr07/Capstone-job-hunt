import asyncio

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Company, Job, JobSkill, Skill
from db.vector_store import upsert_job_vector
from helpers.embeddings import EMBEDDING_MODEL_NAME, embed_text
from helpers.jd_scraper import scrape_job_description


def _apply_job_filters(query, status: str | None, industry_type: str | None):
    if status:
        query = query.where(Job.status == status)
    if industry_type:
        query = query.where(Job.industry_type == industry_type)
    return query


async def list_jobs(
    session: AsyncSession,
    status: str | None = None,
    industry_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[Job], int]:
    """Returns (jobs, total_matching_count) for pagination."""
    count_query = _apply_job_filters(select(func.count(Job.id)), status, industry_type)
    total = (await session.execute(count_query)).scalar_one()

    query = _apply_job_filters(select(Job), status, industry_type)
    query = query.order_by(Job.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(query)

    return list(result.scalars().all()), total


async def get_or_create_company(session: AsyncSession, company_name: str) -> Company:
    existing = await session.execute(select(Company).where(Company.name == company_name))
    company = existing.scalars().first()
    if company:
        return company

    company = Company(name=company_name)
    session.add(company)
    await session.flush()
    return company


async def get_or_create_skills(session: AsyncSession, names: list[str]) -> dict[str, Skill]:
    """Resolve many skill names in one SELECT (+ one flush for any new ones), instead of a query per skill."""
    if not names:
        return {}

    existing = await session.execute(select(Skill).where(Skill.name.in_(names)))
    skills = {skill.name: skill for skill in existing.scalars()}

    new_skills = [Skill(name=name) for name in names if name not in skills]
    if new_skills:
        session.add_all(new_skills)
        await session.flush()
        skills.update({skill.name: skill for skill in new_skills})
    return skills


async def fetch_job_skills(session: AsyncSession, job_ids: list[int]) -> dict[int, tuple[list[str], list[str]]]:
    """
    One Postgres round-trip for many jobs' skills. Returns
    {job_id: (required_skills, preferred_skills)} with an entry for every job_id.
    """
    result: dict[int, tuple[list[str], list[str]]] = {job_id: ([], []) for job_id in job_ids}
    if not job_ids:
        return result

    rows = await session.execute(
        select(JobSkill.job_id, JobSkill.importance, Skill.name).join(Skill).where(JobSkill.job_id.in_(job_ids))
    )
    for job_id, importance, skill_name in rows.all():
        required, preferred = result[job_id]
        (preferred if importance == "preferred" else required).append(skill_name)
    return result


def index_job_vector(job: Job, company_name: str, skills: list[str], cleaned_description: str | None) -> None:
    """Embed one job and upsert it into Qdrant. Shared by ingestion and the Postgres -> Qdrant reindex."""
    embedding_text = "\n".join(
        part
        for part in [
            f"Title: {job.title}",
            f"Company: {company_name}",
            f"Skills: {', '.join(skills)}" if skills else "",
            f"Industry: {job.industry_type}" if job.industry_type else "",
            cleaned_description or "",
        ]
        if part
    )
    upsert_job_vector(
        job.id,
        embed_text(embedding_text),
        {
            "title": job.title,
            "company": company_name,
            "location": job.location,
            "remote_type": job.remote_type,
            "experience_min": job.experience_min,
            "experience_max": job.experience_max,
            "skills": skills,
            "industry_type": job.industry_type,
            "status": job.status,
        },
    )


async def ingest_job(
    session: AsyncSession,
    title: str,
    company_name: str,
    jd_text: str,
    location: list[str] | None = None,
    source: str | None = None,
    source_url: str | None = None,
    external_job_id: str | None = None,
    employment_type_override: str | None = None,
    salary_min: int | None = None,
    salary_max: int | None = None,
    salary_currency: str | None = None,
    extracted: dict | None = None,
) -> tuple[Job, bool]:
    """
    Deduplicate by content hash, persist structured fields + required/preferred
    skills to Postgres, and embed the job into Qdrant.

    If `extracted` is not provided, runs the regex/keyword-based
    scrape_job_description pipeline on jd_text. Pass a pre-computed `extracted`
    dict (e.g. from LLM-based extraction) to skip that and use it as-is.

    Returns (job, is_duplicate). When a duplicate is found, the existing job
    row is returned unchanged rather than creating a new one.
    """
    title = title[:255]
    company_name = company_name[:255]

    if extracted is None:
        try:
            extracted = scrape_job_description(title, company_name, jd_text, location=location)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    content_hash = extracted["content_hash"]

    existing = await session.execute(select(Job).where(Job.content_hash == content_hash))
    duplicate_job = existing.scalars().first()
    if duplicate_job:
        return duplicate_job, True

    company = await get_or_create_company(session, company_name)

    job = Job(
        company_id=company.id,
        title=title,
        description=jd_text,
        cleaned_description=extracted["cleaned_description"],
        source=source,
        source_url=source_url,
        external_job_id=external_job_id,
        location=location,
        remote_type=extracted["remote_type"],
        employment_type=employment_type_override or extracted["employment_type"],
        experience_min=extracted["experience_min"],
        experience_max=extracted["experience_max"],
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=salary_currency,
        industry_type=extracted["industry_type"],
        status="ACTIVE",
        content_hash=content_hash,
        embedding_model=EMBEDDING_MODEL_NAME,
    )
    session.add(job)
    await session.flush()

    # Required wins if the extractor listed a skill under both (uq_job_skill allows one row per skill).
    importance_by_skill = {name[:150]: "preferred" for name in extracted["preferred_skills"] if name}
    importance_by_skill.update({name[:150]: "required" for name in extracted["required_skills"] if name})
    skills = await get_or_create_skills(session, list(importance_by_skill))
    session.add_all(
        JobSkill(job_id=job.id, skill_id=skills[name].id, importance=importance)
        for name, importance in importance_by_skill.items()
    )

    await session.commit()
    await session.refresh(job)

    # Embedding (CPU) and the Qdrant upsert are blocking -- run off the event loop.
    await asyncio.to_thread(
        index_job_vector, job, company_name, extracted["skills"], extracted["cleaned_description"]
    )

    return job, False


def _job_response(job: Job, company_name: str, skills: tuple[list[str], list[str]], is_duplicate: bool) -> dict:
    required_skills, preferred_skills = skills
    return {
        "id": job.id,
        "title": job.title,
        "company_name": company_name,
        "description": job.description,
        "location": job.location,
        "remote_type": job.remote_type,
        "employment_type": job.employment_type,
        "experience_min": job.experience_min,
        "experience_max": job.experience_max,
        "industry_type": job.industry_type,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "status": job.status,
        "is_duplicate": is_duplicate,
        "created_at": job.created_at,
    }


async def build_job_responses(session: AsyncSession, jobs: list[Job]) -> list[dict]:
    """Serialize many jobs with 2 queries total (skills + company names), not 2 per job."""
    if not jobs:
        return []

    skills_by_job = await fetch_job_skills(session, [job.id for job in jobs])
    company_rows = await session.execute(
        select(Company.id, Company.name).where(Company.id.in_({job.company_id for job in jobs}))
    )
    company_names = dict(company_rows.all())

    return [
        _job_response(job, company_names.get(job.company_id, ""), skills_by_job[job.id], is_duplicate=False)
        for job in jobs
    ]


async def build_job_response(session: AsyncSession, job: Job, is_duplicate: bool = False) -> dict:
    skills = (await fetch_job_skills(session, [job.id]))[job.id]
    company = await session.get(Company, job.company_id)
    return _job_response(job, company.name if company else "", skills, is_duplicate)
