from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Company, Job, JobSkill, Skill
from db.vector_store import upsert_job_vector
from helpers.embeddings import EMBEDDING_MODEL_NAME, embed_text
from helpers.jd_scraper import scrape_job_description


async def get_job_by_id(session: AsyncSession, job_id: int) -> Job | None:
    return await session.get(Job, job_id)


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


async def get_or_create_skill(session: AsyncSession, name: str) -> Skill:
    existing = await session.execute(select(Skill).where(Skill.name == name))
    skill = existing.scalars().first()
    if skill:
        return skill

    skill = Skill(name=name)
    session.add(skill)
    await session.flush()
    return skill


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

    for skill_name in extracted["required_skills"]:
        skill = await get_or_create_skill(session, skill_name)
        session.add(JobSkill(job_id=job.id, skill_id=skill.id, importance="required"))

    for skill_name in extracted["preferred_skills"]:
        skill = await get_or_create_skill(session, skill_name)
        session.add(JobSkill(job_id=job.id, skill_id=skill.id, importance="preferred"))

    await session.commit()
    await session.refresh(job)

    embedding_text = "\n".join(
        part
        for part in [
            f"Title: {title}",
            f"Company: {company_name}",
            f"Skills: {', '.join(extracted['skills'])}" if extracted["skills"] else "",
            f"Industry: {extracted['industry_type']}" if extracted["industry_type"] else "",
            extracted["cleaned_description"],
        ]
        if part
    )
    embedding = embed_text(embedding_text)

    upsert_job_vector(
        job_id=job.id,
        vector=embedding,
        payload={
            "title": title,
            "company": company_name,
            "location": location,
            "remote_type": extracted["remote_type"],
            "experience_min": extracted["experience_min"],
            "experience_max": extracted["experience_max"],
            "skills": extracted["skills"],
            "industry_type": extracted["industry_type"],
            "status": job.status,
        },
    )

    return job, False


async def build_job_response(session: AsyncSession, job: Job, is_duplicate: bool = False) -> dict:
    skill_rows = await session.execute(select(JobSkill, Skill).join(Skill).where(JobSkill.job_id == job.id))
    required_skills = []
    preferred_skills = []
    for job_skill, skill in skill_rows.all():
        if job_skill.importance == "preferred":
            preferred_skills.append(skill.name)
        else:
            required_skills.append(skill.name)

    company = await session.get(Company, job.company_id)

    return {
        "id": job.id,
        "title": job.title,
        "company_name": company.name if company else "",
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
