from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_session
from db.models import JobPost, UserDetails
from db.schemas import JobCreate, JobResponse, ResumeFetchRequest, ResumeFetchResponse
from db.vector_store import upsert_resume_vector
from helpers.embeddings import EMBEDDING_MODEL_NAME, embed_profile
from helpers.resume_scraper import scrape_resume

router = APIRouter(prefix="/jobs")

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()


@router.post("/exportJobDesc", response_model=JobResponse,dependencies=[Depends(security)])
async def insert_JobDescription(job_data: JobCreate, session: AsyncSession = Depends(get_session)):
    job = JobPost(
        jd=job_data.jd.strip(),
        company_name=job_data.company_name.strip(),
        position=job_data.position.strip(),
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    return {
        "id": job.id,
        "jd": job.jd,
        "company_name": job.company_name,
        "position": job.position,
        "created_at": job.created_at,
    }


@router.post("/resume", response_model=ResumeFetchResponse, dependencies=[Depends(security)])
async def fetch_resume_profile(
    payload: ResumeFetchRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    username = request.state.user.get("sub") if hasattr(request.state, "user") else None
    if not username:
        raise HTTPException(status_code=401, detail="User is not authenticated")

    try:
        extracted = scrape_resume(payload.resume_text, payload.resume_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    skills = extracted.get("skills") or []
    experience = extracted.get("experience")
    industry_type = extracted.get("industry_type")
    embedding = embed_profile(skills, industry_type, experience)

    existing = await session.execute(select(UserDetails).where(UserDetails.username == username))
    user_details = existing.scalars().first()

    if user_details:
        user_details.resume_text = payload.resume_text or user_details.resume_text
        user_details.skills = skills
        user_details.experience = experience
        user_details.industry_type = industry_type
        user_details.vector = embedding
        user_details.embedding_model = EMBEDDING_MODEL_NAME
        user_details.updated_at = datetime.utcnow()
    else:
        user_details = UserDetails(
            username=username,
            resume_text=payload.resume_text,
            skills=skills,
            experience=experience,
            industry_type=industry_type,
            vector=embedding,
            embedding_model=EMBEDDING_MODEL_NAME,
        )
        session.add(user_details)

    await session.commit()
    await session.refresh(user_details)

    upsert_resume_vector(
        username=username,
        vector=embedding,
        skills=skills,
        industry_type=industry_type,
        experience=experience,
    )

    return {
        "username": user_details.username,
        "skills": user_details.skills or [],
        "experience": user_details.experience,
        "industry_type": user_details.industry_type,
        "vector": user_details.vector or [],
    }


@router.get("/user", dependencies=[Depends(security)])
async def get_job(request: Request):
    return {"message": f"Details of job {request.state.user}"}


