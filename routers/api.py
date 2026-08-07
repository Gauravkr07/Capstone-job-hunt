from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_session
from db.models import JobPost
from db.schemas import JobCreate, JobResponse

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


@router.get("/user", dependencies=[Depends(security)])
async def get_job(request: Request):
    return {"message": f"Details of job {request.state.user}"}


