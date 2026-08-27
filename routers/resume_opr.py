from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_session
from db.schemas import ResumeFetchRequest, ResumeFetchResponse
from db.vector_store import get_resume_chunk_points, get_resume_vector_point
from helpers.auth_context import require_username
from helpers.resume_file_parser import extract_resume_text
from services.resume_service import get_resume_profile, save_resume_profile

router = APIRouter(prefix="/jobs", tags=["resume"])

security = HTTPBearer()


@router.get("/resume/me", response_model=ResumeFetchResponse, dependencies=[Depends(security)])
async def get_my_resume_profile(request: Request, session: AsyncSession = Depends(get_session)):
    """
    Return the authenticated caller's stored resume profile from Postgres
    (skills, experience, industry, and the stored embedding vector).

    Raises 404 if the caller has never submitted a resume.
    """
    username = require_username(request)
    user_details = await get_resume_profile(session, username)

    if not user_details:
        raise HTTPException(status_code=404, detail="No resume profile found for this user")

    return {
        "username": user_details.username,
        "skills": user_details.skills or [],
        "experience": user_details.experience,
        "industry_type": user_details.industry_type,
        "vector": user_details.vector or [],
    }


@router.get("/resume/me/qdrant", dependencies=[Depends(security)])
async def get_my_resume_qdrant_points(request: Request):
    """
    Debug endpoint: return the caller's raw Qdrant points directly (not via
    similarity search) -- the resume_profiles point and all resume_chunks
    points, each with its stored vector and payload.
    """
    username = require_username(request)

    profile_point = get_resume_vector_point(username)
    chunk_points = get_resume_chunk_points(username)

    return {
        "username": username,
        "profile_point": profile_point,
        "chunk_points": chunk_points,
    }


@router.post("/resume", response_model=ResumeFetchResponse, dependencies=[Depends(security)])
async def fetch_resume_profile(
    payload: ResumeFetchRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Ingest a resume from raw text or a URL (JSON body, not a file upload).

    - Input: ResumeFetchRequest -- exactly one of `resume_text` or `resume_url`.
    - If `resume_url` is given, the URL is fetched and its bytes are run through
      helpers.resume_file_parser.extract_resume_text (detects PDF/DOCX/plain
      text by content-type or extension).
    - Delegates to services.resume_service.save_resume_profile for the shared
      pipeline: scrape skills/experience/industry, embed the profile summary,
      chunk + embed resume sections, persist to Postgres and Qdrant.
    - Overwrites the caller's existing UserDetails row if one exists (one
      resume profile per user, always reflecting the latest submission).
    """
    username = require_username(request)

    try:
        resume_text = payload.resume_text
        if not resume_text and payload.resume_url:
            import urllib.request

            with urllib.request.urlopen(payload.resume_url, timeout=10) as response:
                content = response.read()
                content_type = response.headers.get("Content-Type")
                resume_text = extract_resume_text(content, filename=payload.resume_url, content_type=content_type)

        if not resume_text or not resume_text.strip():
            raise ValueError("A resume body or resume_url is required")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user_details = await save_resume_profile(username, resume_text, session)

    return {
        "username": user_details.username,
        "skills": user_details.skills or [],
        "experience": user_details.experience,
        "industry_type": user_details.industry_type,
        "vector": user_details.vector or [],
    }


@router.post("/resume/upload", response_model=ResumeFetchResponse, dependencies=[Depends(security)])
async def upload_resume_file(
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """
    Ingest a resume from a directly uploaded file (multipart/form-data).

    - Input: UploadFile -- PDF, DOCX, or plain text, detected by filename
      extension or content-type via helpers.resume_file_parser.extract_resume_text.
    - Raises 400 if the file is empty or no text could be extracted from it.
    - Same shared pipeline as /resume from that point on
      (services.resume_service.save_resume_profile).
    """
    username = require_username(request)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        resume_text = extract_resume_text(content, filename=file.filename, content_type=file.content_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read resume file: {exc}") from exc

    if not resume_text or not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the uploaded file")

    user_details = await save_resume_profile(username, resume_text, session)

    return {
        "username": user_details.username,
        "skills": user_details.skills or [],
        "experience": user_details.experience,
        "industry_type": user_details.industry_type,
        "vector": user_details.vector or [],
    }
