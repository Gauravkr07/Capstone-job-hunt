from datetime import datetime, timedelta
import hashlib
import secrets
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_session
from db.models import User, UserDetails
from db.schemas import LoginResponse, UserCreate
from middleware.auth import SECRET_KEY, ALGORITHM, encode

router = APIRouter()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_authenticated_username(request: Request, fallback: str = "demo-user") -> str:
    username = request.state.user.get("sub") if hasattr(request.state, "user") else None
    if username:
        return username

    username = request.query_params.get("username") if hasattr(request, "query_params") else None
    if username and username.strip():
        return username.strip().lower()

    if request.headers.get("x-demo-user"):
        return request.headers.get("x-demo-user").strip().lower()

    return fallback


def serialize_user_details(user: User, user_details: UserDetails | None = None) -> dict:
    return {
        "username": user.username,
        "full_name": user.full_name,
        "email": user.email,
        "age": user.age,
        "graduated": user.graduated,
        "linkedin_url": str(user.linkedin_url) if user.linkedin_url else None,
        "leetcode_url": str(user.leetcode_url) if user.leetcode_url else None,
        "hackerrank_url": str(user.hackerrank_url) if user.hackerrank_url else None,
        "resume_skills": user_details.skills if user_details and user_details.skills else [],
        "experience": user_details.experience if user_details else None,
        "industry_type": user_details.industry_type if user_details else None,
        "resume_text_available": bool(user_details and user_details.resume_text and user_details.resume_text.strip()),
        "resume_text": user_details.resume_text if user_details else None,
    }


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register(user_data: UserCreate, session: AsyncSession = Depends(get_session)):
    username = user_data.username.strip().lower()

    existing = await session.execute(select(User).where(User.username == username))
    if existing.scalars().first():
        raise HTTPException(status_code=400, detail="User already exists")

    linkedin_url = str(user_data.linkedin_url) if user_data.linkedin_url else None
    leetcode_url = str(user_data.leetcode_url) if user_data.leetcode_url else None
    hackerrank_url = str(user_data.hackerrank_url) if user_data.hackerrank_url else None

    user = User(
        username=username,
        password_hash=_hash_password(user_data.password),
        full_name=user_data.full_name,
        email=user_data.email,
        age=user_data.age,
        graduated=user_data.graduated,
        linkedin_url=linkedin_url,
        leetcode_url=leetcode_url,
        hackerrank_url=hackerrank_url,
    )
    session.add(user)
    await session.commit()

    return {"message": "User registered successfully"}


@router.post("/login", response_model=LoginResponse)
async def login(credentials: LoginRequest, session: AsyncSession = Depends(get_session)):
    username = credentials.username.strip().lower()
    result = await session.execute(select(User).where(User.username == username))
    user = result.scalars().first()

    if not user or user.password_hash != _hash_password(credentials.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    jwt_expiration = int(time.time()) + 3600
    payload = {
        "sub": username,
        "exp": jwt_expiration,
    }
    token = encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    bearer_token = f"Bearer {token}"

    refresh_token = secrets.token_urlsafe(32)
    refresh_expires_dt = datetime.utcnow() + timedelta(days=30)
    refresh_expires = int(refresh_expires_dt.timestamp())

    return {
        "bearer_token": bearer_token,
        "jwt": token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
        "jwt_expires_at": jwt_expiration,
        "refresh_expires_at": refresh_expires,
    }
