import hashlib
import hmac
import secrets
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_session
from db.models import User
from db.schemas import LoginResponse, UserCreate
from middleware.auth import SECRET_KEY, ALGORITHM, encode

router = APIRouter()

JWT_TTL_SECONDS = 3600
REFRESH_TTL_SECONDS = 30 * 24 * 3600


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register(user_data: UserCreate, session: AsyncSession = Depends(get_session)):
    username = user_data.username.strip().lower()

    if await session.get(User, username):
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
    user = await session.get(User, username)

    if not user or not hmac.compare_digest(user.password_hash, _hash_password(credentials.password)):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    now = int(time.time())
    jwt_expiration = now + JWT_TTL_SECONDS
    payload = {
        "sub": username,
        "exp": jwt_expiration,
    }
    token = encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "bearer_token": f"Bearer {token}",
        "jwt": token,
        "token_type": "bearer",
        "refresh_token": secrets.token_urlsafe(32),
        "jwt_expires_at": jwt_expiration,
        "refresh_expires_at": now + REFRESH_TTL_SECONDS,
    }
