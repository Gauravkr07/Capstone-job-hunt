from datetime import datetime
from pydantic import BaseModel, HttpUrl, constr
from typing import Optional


# PasswordStr = constr(min_length=8, max_length=64, regex=r"^[a-zA-Z0-9]+$")


class JobCreate(BaseModel):
    jd: str
    company_name: str
    position: str


class JobResponse(BaseModel):
    id: int
    jd: str
    company_name: str
    position: str
    created_at: datetime


class UserCreate(BaseModel):
    username: str
    password: str
    age: int
    graduated: bool
    linkedin_url: Optional[HttpUrl] = None
    leetcode_url: Optional[HttpUrl] = None
    hackerrank_url: Optional[HttpUrl] = None


class UserResponse(BaseModel):
    username: str
    age: int
    graduated: bool
    linkedin_url: Optional[HttpUrl] = None
    leetcode_url: Optional[HttpUrl] = None
    hackerrank_url: Optional[HttpUrl] = None

    # class Config:
    #     orm_mode = True


class LoginResponse(BaseModel):
    bearer_token: str
    jwt: str
    token_type: str = "bearer"
    refresh_token: str
    jwt_expires_at: int
    refresh_expires_at: int
