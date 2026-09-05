from datetime import datetime
from pydantic import BaseModel, HttpUrl
from typing import Optional


class ResumeFetchRequest(BaseModel):
    resume_text: Optional[str] = None
    resume_url: Optional[str] = None


class ResumeFetchResponse(BaseModel):
    username: str
    skills: list[str]
    experience: Optional[int] = None
    industry_type: Optional[str] = None
    vector: Optional[list[float]] = None


class JobCreate(BaseModel):
    jd: str
    company_name: str
    position: str
    location: Optional[list[str]] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    external_job_id: Optional[str] = None
    employment_type: Optional[str] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    salary_currency: Optional[str] = None


class JobRawTextCreate(BaseModel):
    raw_text: str
    source: Optional[str] = None
    source_url: Optional[str] = None
    external_job_id: Optional[str] = None


class JobResponse(BaseModel):
    id: int
    title: str
    company_name: str
    description: str
    location: Optional[list[str]] = None
    remote_type: Optional[str] = None
    employment_type: Optional[str] = None
    experience_min: Optional[int] = None
    experience_max: Optional[int] = None
    industry_type: Optional[str] = None
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    status: str
    is_duplicate: bool = False
    created_at: datetime


class MatchExplanation(BaseModel):
    summary: Optional[str] = None
    supporting_points: list[str] = []
    gaps: list[str] = []


class RetrievedChunk(BaseModel):
    section: Optional[str] = None
    text: Optional[str] = None
    score: Optional[float] = None


class AtsCheckRequest(BaseModel):
    job_id: int
    include_explanation: bool = False


class AtsCheckResponse(BaseModel):
    job_id: int
    vector_score: float
    skill_score: float
    fused_score: float
    has_skill_data: bool
    matched_required_skills: list[str]
    missing_required_skills: list[str]
    matched_preferred_skills: list[str]
    missing_preferred_skills: list[str]
    explanation: Optional[MatchExplanation] = None
    retrieved_chunks: list[RetrievedChunk] = []


class ExplainMatchRequest(BaseModel):
    job_id: int


class ExplainMatchResponse(BaseModel):
    explanation: Optional[MatchExplanation] = None
    retrieved_chunks: list[RetrievedChunk] = []


class JobListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    jobs: list[JobResponse]


class JobRecommendation(BaseModel):
    job: JobResponse
    vector_score: float
    skill_score: float
    fused_score: float
    has_skill_data: bool
    matched_required_skills: list[str]
    missing_required_skills: list[str]
    matched_preferred_skills: list[str]
    missing_preferred_skills: list[str]


class JobRecommendationsResponse(BaseModel):
    recommendations: list[JobRecommendation]


class UserCreate(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    age: int
    graduated: bool
    linkedin_url: Optional[HttpUrl] = None
    leetcode_url: Optional[HttpUrl] = None
    hackerrank_url: Optional[HttpUrl] = None


class LoginResponse(BaseModel):
    bearer_token: str
    jwt: str
    token_type: str = "bearer"
    refresh_token: str
    jwt_expires_at: int
    refresh_expires_at: int
