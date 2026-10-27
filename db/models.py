from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    username = Column(String(150), nullable=False, primary_key=True, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True, index=True)
    age = Column(Integer, nullable=False)
    graduated = Column(Boolean, nullable=False, default=False)
    linkedin_url = Column(String(255), nullable=True)
    leetcode_url = Column(String(255), nullable=True)
    hackerrank_url = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=True, default=datetime.utcnow)


class UserDetails(Base):
    """The user's CV profile -- exactly one row per user, fully replaced on every CV upload."""

    __tablename__ = "user_details"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String(150), ForeignKey("users.username"), nullable=False, unique=True, index=True)
    resume_text = Column(Text, nullable=True)
    skills = Column(JSON, nullable=True)
    experience = Column(Integer, nullable=True)
    industry_type = Column(String(255), nullable=True)
    resume_source = Column(String(20), nullable=True)  # "file" | "text" | "url"
    resume_filename = Column(String(500), nullable=True)  # file name or URL it came from
    vector = Column(JSON, nullable=True)
    embedding_model = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class ResumeChunk(Base):
    __tablename__ = "resume_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String(150), ForeignKey("users.username"), nullable=False, index=True)
    section = Column(String(50), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding_model = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)

    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=False)
    cleaned_description = Column(Text, nullable=True)

    source = Column(String(100), nullable=True)
    source_url = Column(String(1000), nullable=True)
    external_job_id = Column(String(255), nullable=True, index=True)

    location = Column(JSON, nullable=True)
    remote_type = Column(String(50), nullable=True)
    employment_type = Column(String(50), nullable=True)

    experience_min = Column(Integer, nullable=True)
    experience_max = Column(Integer, nullable=True)

    salary_min = Column(Integer, nullable=True)
    salary_max = Column(Integer, nullable=True)
    salary_currency = Column(String(10), nullable=True)

    industry_type = Column(String(255), nullable=True)
    role = Column(String(255), nullable=True)
    posted_by = Column(String(150), ForeignKey("users.username"), nullable=True, index=True)

    status = Column(String(20), nullable=False, default="ACTIVE", index=True)
    content_hash = Column(String(64), nullable=False, unique=True, index=True)

    embedding_model = Column(String(100), nullable=True)

    posted_at = Column(DateTime, nullable=True)
    scraped_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Skill(Base):
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    name = Column(String(150), nullable=False, unique=True, index=True)
    category = Column(String(100), nullable=True, index=True)


class JobSkill(Base):
    __tablename__ = "job_skills"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    skill_id = Column(Integer, ForeignKey("skills.id"), nullable=False, index=True)
    importance = Column(String(20), nullable=False, default="required")

    __table_args__ = (UniqueConstraint("job_id", "skill_id", name="uq_job_skill"),)


class JobMatch(Base):
    """
    Stored recommendation: how well one user's CV matches one job. Recomputed
    for a user whenever their CV changes and for every user when a job is
    added, so reading recommendations is a single indexed query.
    """

    __tablename__ = "job_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(150), ForeignKey("users.username", ondelete="CASCADE"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    vector_score = Column(Float, nullable=False)
    skill_score = Column(Float, nullable=False)
    fused_score = Column(Float, nullable=False)
    has_skill_data = Column(Boolean, nullable=False, default=True)
    matched_required_skills = Column(JSON, nullable=False, default=list)
    missing_required_skills = Column(JSON, nullable=False, default=list)
    matched_preferred_skills = Column(JSON, nullable=False, default=list)
    missing_preferred_skills = Column(JSON, nullable=False, default=list)
    computed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("username", "job_id", name="uq_job_match_user_job"),
        Index("ix_job_matches_user_score", "username", "fused_score"),
    )


class ChatMessage(Base):
    """Every chatbot turn, so the candidate can review who talked to their assistant."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate = Column(String(150), nullable=True, index=True)  # whose chatbot this was
    conversation_id = Column(String(64), nullable=False, index=True)
    visitor = Column(String(150), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class ChatLead(Base):
    """Contact details a visitor (e.g. a recruiter) left in the chat."""

    __tablename__ = "chat_leads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate = Column(String(150), nullable=True, index=True)  # whose chatbot received it
    conversation_id = Column(String(64), nullable=False, index=True)
    visitor = Column(String(150), nullable=False)
    email = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
