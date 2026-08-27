from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    username = Column(String(150), nullable=False, primary_key=True, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    age = Column(Integer, nullable=False)
    graduated = Column(Boolean, nullable=False, default=False)
    linkedin_url = Column(String(255), nullable=True)
    leetcode_url = Column(String(255), nullable=True)
    hackerrank_url = Column(String(255), nullable=True)


class UserDetails(Base):
    __tablename__ = "user_details"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String(150), ForeignKey("users.username"), nullable=False, unique=True, index=True)
    resume_text = Column(Text, nullable=True)
    skills = Column(JSON, nullable=True)
    experience = Column(Integer, nullable=True)
    industry_type = Column(String(255), nullable=True)
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



