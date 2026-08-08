from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text
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


class JobPost(Base):
    __tablename__ = "job_posts"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    jd = Column(Text, nullable=False)
    company_name = Column(String(255), nullable=False, index=True)
    position = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)



