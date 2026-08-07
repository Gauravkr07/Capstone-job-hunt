from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
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


class JobPost(Base):
    __tablename__ = "job_posts"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    jd = Column(Text, nullable=False)
    company_name = Column(String(255), nullable=False, index=True)
    position = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)



