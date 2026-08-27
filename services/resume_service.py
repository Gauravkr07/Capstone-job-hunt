from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ResumeChunk, UserDetails
from db.vector_store import upsert_resume_chunks, upsert_resume_vector
from helpers.embeddings import EMBEDDING_MODEL_NAME, embed_profile, embed_text
from helpers.resume_chunker import chunk_resume
from helpers.resume_scraper import scrape_resume


async def get_resume_profile(session: AsyncSession, username: str) -> UserDetails | None:
    result = await session.execute(select(UserDetails).where(UserDetails.username == username))
    return result.scalars().first()


async def save_resume_profile(
    username: str,
    resume_text: str,
    session: AsyncSession,
) -> UserDetails:
    """
    Shared pipeline for any resume ingestion path (raw text, URL, or file upload):
    scrape skills/experience/industry, embed the profile summary and section
    chunks, and persist both to Postgres and Qdrant.
    """
    try:
        extracted = scrape_resume(resume_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_text = extracted.get("resume_text")
    skills = extracted.get("skills") or []
    experience = extracted.get("experience")
    industry_type = extracted.get("industry_type")
    embedding = embed_profile(skills, industry_type, experience)

    existing = await session.execute(select(UserDetails).where(UserDetails.username == username))
    user_details = existing.scalars().first()

    if user_details:
        user_details.resume_text = resolved_text or user_details.resume_text
        user_details.skills = skills
        user_details.experience = experience
        user_details.industry_type = industry_type
        user_details.vector = embedding
        user_details.embedding_model = EMBEDDING_MODEL_NAME
        user_details.updated_at = datetime.utcnow()
    else:
        user_details = UserDetails(
            username=username,
            resume_text=resolved_text,
            skills=skills,
            experience=experience,
            industry_type=industry_type,
            vector=embedding,
            embedding_model=EMBEDDING_MODEL_NAME,
        )
        session.add(user_details)

    await session.commit()
    await session.refresh(user_details)

    upsert_resume_vector(
        username=username,
        vector=embedding,
        skills=skills,
        industry_type=industry_type,
        experience=experience,
    )

    if resolved_text:
        chunks = chunk_resume(resolved_text)

        await session.execute(delete(ResumeChunk).where(ResumeChunk.username == username))

        chunk_payloads = []
        for idx, chunk in enumerate(chunks):
            chunk_vector = embed_text(chunk.text)
            if not chunk_vector:
                continue

            session.add(
                ResumeChunk(
                    username=username,
                    section=chunk.section,
                    chunk_index=idx,
                    chunk_text=chunk.text,
                    embedding_model=EMBEDDING_MODEL_NAME,
                )
            )
            chunk_payloads.append(
                {
                    "section": chunk.section,
                    "chunk_index": idx,
                    "text": chunk.text,
                    "vector": chunk_vector,
                }
            )

        await session.commit()

        upsert_resume_chunks(username=username, chunks=chunk_payloads)

    return user_details
