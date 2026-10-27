import asyncio
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ResumeChunk, User, UserDetails
from db.vector_store import delete_resume_data, upsert_resume_chunks, upsert_resume_vector
from helpers.chatbot_rag import invalidate_candidate_cache
from helpers.embeddings import EMBEDDING_MODEL_NAME, embed_profile, embed_texts
from helpers.resume_chunker import chunk_embedding_text, chunk_resume
from helpers.resume_scraper import scrape_resume
from helpers.logger import get_logger
from services.ats_service import refresh_matches_for_user

logger = get_logger("resume_service")


async def get_resume_profile(session: AsyncSession, username: str) -> UserDetails | None:
    result = await session.execute(select(UserDetails).where(UserDetails.username == username))
    return result.scalars().first()


def contact_payload(user: User | None) -> dict:
    """Basic registration details stored alongside the CV profile vector in Qdrant."""
    if not user:
        return {}
    return {
        "full_name": user.full_name,
        "email": user.email,
        "linkedin_url": user.linkedin_url,
        "leetcode_url": user.leetcode_url,
        "hackerrank_url": user.hackerrank_url,
    }


async def save_resume_profile(
    username: str,
    resume_text: str,
    session: AsyncSession,
    source: str | None = None,
    filename: str | None = None,
) -> UserDetails:
    """
    Shared pipeline for any resume ingestion path (raw text, URL, or file upload).
    A new CV fully replaces the old one everywhere:
      1. Postgres: user_details (skills, experience, industry, vector) and all
         resume_chunks rows are overwritten in one commit.
      2. Qdrant: the user's profile point (CV vector + basic details/links) and
         all CV chunk points are deleted, then re-written.
      3. job_matches: the user's stored recommendations are recomputed.
    `source` is "file" | "text" | "url"; `filename` is the file name or URL.
    """
    try:
        extracted = scrape_resume(resume_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_text = extracted.get("resume_text")
    skills = extracted.get("skills") or []
    experience = extracted.get("experience")
    industry_type = extracted.get("industry_type")
    chunks = chunk_resume(resolved_text) if resolved_text else []

    # CPU-bound: embed the profile summary and every section chunk off the event loop,
    # chunks in a single batched encode() call.
    embedding, chunk_vectors = await asyncio.to_thread(
        lambda: (embed_profile(skills, industry_type, experience), embed_texts([chunk_embedding_text(c.section, c.text) for c in chunks]))
    )

    user_details = await get_resume_profile(session, username)

    if user_details:
        user_details.resume_text = resolved_text or user_details.resume_text
        user_details.skills = skills
        user_details.experience = experience
        user_details.industry_type = industry_type
        user_details.vector = embedding
        user_details.embedding_model = EMBEDDING_MODEL_NAME
        user_details.resume_source = source
        user_details.resume_filename = filename
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
            resume_source=source,
            resume_filename=filename,
        )
        session.add(user_details)

    # A new CV fully replaces the old one: drop every old chunk, then insert the new ones.
    await session.execute(delete(ResumeChunk).where(ResumeChunk.username == username))
    chunk_payloads = []
    for idx, (chunk, chunk_vector) in enumerate(zip(chunks, chunk_vectors)):
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
        chunk_payloads.append({"section": chunk.section, "chunk_index": idx, "text": chunk.text, "vector": chunk_vector})

    # Profile + chunks committed together, so Postgres never holds a half-updated resume.
    await session.commit()
    await session.refresh(user_details)

    # Qdrant: wipe the old profile point and chunks unconditionally (even if the new CV
    # yields no vector/chunks, nothing stale may survive), then write the new data.
    await asyncio.to_thread(delete_resume_data, username)
    user = await session.get(User, username)
    await asyncio.to_thread(
        upsert_resume_vector, username, embedding, skills, industry_type, experience, contact_payload(user)
    )
    if chunk_payloads:
        await asyncio.to_thread(upsert_resume_chunks, username, chunk_payloads)

    invalidate_candidate_cache(username)
    try:
        await refresh_matches_for_user(session, username)
    except Exception as exc:  # the CV is saved either way; recommendations recompute on first read
        await session.rollback()
        logger.warning("Could not refresh job matches for %s: %s", username, exc)
    return user_details
