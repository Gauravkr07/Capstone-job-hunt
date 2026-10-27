"""
Rebuild every Qdrant collection from Postgres (the source of truth), e.g.
after pointing QDRANT_URL at a new or emptied cluster.

    python -m services.reindex            # locally
    docker compose exec fastapi python -m services.reindex

- resume_profiles: re-upserted from the vector already stored in user_details.
- resume_chunks: every stored CV (user_details.resume_text) is re-chunked with
  the current chunker, so chunking improvements apply to existing CVs, then
  the rows in Postgres and the points in Qdrant are replaced.
- job_postings: every job re-embedded from its stored fields and skills.
- job_matches: every user's stored recommendations are recomputed.
Safe to re-run: point ids are deterministic, so existing points are overwritten.
"""
import asyncio

from sqlalchemy import delete, select

from db.connection import AsyncSessionLocal
from db.models import Company, Job, ResumeChunk, User, UserDetails
from db.vector_store import delete_resume_chunks, upsert_resume_chunks, upsert_resume_vector
from helpers.chatbot_rag import invalidate_candidate_cache
from helpers.embeddings import EMBEDDING_MODEL_NAME, embed_profile, embed_texts
from helpers.logger import get_logger
from helpers.resume_chunker import chunk_embedding_text, chunk_resume
from services.ats_service import refresh_matches_for_user
from services.job_service import fetch_job_skills, index_job_vector
from services.resume_service import contact_payload

logger = get_logger("reindex")


async def reindex_all() -> dict[str, int]:
    async with AsyncSessionLocal() as session:
        profiles = (await session.execute(select(UserDetails))).scalars().all()
        users = {u.username: u for u in (await session.execute(select(User))).scalars()}
        jobs = (await session.execute(select(Job))).scalars().all()
        companies = dict((await session.execute(select(Company.id, Company.name))).all())
        job_skills = await fetch_job_skills(session, [job.id for job in jobs])

    for profile in profiles:
        vector = profile.vector or await asyncio.to_thread(
            embed_profile, profile.skills or [], profile.industry_type, profile.experience
        )
        await asyncio.to_thread(
            upsert_resume_vector,
            profile.username,
            vector,
            profile.skills or [],
            profile.industry_type,
            profile.experience,
            contact_payload(users.get(profile.username)),
        )

    chunk_count = 0
    for profile in profiles:
        if not profile.resume_text:
            continue
        chunks = chunk_resume(profile.resume_text)
        vectors = await asyncio.to_thread(embed_texts, [chunk_embedding_text(c.section, c.text) for c in chunks])
        async with AsyncSessionLocal() as session:
            await session.execute(delete(ResumeChunk).where(ResumeChunk.username == profile.username))
            session.add_all(
                ResumeChunk(
                    username=profile.username,
                    section=c.section,
                    chunk_index=i,
                    chunk_text=c.text,
                    embedding_model=EMBEDDING_MODEL_NAME,
                )
                for i, c in enumerate(chunks)
            )
            await session.commit()
        await asyncio.to_thread(delete_resume_chunks, profile.username)
        await asyncio.to_thread(
            upsert_resume_chunks,
            profile.username,
            [
                {"section": c.section, "chunk_index": i, "text": c.text, "vector": v}
                for i, (c, v) in enumerate(zip(chunks, vectors))
                if v
            ],
        )
        invalidate_candidate_cache(profile.username)
        chunk_count += len(chunks)

    for job in jobs:
        required, preferred = job_skills[job.id]
        await asyncio.to_thread(
            index_job_vector, job, companies.get(job.company_id, ""), sorted(set(required + preferred)), job.cleaned_description
        )

    match_count = 0
    async with AsyncSessionLocal() as session:
        for profile in profiles:
            match_count += await refresh_matches_for_user(session, profile.username)

    counts = {
        "resume_profiles": len(profiles),
        "resume_chunks": chunk_count,
        "job_postings": len(jobs),
        "job_matches": match_count,
    }
    logger.info("Reindexed Qdrant from Postgres: %s", counts)
    return counts


if __name__ == "__main__":
    print(asyncio.run(reindex_all()))
