"""
Rebuild every Qdrant collection from Postgres (the source of truth), e.g.
after pointing QDRANT_URL at a new or emptied cluster.

    python -m services.reindex            # locally
    docker compose exec fastapi python -m services.reindex

- resume_profiles: re-upserted from the vector already stored in user_details.
- resume_chunks: CV sections re-embedded from resume_chunks.chunk_text.
- job_postings: every job re-embedded from its stored fields and skills.
Safe to re-run: point ids are deterministic, so existing points are overwritten.
"""
import asyncio
from collections import defaultdict

from sqlalchemy import select

from db.connection import AsyncSessionLocal
from db.models import Company, Job, ResumeChunk, UserDetails
from db.vector_store import delete_resume_chunks, upsert_resume_chunks, upsert_resume_vector
from helpers.embeddings import embed_profile, embed_texts
from helpers.logger import get_logger
from services.job_service import fetch_job_skills, index_job_vector

logger = get_logger("reindex")


async def reindex_all() -> dict[str, int]:
    async with AsyncSessionLocal() as session:
        profiles = (await session.execute(select(UserDetails))).scalars().all()
        chunks = (await session.execute(select(ResumeChunk).order_by(ResumeChunk.username, ResumeChunk.chunk_index))).scalars().all()
        jobs = (await session.execute(select(Job))).scalars().all()
        companies = dict((await session.execute(select(Company.id, Company.name))).all())
        job_skills = await fetch_job_skills(session, [job.id for job in jobs])

    for profile in profiles:
        vector = profile.vector or await asyncio.to_thread(
            embed_profile, profile.skills or [], profile.industry_type, profile.experience
        )
        await asyncio.to_thread(
            upsert_resume_vector, profile.username, vector, profile.skills or [], profile.industry_type, profile.experience
        )

    chunks_by_user: dict[str, list[ResumeChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_user[chunk.username].append(chunk)
    for username, user_chunks in chunks_by_user.items():
        vectors = await asyncio.to_thread(embed_texts, [c.chunk_text for c in user_chunks])
        await asyncio.to_thread(delete_resume_chunks, username)
        await asyncio.to_thread(
            upsert_resume_chunks,
            username,
            [
                {"section": c.section, "chunk_index": c.chunk_index, "text": c.chunk_text, "vector": v}
                for c, v in zip(user_chunks, vectors)
                if v
            ],
        )

    for job in jobs:
        required, preferred = job_skills[job.id]
        await asyncio.to_thread(
            index_job_vector, job, companies.get(job.company_id, ""), sorted(set(required + preferred)), job.cleaned_description
        )

    counts = {"resume_profiles": len(profiles), "resume_chunks": len(chunks), "job_postings": len(jobs)}
    logger.info("Reindexed Qdrant from Postgres: %s", counts)
    return counts


if __name__ == "__main__":
    print(asyncio.run(reindex_all()))
