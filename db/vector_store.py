import time
from functools import lru_cache
from typing import Any

from config import QDRANT_API_KEY, QDRANT_URL
from helpers.embeddings import EMBEDDING_DIM
from helpers.logger import get_logger

logger = get_logger("vectorization")

RESUME_COLLECTION = "resume_profiles"
RESUME_CHUNKS_COLLECTION = "resume_chunks"
JOB_POSTINGS_COLLECTION = "job_postings"


@lru_cache(maxsize=1)
def _get_client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _ensure_collection(collection_name: str) -> None:
    from qdrant_client.models import Distance, VectorParams

    client = _get_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def ensure_collection() -> None:
    _ensure_collection(RESUME_COLLECTION)


def _point_id(username: str) -> str:
    import uuid

    # Deterministic UUID from username so re-upserts overwrite the same point.
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"resume-profile:{username}"))


def _chunk_point_id(username: str, section: str, chunk_index: int) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"resume-chunk:{username}:{section}:{chunk_index}"))


def _job_point_id(job_id: int) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"job-posting:{job_id}"))


def get_resume_vector_point(username: str) -> dict[str, Any] | None:
    """Direct point lookup (not similarity search) -- returns the stored vector + payload for one user, if any."""
    client = _get_client()
    _ensure_collection(RESUME_COLLECTION)

    points = client.retrieve(collection_name=RESUME_COLLECTION, ids=[_point_id(username)], with_vectors=True)
    if not points:
        return None

    point = points[0]
    return {"id": point.id, "vector": point.vector, "payload": point.payload}


def get_resume_chunk_points(username: str) -> list[dict[str, Any]]:
    """Direct lookup (not similarity search) -- returns all stored chunk points + vectors for one user."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = _get_client()
    _ensure_collection(RESUME_CHUNKS_COLLECTION)

    points, _ = client.scroll(
        collection_name=RESUME_CHUNKS_COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="username", match=MatchValue(value=username))]),
        with_vectors=True,
        limit=100,
    )
    return [{"id": point.id, "vector": point.vector, "payload": point.payload} for point in points]


def get_job_vector_point(job_id: int) -> dict[str, Any] | None:
    """Direct point lookup (not similarity search) -- returns the stored vector + payload for one job, if any."""
    client = _get_client()
    _ensure_collection(JOB_POSTINGS_COLLECTION)

    points = client.retrieve(collection_name=JOB_POSTINGS_COLLECTION, ids=[_job_point_id(job_id)], with_vectors=True)
    if not points:
        return None

    point = points[0]
    return {"id": point.id, "vector": point.vector, "payload": point.payload}


def get_job_vector_points(job_ids: list[int]) -> dict[int, dict[str, Any]]:
    """
    Batched version of get_job_vector_point -- one Qdrant round-trip for many
    jobs instead of one per job. Returns {job_id: {vector, payload}}, omitting
    any job_id with no stored point.
    """
    if not job_ids:
        return {}

    client = _get_client()
    _ensure_collection(JOB_POSTINGS_COLLECTION)

    ids = [_job_point_id(job_id) for job_id in job_ids]
    points = client.retrieve(collection_name=JOB_POSTINGS_COLLECTION, ids=ids, with_vectors=True)

    return {point.payload["job_id"]: {"vector": point.vector, "payload": point.payload} for point in points}


def upsert_resume_vector(
    username: str,
    vector: list[float],
    skills: list[str],
    industry_type: str | None,
    experience: int | None,
) -> None:
    if not vector:
        logger.warning("Skipping Qdrant upsert for %s: empty vector", username)
        return

    from qdrant_client.models import PointStruct

    client = _get_client()
    ensure_collection()

    start = time.perf_counter()
    client.upsert(
        collection_name=RESUME_COLLECTION,
        points=[
            PointStruct(
                id=_point_id(username),
                vector=vector,
                payload={
                    "username": username,
                    "skills": skills,
                    "industry_type": industry_type,
                    "experience": experience,
                },
            )
        ],
    )
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("Upserted vector for %s into %s in %.1fms", username, RESUME_COLLECTION, duration_ms)


def delete_resume_chunks(username: str) -> None:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = _get_client()
    _ensure_collection(RESUME_CHUNKS_COLLECTION)

    client.delete(
        collection_name=RESUME_CHUNKS_COLLECTION,
        points_selector=Filter(must=[FieldCondition(key="username", match=MatchValue(value=username))]),
    )
    logger.info("Deleted existing chunks for %s from %s", username, RESUME_CHUNKS_COLLECTION)


def upsert_resume_chunks(
    username: str,
    chunks: list[dict[str, Any]],
) -> None:
    """
    chunks: list of {"section": str, "chunk_index": int, "text": str, "vector": list[float]}

    Replaces all of the user's existing chunks with the given set, since section
    boundaries/count can change between resume versions.
    """
    if not chunks:
        logger.warning("Skipping Qdrant chunk upsert for %s: no chunks", username)
        return

    from qdrant_client.models import PointStruct

    client = _get_client()
    _ensure_collection(RESUME_CHUNKS_COLLECTION)

    delete_resume_chunks(username)

    start = time.perf_counter()
    client.upsert(
        collection_name=RESUME_CHUNKS_COLLECTION,
        points=[
            PointStruct(
                id=_chunk_point_id(username, chunk["section"], chunk["chunk_index"]),
                vector=chunk["vector"],
                payload={
                    "username": username,
                    "section": chunk["section"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                },
            )
            for chunk in chunks
            if chunk.get("vector")
        ],
    )
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Upserted %d chunks for %s into %s in %.1fms",
        len(chunks),
        username,
        RESUME_CHUNKS_COLLECTION,
        duration_ms,
    )


def search_resume_chunks(
    vector: list[float],
    limit: int = 10,
    username: str | None = None,
) -> list[dict[str, Any]]:
    if not vector:
        logger.warning("Skipping Qdrant chunk search: empty query vector")
        return []

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = _get_client()
    query_filter = None
    if username:
        query_filter = Filter(must=[FieldCondition(key="username", match=MatchValue(value=username))])

    start = time.perf_counter()
    results = client.query_points(
        collection_name=RESUME_CHUNKS_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=limit,
    )
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Searched %s (username_filter=%s, limit=%d) -> %d results in %.1fms",
        RESUME_CHUNKS_COLLECTION,
        username,
        limit,
        len(results.points),
        duration_ms,
    )
    return [{"score": point.score, **point.payload} for point in results.points]


def search_similar_resumes(
    vector: list[float],
    limit: int = 10,
    industry_type: str | None = None,
) -> list[dict[str, Any]]:
    if not vector:
        logger.warning("Skipping Qdrant search: empty query vector")
        return []

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = _get_client()
    query_filter = None
    if industry_type:
        query_filter = Filter(must=[FieldCondition(key="industry_type", match=MatchValue(value=industry_type))])

    start = time.perf_counter()
    results = client.query_points(
        collection_name=RESUME_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=limit,
    )
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Searched %s (industry_filter=%s, limit=%d) -> %d results in %.1fms",
        RESUME_COLLECTION,
        industry_type,
        limit,
        len(results.points),
        duration_ms,
    )
    return [{"score": point.score, **point.payload} for point in results.points]


def upsert_job_vector(
    job_id: int,
    vector: list[float],
    payload: dict[str, Any],
) -> None:
    """
    payload should carry only the compact fields needed for filtering/ranking
    (title, company, location, remote_type, experience range, skills, industry, status)
    -- not the full description. Re-upserting with the same job_id overwrites the point.
    """
    if not vector:
        logger.warning("Skipping Qdrant upsert for job %s: empty vector", job_id)
        return

    from qdrant_client.models import PointStruct

    client = _get_client()
    _ensure_collection(JOB_POSTINGS_COLLECTION)

    start = time.perf_counter()
    client.upsert(
        collection_name=JOB_POSTINGS_COLLECTION,
        points=[
            PointStruct(
                id=_job_point_id(job_id),
                vector=vector,
                payload={"job_id": job_id, **payload},
            )
        ],
    )
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("Upserted vector for job %s into %s in %.1fms", job_id, JOB_POSTINGS_COLLECTION, duration_ms)


def search_similar_jobs(
    vector: list[float],
    limit: int = 10,
    status: str | None = "ACTIVE",
    industry_type: str | None = None,
) -> list[dict[str, Any]]:
    if not vector:
        logger.warning("Skipping Qdrant job search: empty query vector")
        return []

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = _get_client()
    must_conditions = []
    if status:
        must_conditions.append(FieldCondition(key="status", match=MatchValue(value=status)))
    if industry_type:
        must_conditions.append(FieldCondition(key="industry_type", match=MatchValue(value=industry_type)))
    query_filter = Filter(must=must_conditions) if must_conditions else None

    start = time.perf_counter()
    results = client.query_points(
        collection_name=JOB_POSTINGS_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=limit,
    )
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Searched %s (status=%s, industry_filter=%s, limit=%d) -> %d results in %.1fms",
        JOB_POSTINGS_COLLECTION,
        status,
        industry_type,
        limit,
        len(results.points),
        duration_ms,
    )
    return [{"score": point.score, **point.payload} for point in results.points]
