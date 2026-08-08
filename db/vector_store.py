import time
from functools import lru_cache
from typing import Any

from config import QDRANT_API_KEY, QDRANT_URL
from helpers.embeddings import EMBEDDING_DIM
from helpers.logger import get_logger

logger = get_logger("vectorization")

RESUME_COLLECTION = "resume_profiles"


@lru_cache(maxsize=1)
def _get_client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection() -> None:
    from qdrant_client.models import Distance, VectorParams

    client = _get_client()
    if not client.collection_exists(RESUME_COLLECTION):
        client.create_collection(
            collection_name=RESUME_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def _point_id(username: str) -> str:
    import uuid

    # Deterministic UUID from username so re-upserts overwrite the same point.
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"resume-profile:{username}"))


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
