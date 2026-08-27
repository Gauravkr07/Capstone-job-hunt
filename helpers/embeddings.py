import time
from functools import lru_cache

from helpers.logger import get_logger

logger = get_logger("embedding")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_model():
    # sentence-transformers is a runtime dependency used only when embeddings are requested.
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to generate embeddings. "
            "Install it with `pip install sentence-transformers`."
        ) from exc

    logger.info("Loading embedding model %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_profile_text(skills: list[str], industry_type: str | None, experience: int | None) -> str:
    parts = []
    if skills:
        parts.append(f"Skills: {', '.join(skills)}")
    if industry_type:
        parts.append(f"Industry: {industry_type}")
    if experience is not None:
        parts.append(f"Experience: {experience} years")
    return ". ".join(parts)


def embed_profile(skills: list[str], industry_type: str | None, experience: int | None) -> list[float]:
    text = build_profile_text(skills, industry_type, experience)
    if not text:
        logger.warning("Skipping profile embedding: no skills/industry/experience to embed")
        return []

    start = time.perf_counter()
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Embedded profile (skills=%d, industry=%s, experience=%s) in %.1fms",
        len(skills),
        industry_type,
        experience,
        duration_ms,
    )
    return embedding.tolist()


def embed_text(text: str) -> list[float]:
    """Embed arbitrary free text (e.g. a job description) with the same model/space as resumes."""
    if not text or not text.strip():
        logger.warning("Skipping text embedding: empty text")
        return []

    start = time.perf_counter()
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info("Embedded text (%d chars) in %.1fms", len(text), duration_ms)
    return embedding.tolist()
