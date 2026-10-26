import threading
import time
from functools import lru_cache

from helpers.logger import get_logger

logger = get_logger("embedding")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# lru_cache doesn't serialize concurrent first calls, so without this a request
# arriving during background warm-up would start a second model load.
_load_lock = threading.Lock()


def _get_model():
    with _load_lock:
        return _load_model()


@lru_cache(maxsize=1)
def _load_model():
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


def is_model_ready() -> bool:
    """True once the model is loaded -- lets callers skip vector search instead of blocking on a slow download."""
    return _load_model.cache_info().currsize > 0


def warm_up_model() -> None:
    _get_model()


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
    return embed_text(text)


def embed_text(text: str) -> list[float]:
    """Embed arbitrary free text (e.g. a job description) with the same model/space as resumes."""
    if not text or not text.strip():
        logger.warning("Skipping text embedding: empty text")
        return []
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Batch-embed many texts in a single model.encode() call -- much faster than
    one call per text (e.g. per resume chunk). Empty texts map to [].
    """
    non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    vectors: list[list[float]] = [[] for _ in texts]
    if not non_empty:
        return vectors

    start = time.perf_counter()
    encoded = _get_model().encode([t for _, t in non_empty], normalize_embeddings=True)
    for (i, _), vector in zip(non_empty, encoded):
        vectors[i] = vector.tolist()

    logger.info("Embedded %d text(s) in %.1fms", len(non_empty), (time.perf_counter() - start) * 1000)
    return vectors
