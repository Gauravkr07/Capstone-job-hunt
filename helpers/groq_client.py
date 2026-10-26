"""
Single Groq (OpenAI-compatible chat completions) client shared by every LLM
caller in the app: JD extraction, match explanation, and the RAG chatbot.

One pooled httpx.AsyncClient is reused across requests instead of opening a
new connection + TLS handshake per call. Every helper here returns None / an
empty stream (never raises) when no key is configured or the call fails, so
callers can fall back to their non-LLM path.
"""
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from config import GROQ_API_KEY, GROQ_FALLBACK_MODEL, GROQ_MODEL
from helpers.logger import get_logger

logger = get_logger("groq_client")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

_client: httpx.AsyncClient | None = None


def is_configured() -> bool:
    return bool(GROQ_API_KEY)


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(60, connect=10),
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


async def chat_json(messages: list[dict[str, str]], temperature: float = 0, label: str = "groq") -> dict[str, Any] | None:
    """Non-streaming call in JSON mode; returns the parsed JSON object, or None on any failure."""
    if not is_configured():
        logger.info("GROQ_API_KEY not configured; skipping %s", label)
        return None

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    try:
        response = await _get_client().post(GROQ_API_URL, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception as exc:
        logger.warning("Groq %s request failed: %s", label, exc)
        return None

    return parsed if isinstance(parsed, dict) else None


async def stream_chat(messages: list[dict[str, str]], temperature: float = 0.3) -> AsyncIterator[str]:
    """
    Yield content deltas from Groq's SSE stream. If the main model is rate-limited
    (429) before anything was streamed, retries once on GROQ_FALLBACK_MODEL.
    Yields nothing if unconfigured or every attempt fails.
    """
    if not is_configured():
        logger.info("GROQ_API_KEY not configured; skipping streaming generation")
        return

    models = [GROQ_MODEL] + ([GROQ_FALLBACK_MODEL] if GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != GROQ_MODEL else [])
    for model in models:
        payload = {"model": model, "messages": messages, "temperature": temperature, "stream": True}
        try:
            async with _get_client().stream("POST", GROQ_API_URL, json=payload) as response:
                if response.status_code == 429:
                    logger.warning("Groq model %s rate-limited (429)%s", model, "; trying fallback" if model != models[-1] else "")
                    continue
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
                return
        except Exception as exc:
            logger.warning("Groq streaming request failed (%s): %s", model, exc)
            return
