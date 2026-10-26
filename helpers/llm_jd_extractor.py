from typing import Any

from helpers import groq_client

_SYSTEM_PROMPT = """You are an expert job-description parser: a precise information-extraction \
assistant specialized in reading raw, unstructured job postings (any format, any source) and \
extracting structured data from them with zero guesswork.

Extract structured data from the raw job posting text you are given. \
Return ONLY a JSON object matching this exact shape, no prose, no markdown fences:

{
  "title": string,
  "company_name": string,
  "location": array of strings or null,
  "remote_type": one of "remote", "hybrid", "onsite", or null,
  "employment_type": one of "full_time", "part_time", "contract", "internship", or null,
  "experience_min": integer or null,
  "experience_max": integer or null,
  "required_skills": array of lowercase strings (technologies, tools, languages, frameworks only),
  "preferred_skills": array of lowercase strings (skills explicitly marked nice-to-have/preferred/bonus),
  "industry_type": short lowercase string (e.g. "fintech", "healthcare", "ecommerce", "technology") or null
}

Rules:
- title must be the actual job title (e.g. "Senior Software Engineer"), never a section heading like "About the job".
- company_name must be the hiring company's actual name, never a generic word.
- If a field cannot be determined, use null (or [] for skill/location arrays).
- required_skills and preferred_skills must not overlap.
- Do not invent skills that are not explicitly mentioned in the text."""


def _build_user_prompt(raw_text: str) -> str:
    return f"Extract structured data from this job posting:\n\n{raw_text}"


async def extract_job_fields_via_llm(raw_text: str) -> dict[str, Any] | None:
    """
    Ask Groq's hosted LLM to extract structured job-posting fields from raw text.
    Returns None (never raises) when no API key is configured or the call fails,
    so callers can fall back to the regex-based parser.
    """
    parsed = await groq_client.chat_json(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(raw_text)},
        ],
        temperature=0,
        label="JD extraction",
    )
    if parsed is None:
        return None

    return _normalize_llm_result(parsed)


def _to_int(value: Any) -> int | None:
    # LLMs occasionally return "3" or 3.0 for integer fields; the DB columns are Integer.
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_llm_result(parsed: dict[str, Any]) -> dict[str, Any]:
    title = (parsed.get("title") or "").strip()[:255]
    company_name = (parsed.get("company_name") or "").strip()[:255]

    location = parsed.get("location")
    if isinstance(location, str):
        location = [location] if location.strip() else None
    elif isinstance(location, list):
        location = [str(loc).strip() for loc in location if str(loc).strip()] or None
    else:
        location = None

    required_skills = [str(s).strip().lower() for s in (parsed.get("required_skills") or []) if str(s).strip()]
    preferred_skills = [str(s).strip().lower() for s in (parsed.get("preferred_skills") or []) if str(s).strip()]

    return {
        "title": title or None,
        "company_name": company_name or None,
        "location": location,
        "remote_type": parsed.get("remote_type") or None,
        "employment_type": parsed.get("employment_type") or None,
        "experience_min": _to_int(parsed.get("experience_min")),
        "experience_max": _to_int(parsed.get("experience_max")),
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "industry_type": (parsed.get("industry_type") or "").strip().lower() or None,
    }
