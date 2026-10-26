"""
Fetch live job postings from public ATS APIs (Greenhouse, Lever, Ashby) and
deterministically prefilter them by title/location/freshness before they ever
reach ingestion. No auth, no scraping, no ToS risk -- these are documented,
unauthenticated board endpoints.

Adapted from a standalone job-search agent's fetch.py/prefilter.py: HTTP is
kept out of the parsers on purpose, so each parse_* function is independently
testable against a fixture body.
"""
from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from helpers.logger import get_logger

logger = get_logger("job_board_fetch")

UA = {"User-Agent": "capstone-job-hunt/1.0 (job board fetcher)"}
TIMEOUT = 20

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")

REMOTE_HINTS = ("remote", "anywhere", "work from home", "wfh", "distributed")

# Curated boards matched to a backend/Python/data/AI-engineer profile.
# ats: greenhouse | lever | ashby -- slug: the board's URL path segment.
DEFAULT_COMPANIES: list[dict[str, str]] = [
    {"ats": "greenhouse", "slug": "stripe", "name": "Stripe"},
    {"ats": "greenhouse", "slug": "databricks", "name": "Databricks"},
    {"ats": "greenhouse", "slug": "cloudflare", "name": "Cloudflare"},
    {"ats": "greenhouse", "slug": "figma", "name": "Figma"},
    {"ats": "greenhouse", "slug": "dropbox", "name": "Dropbox"},
    {"ats": "greenhouse", "slug": "coinbase", "name": "Coinbase"},
    {"ats": "greenhouse", "slug": "reddit", "name": "Reddit"},
    {"ats": "greenhouse", "slug": "postman", "name": "Postman"},
    {"ats": "ashby", "slug": "openai", "name": "OpenAI"},
    {"ats": "ashby", "slug": "ramp", "name": "Ramp"},
    {"ats": "ashby", "slug": "linear", "name": "Linear"},
    {"ats": "ashby", "slug": "vanta", "name": "Vanta"},
    {"ats": "ashby", "slug": "confluent", "name": "Confluent"},
    {"ats": "ashby", "slug": "zapier", "name": "Zapier"},
    # India-focused additions -- slugs verified live (HTTP 200 with jobs) as
    # of this writing. ATS board slugs are not always the company's obvious
    # name (e.g. Razorpay's is its registered entity name) and go dead when a
    # company migrates vendors, so verify before adding more here rather than
    # guessing -- see companies.yaml comments in the project this was adapted
    # from for the exact per-ATS URL to check a slug against.
    {"ats": "greenhouse", "slug": "groww", "name": "Groww"},
    {"ats": "greenhouse", "slug": "druva", "name": "Druva"},
    {"ats": "greenhouse", "slug": "highradius", "name": "HighRadius"},
    {"ats": "greenhouse", "slug": "razorpaysoftwareprivatelimited", "name": "Razorpay"},
    {"ats": "ashby", "slug": "supabase", "name": "Supabase"},
    {"ats": "ashby", "slug": "posthog", "name": "PostHog"},
]

DEFAULT_FILTERS: dict[str, Any] = {
    "include_titles": [
        r"\b(software|backend|back-end|python|server|platform|distributed)\b.*\bengineer\b",
        r"\bengineer\b.*\b(backend|back-end|python|platform|distributed)\b",
        r"\bsde\b",
        r"\bsde\s*-?\s*(i{1,3}|[123])\b",
        "software development engineer",
        "software engineer",
        "backend engineer",
        "backend developer",
        "python developer",
        "python engineer",
        "site reliability engineer",
        r"\bsre\b",
        "member of technical staff",
        r"\bdata\s+engineer\b",
        r"\bengineer\b.*\b(data pipeline|etl|streaming|kafka)\b",
        r"\b(data|analytics)\s+platform\s+engineer\b",
        r"\b(ai|ml|llm)\s+(platform\s+)?engineer\b",
        r"\bengineer\b.*\b(llm|rag|genai|generative ai)\b",
        r"\bapplied\s+ai\s+engineer\b",
    ],
    "exclude_titles": [
        r"\b(staff|principal|distinguished|fellow|architect)\b",
        r"\b(director|vp|vice president|head of|chief|cto)\b",
        r"\b(manager|management)\b",
        r"\b(intern|internship|apprentice|trainee)\b",
        r"\b(sales|account executive|marketing|recruit|support|success)\b",
        r"\b(ios|android|mobile game|game developer|designer)\b",
        r"\b(data scientist|machine learning scientist|research scientist)\b",
        r"\b(c\+\+|embedded|firmware|kernel|dpdk|dataplane|driver)\b",
    ],
    "locations": [
        "bangalore", "bengaluru", "india", "delhi", "gurgaon", "noida",
        "hyderabad", "pune", "chennai", "greater noida", "gurugram",
        "kolkata", "karnataka",
    ],
    "allow_remote": True,
    "max_age_days": 30,
}


def strip_html(raw: str | None) -> str:
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"<\s*(br|/p|/div|/li|/h[1-6])\s*/?>", "\n", text, flags=re.I)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()


@dataclass
class FetchedJob:
    job_id: str  # globally unique for dedupe: "<ats>:<slug>:<id>"
    ats: str
    company: str
    title: str
    location: str
    url: str
    description: str
    posted_at: str | None = None
    salary: str | None = None


def _parse_greenhouse(slug: str, company: str, body: Any) -> list[FetchedJob]:
    out = []
    for j in (body or {}).get("jobs", []):
        loc = (j.get("location") or {}).get("name") or ""
        out.append(FetchedJob(
            job_id=f"greenhouse:{slug}:{j.get('id')}",
            ats="greenhouse",
            company=company,
            title=(j.get("title") or "").strip(),
            location=loc.strip(),
            url=j.get("absolute_url") or "",
            description=strip_html(j.get("content")),
            posted_at=j.get("updated_at") or j.get("first_published"),
        ))
    return out


def _parse_lever(slug: str, company: str, body: Any) -> list[FetchedJob]:
    out = []
    for j in (body or []):
        cats = j.get("categories") or {}
        # Lever splits the JD across descriptionPlain + a `lists` array --
        # concatenate all of it or the requirements section goes missing.
        chunks = [j.get("descriptionPlain") or strip_html(j.get("description"))]
        for lst in (j.get("lists") or []):
            chunks.append(str(lst.get("text") or ""))
            chunks.append(strip_html(lst.get("content")))
        chunks.append(j.get("additionalPlain") or strip_html(j.get("additional")))
        ts = j.get("createdAt")
        posted = None
        if isinstance(ts, (int, float)):
            # Lever's createdAt is epoch milliseconds, not seconds.
            posted = time.strftime("%Y-%m-%d", time.gmtime(ts / 1000))
        out.append(FetchedJob(
            job_id=f"lever:{slug}:{j.get('id')}",
            ats="lever",
            company=company,
            title=(j.get("text") or "").strip(),
            location=(cats.get("location") or "").strip(),
            url=j.get("hostedUrl") or j.get("applyUrl") or "",
            description="\n\n".join(c for c in chunks if c).strip(),
            posted_at=posted,
            salary=cats.get("commitment"),
        ))
    return out


def _parse_ashby(slug: str, company: str, body: Any) -> list[FetchedJob]:
    out = []
    for j in (body or {}).get("jobs", []):
        if j.get("isListed") is False:
            continue  # unpublished draft
        comp = j.get("compensation") or {}
        salary = None
        summary = comp.get("compensationTierSummary") or comp.get("summaryComponents")
        if isinstance(summary, str):
            salary = summary
        out.append(FetchedJob(
            job_id=f"ashby:{slug}:{j.get('id')}",
            ats="ashby",
            company=company,
            title=(j.get("title") or "").strip(),
            location=(j.get("location") or "").strip(),
            url=j.get("jobUrl") or j.get("applyUrl") or "",
            description=(j.get("descriptionPlain") or strip_html(j.get("descriptionHtml")) or "").strip(),
            posted_at=j.get("publishedAt"),
            salary=salary,
        ))
    return out


_ENDPOINTS = {
    "greenhouse": ("https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true", _parse_greenhouse),
    "lever": ("https://api.lever.co/v0/postings/{slug}?mode=json", _parse_lever),
    "ashby": ("https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true", _parse_ashby),
}


def fetch_board(client: httpx.Client, ats: str, slug: str, company: str | None = None) -> list[FetchedJob]:
    """Hit one company's public board. Returns [] on any failure -- never raises, so one dead slug can't kill a batch fetch."""
    if ats not in _ENDPOINTS:
        raise ValueError(f"unknown ATS: {ats}")
    url_tpl, parser = _ENDPOINTS[ats]
    try:
        response = client.get(url_tpl.format(slug=slug))
        if response.status_code != 200:
            logger.warning("%s/%s -> HTTP %s", ats, slug, response.status_code)
            return []
        return parser(slug, company or slug, response.json())
    except Exception as exc:
        logger.warning("%s/%s -> %s: %s", ats, slug, type(exc).__name__, exc)
        return []


def fetch_all(companies: list[dict[str, str]] | None = None, sleep: float = 0.25) -> list[FetchedJob]:
    """Fetch every configured board, sequentially with a small delay to be a polite client."""
    companies = companies if companies is not None else DEFAULT_COMPANIES
    jobs: list[FetchedJob] = []
    with httpx.Client(headers=UA, timeout=TIMEOUT, follow_redirects=True) as client:
        for company in companies:
            found = fetch_board(client, company["ats"], company["slug"], company.get("name"))
            if found:
                logger.info("%s: %d jobs (%s)", company.get("name") or company["slug"], len(found), company["ats"])
            jobs.extend(found)
            time.sleep(sleep)
    return jobs


def _any_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    v = value.replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.fromisoformat(v) if fmt is None else datetime.strptime(v, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def prefilter(jobs: list[FetchedJob], filters: dict[str, Any] | None = None) -> list[FetchedJob]:
    """
    Deterministic, free gate that runs before any expensive step (LLM
    extraction, embedding, DB write): title must match an include pattern and
    no exclude pattern, location must be allowed (or remote, if permitted),
    and the posting must be recent enough. This is what keeps a batch fetch
    of thousands of raw postings down to a few dozen worth processing.
    """
    cfg = filters or DEFAULT_FILTERS
    include = cfg.get("include_titles") or [r"."]
    exclude = cfg.get("exclude_titles") or []
    locations = [loc.lower() for loc in (cfg.get("locations") or [])]
    allow_remote = bool(cfg.get("allow_remote", True))
    max_age_days = cfg.get("max_age_days")
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days) if max_age_days else None

    kept: list[FetchedJob] = []
    dropped = {"title": 0, "location": 0, "stale": 0}
    for job in jobs:
        if not _any_match(include, job.title) or (exclude and _any_match(exclude, job.title)):
            dropped["title"] += 1
            continue

        if locations:
            haystack = f"{job.location} {job.title}".lower()
            is_remote = allow_remote and any(hint in haystack for hint in REMOTE_HINTS)
            if not is_remote and not any(loc in haystack for loc in locations):
                dropped["location"] += 1
                continue

        if cutoff:
            posted = _parse_date(job.posted_at)
            if posted and posted < cutoff:
                dropped["stale"] += 1
                continue

        kept.append(job)

    logger.info(
        "prefilter: %d -> %d (dropped title=%d location=%d stale=%d)",
        len(jobs), len(kept), dropped["title"], dropped["location"], dropped["stale"],
    )
    return kept


def fetch_and_filter(companies: list[dict[str, str]] | None = None, filters: dict[str, Any] | None = None) -> list[FetchedJob]:
    """Convenience wrapper: fetch every configured board, then apply the deterministic prefilter."""
    raw = fetch_all(companies)
    return prefilter(raw, filters)
