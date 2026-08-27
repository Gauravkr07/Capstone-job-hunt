import hashlib
import re
from typing import Any

from helpers.llm_jd_extractor import extract_job_fields_via_llm
from helpers.logger import get_logger
from helpers.text_extraction import _keyword_in, extract_skills, infer_industry_type, normalize_text

logger = get_logger("jd_scraper")

_PREFERRED_MARKERS = ("preferred", "nice to have", "good to have", "bonus", "plus")

_REMOTE_TYPE_PATTERNS = {
    "remote": ["remote", "work from home", "wfh"],
    "hybrid": ["hybrid"],
    "onsite": ["onsite", "on-site", "in office", "in-office"],
}

_EMPLOYMENT_TYPE_PATTERNS = {
    "full_time": ["full-time", "full time", "permanent"],
    "part_time": ["part-time", "part time"],
    "contract": ["contract", "contractor", "freelance"],
    "internship": ["internship", "intern"],
}


def clean_description(raw_html_or_text: str) -> str:
    """Strip HTML tags and collapse whitespace, producing a plain-text description."""
    without_tags = re.sub(r"<[^>]+>", " ", raw_html_or_text)
    return normalize_text(without_tags)


def compute_content_hash(title: str, company_name: str, location: list[str] | None, description: str) -> str:
    location_basis = ",".join(sorted(loc.strip().lower() for loc in location)) if location else ""
    basis = "|".join([title.strip().lower(), company_name.strip().lower(), location_basis, description.strip()])
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _find_experience_range(text: str) -> tuple[int | None, int | None]:
    lowered = text.lower()

    range_match = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)\s*(?:yr|yrs|years|year)", lowered)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    plus_match = re.search(r"(\d+)\s*\+\s*(?:yr|yrs|years|year)", lowered)
    if plus_match:
        return int(plus_match.group(1)), None

    single_match = re.search(r"(\d+)\s*(?:yr|yrs|years|year)", lowered)
    if single_match:
        value = int(single_match.group(1))
        return value, value

    return None, None


def _classify_skills(text: str, skills: list[str]) -> dict[str, list[str]]:
    """
    Split skills into required vs preferred based on whether they appear closer
    to a "preferred/nice to have" marker than the start of the requirements text.
    Falls back to "required" when no preference markers are present.
    """
    lowered = text.lower()
    preferred: list[str] = []
    required: list[str] = []

    marker_positions = [lowered.find(marker) for marker in _PREFERRED_MARKERS if marker in lowered]
    first_preferred_marker = min(marker_positions) if marker_positions else None

    for skill in skills:
        skill_pos = lowered.find(skill)
        if first_preferred_marker is not None and skill_pos != -1 and skill_pos > first_preferred_marker:
            preferred.append(skill)
        else:
            required.append(skill)

    return {"required": required, "preferred": preferred}


def _match_pattern_group(text: str, patterns: dict[str, list[str]]) -> str | None:
    for label, keywords in patterns.items():
        if any(_keyword_in(keyword, text) for keyword in keywords):
            return label
    return None


_COMPANY_LINE_RE = re.compile(r"^(?:company|employer)\b\s*:?\s*(.+)$", re.IGNORECASE)
_AT_COMPANY_RE = re.compile(r"\bat\s+([A-Z][\w&.,'-]*(?:\s+[A-Z][\w&.,'-]*){0,4})", re.MULTILINE)
# "Playo is a leading...", "Playo is building..." -- common JD-opener phrasing.
_COMPANY_IS_RE = re.compile(r"\b([A-Z][\w&'-]{1,40})(?:'s)?\s+is\s+(?:a|an|the|building|hiring)\b")
# "join Playo's engineering team", "join Playo as a..."
_JOIN_COMPANY_RE = re.compile(r"\bjoin\s+([A-Z][\w&'-]{1,40})(?:'s)?\b")
_LOCATION_LINE_RE = re.compile(r"^(?:location|locations|based\s+in)\b\s*:?\s*(.+)$", re.IGNORECASE)
_TITLE_LINE_RE = re.compile(r"^(?:job\s*title|title|position|role)\b\s*:?\s*(.+)$", re.IGNORECASE)
# "hiring a Backend Python Developer to/for/at ..." -- common embedded-title phrasing.
_HIRING_TITLE_RE = re.compile(
    r"\bhiring\s+(?:a|an)\s+([A-Z][\w/+#.-]*(?:\s+[A-Z][\w/+#.-]*){0,6})(?=\s+(?:to|for|at|who|that|\.|,))"
)

MAX_TITLE_LENGTH = 255
MAX_COMPANY_NAME_LENGTH = 255


def _split_locations(raw: str) -> list[str]:
    parts = re.split(r"\s*(?:,|/|;|\band\b|\bor\b)\s*", raw, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def _looks_like_title_line(line: str) -> bool:
    """
    A plausible job-title line is short, single-sentence, and doesn't read like
    marketing/boilerplate prose (which real JD postings often lead with).
    """
    if not line or len(line) > 100:
        return False
    if line.count(".") > 1:
        return False
    word_count = len(line.split())
    return 1 <= word_count <= 12


def _parse_title_company_location(text: str) -> tuple[str | None, str | None, list[str] | None]:
    """
    Heuristically pull title, company, and location(s) out of raw, unstructured
    job-posting text (e.g. copy-pasted from a job board). Looks for explicit
    "Title:"/"Company:"/"Location:" lines first; otherwise scans the first few
    lines for one that looks like a short title (skipping long boilerplate
    paragraphs some postings lead with), and uses the "at <Company>" pattern
    for company as a last resort.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    title = None
    company = None
    location: list[str] | None = None

    for line in lines[:15]:
        if title is None:
            title_match = _TITLE_LINE_RE.match(line)
            if title_match:
                title = title_match.group(1).strip()
                continue

        if company is None:
            company_match = _COMPANY_LINE_RE.match(line)
            if company_match:
                company = company_match.group(1).strip()
                continue

        if location is None:
            location_match = _LOCATION_LINE_RE.match(line)
            if location_match:
                location = _split_locations(location_match.group(1))
                continue

    if company is None:
        at_match = _AT_COMPANY_RE.search(text)
        if at_match:
            company = at_match.group(1).strip()

    if company is None:
        is_match = _COMPANY_IS_RE.search(text)
        if is_match:
            company = is_match.group(1).strip()

    if company is None:
        join_match = _JOIN_COMPANY_RE.search(text)
        if join_match:
            company = join_match.group(1).strip()

    if title is None:
        hiring_match = _HIRING_TITLE_RE.search(text)
        if hiring_match:
            title = hiring_match.group(1).strip()

    if title is None:
        # Scan the first few lines for one that looks like a plausible title,
        # rather than blindly taking line 1 (which may be a boilerplate paragraph).
        for line in lines[:15]:
            if line != company and _looks_like_title_line(line):
                title = line
                break

    if title is None and lines:
        # Nothing looked title-like; fall back to the first line, truncated to fit.
        first_line = lines[0]
        if first_line != company:
            title = first_line[:MAX_TITLE_LENGTH]

    if title:
        title = title[:MAX_TITLE_LENGTH]
    if company:
        company = company[:MAX_COMPANY_NAME_LENGTH]

    return title, company, location


async def parse_raw_job_posting(raw_text: str) -> dict[str, Any]:
    """
    Parse a raw, unstructured job posting (e.g. pasted directly from a job
    board) into title/company/location plus everything scrape_job_description
    already extracts (skills, experience, industry, etc.).

    Tries LLM-based extraction first (accurate on arbitrary real-world JD
    formatting) and falls back to the regex/keyword heuristics below when no
    LLM is configured or the call fails, so this never hard-depends on an
    external API being available.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Job posting text is empty")

    llm_result = await extract_job_fields_via_llm(raw_text)

    if llm_result is None:
        logger.info("Falling back to regex-based extraction: LLM extraction unavailable or failed")
    elif not llm_result.get("title") or not llm_result.get("company_name"):
        missing = [f for f in ("title", "company_name") if not llm_result.get(f)]
        logger.info("LLM result missing %s; filling gaps from regex extraction", ", ".join(missing))

    # Regex-based title/company/location detection, used either as the sole
    # source (LLM unavailable) or to fill in whatever the LLM didn't find.
    regex_title, regex_company, regex_location = _parse_title_company_location(raw_text)

    title = (llm_result.get("title") if llm_result else None) or regex_title
    company_name = (llm_result.get("company_name") if llm_result else None) or regex_company
    location = (llm_result.get("location") if llm_result else None) or regex_location

    if not title:
        raise ValueError("Could not determine job title from the provided text")
    if not company_name:
        raise ValueError("Could not determine company name from the provided text")

    if llm_result is not None:
        logger.info("Using LLM-extracted fields for job posting")
        cleaned = clean_description(raw_text)
        content_hash = compute_content_hash(title, company_name, location, cleaned)
        return {
            "title": title,
            "company_name": company_name,
            "location": location,
            "cleaned_description": cleaned,
            "skills": sorted(set(llm_result["required_skills"] + llm_result["preferred_skills"])),
            "required_skills": llm_result["required_skills"],
            "preferred_skills": llm_result["preferred_skills"],
            "experience_min": llm_result["experience_min"],
            "experience_max": llm_result["experience_max"],
            "industry_type": llm_result["industry_type"],
            "remote_type": llm_result["remote_type"],
            "employment_type": llm_result["employment_type"],
            "content_hash": content_hash,
        }

    logger.info("Using regex-based extraction for job posting")
    extracted = scrape_job_description(title, company_name, raw_text, location=location)
    extracted["title"] = title
    extracted["company_name"] = company_name
    extracted["location"] = location
    return extracted


def scrape_job_description(
    title: str,
    company_name: str,
    jd_text: str,
    location: list[str] | None = None,
) -> dict[str, Any]:
    """
    Extract structured fields from a raw job description: cleaned text, skills
    (split required/preferred), experience range, industry, remote/employment type,
    and a content hash for de-duplication.
    """
    if not jd_text or not jd_text.strip():
        raise ValueError("Job description text is empty")

    cleaned = clean_description(jd_text)
    normalized = normalize_text(cleaned)

    all_skills = extract_skills(normalized)
    skill_split = _classify_skills(normalized, all_skills)

    experience_min, experience_max = _find_experience_range(normalized)
    industry_type = infer_industry_type(normalized)
    remote_type = _match_pattern_group(normalized, _REMOTE_TYPE_PATTERNS)
    employment_type = _match_pattern_group(normalized, _EMPLOYMENT_TYPE_PATTERNS)

    content_hash = compute_content_hash(title, company_name, location, cleaned)

    return {
        "cleaned_description": cleaned,
        "skills": all_skills,
        "required_skills": skill_split["required"],
        "preferred_skills": skill_split["preferred"],
        "experience_min": experience_min,
        "experience_max": experience_max,
        "industry_type": industry_type,
        "remote_type": remote_type,
        "employment_type": employment_type,
        "content_hash": content_hash,
    }
