import re
from dataclasses import dataclass

_SECTION_PATTERNS: list[tuple[str, str]] = [
    ("summary", r"(?:professional\s+)?summary|about\s*(?:me)?|objective|profile"),
    ("experience", r"(?:work\s+)?experience|employment\s*(?:history)?|work\s*history"),
    ("education", r"education|academic\s*background"),
    ("skills", r"skills|technical\s*skills|core\s*competencies"),
    ("certifications", r"certifications?|licenses?"),
    ("projects", r"projects?"),
    ("achievements", r"achievements?|awards?|honors?"),
]

_HEADER_LINE_RE = re.compile(
    r"^\s*(" + "|".join(pattern for _, pattern in _SECTION_PATTERNS) + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_NAME_BY_PATTERN = {pattern: name for name, pattern in _SECTION_PATTERNS}


@dataclass
class ResumeChunk:
    section: str
    text: str


def _section_name_for_match(matched_text: str) -> str:
    lowered = matched_text.strip().lower()
    for name, pattern in _SECTION_PATTERNS:
        if re.fullmatch(pattern, lowered, re.IGNORECASE):
            return name
    return "other"


def chunk_resume(resume_text: str) -> list[ResumeChunk]:
    """
    Split resume text into sections by detecting common resume headers
    (Summary, Experience, Education, Skills, Certifications, Projects, Achievements).

    Falls back to a single "full_text" chunk when no headers are detected,
    so unstructured resumes still get embedded rather than dropped.
    """
    if not resume_text or not resume_text.strip():
        return []

    matches = list(_HEADER_LINE_RE.finditer(resume_text))
    if not matches:
        return [ResumeChunk(section="full_text", text=resume_text.strip())]

    chunks: list[ResumeChunk] = []

    header_text = resume_text[: matches[0].start()].strip()
    if header_text:
        chunks.append(ResumeChunk(section="header", text=header_text))

    for i, match in enumerate(matches):
        section_name = _section_name_for_match(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(resume_text)
        body = resume_text[start:end].strip()
        if body:
            chunks.append(ResumeChunk(section=section_name, text=body))

    return chunks
