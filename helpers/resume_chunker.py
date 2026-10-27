"""
Split a CV into section-labelled chunks for retrieval.

Strategy:
1. Section detection: a short standalone line (e.g. "Professional Experience",
   "EDUCATION & CERTIFICATIONS", "Technical Skills:") is a heading when it
   contains a known section keyword. Matching is by keyword, not exact
   wording, so the many real-world variants of each heading are caught.
2. Size control: sections longer than MAX_CHUNK_CHARS are split on line
   boundaries (each job / project usually starts on its own line).
3. Contextual headers: every continuation chunk starts with the entry line it
   belongs to ("L&T Mindtree - Software Engineer Oct 2024 - Present"), so a
   chunk of bullets is never orphaned from its employer/project and the LLM
   can't attribute it to the wrong job.
4. Every chunk keeps its section label; chunk_embedding_text() prefixes it
   when embedding, so "education" questions land near education chunks.
"""
import re
from dataclasses import dataclass

MAX_CHUNK_CHARS = 900
MAX_HEADING_WORDS = 6

# Section -> keywords that identify its heading. Order matters: the first
# match wins, so "Education & Certifications" is education, not certifications.
_SECTION_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("experience", ("experience", "employment", "work history", "career history", "professional background")),
    ("education", ("education", "academic", "qualification")),
    ("projects", ("project",)),
    ("skills", ("skill", "competenc", "technolog", "tech stack", "toolkit", "expertise")),
    ("certifications", ("certification", "certificate", "licen", "courses")),
    ("achievements", ("achievement", "award", "honor", "honour", "accomplishment")),
    ("summary", ("summary", "profile", "objective", "about me", "about")),
]


@dataclass
class ResumeChunk:
    section: str
    text: str


def _heading_section(line: str) -> str | None:
    text = line.strip().rstrip(":").strip()
    if not text or len(text) > 50 or len(text.split()) > MAX_HEADING_WORDS:
        return None
    if text.endswith((".", ",", ";")) or re.search(r"\d{4}", text) or "@" in text:
        return None  # sentences, date lines ("2019 - 2021") and contact lines aren't headings
    lowered = re.sub(r"[^a-z& ]+", " ", text.lower())
    for section, keywords in _SECTION_KEYWORDS:
        if any(k in lowered for k in keywords):
            return section
    return None


_BULLET_RE = re.compile(r"^\s*[•\-*▪◦·]")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b|present|current", re.I)


def _is_entry_line(line: str) -> bool:
    """A job / project / degree title line: not a bullet, short, and dated or labelled."""
    text = line.strip()
    if not text or _BULLET_RE.match(text) or len(text) > 160:
        return False
    return bool(_YEAR_RE.search(text)) or "project" in text.lower()


def _split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(text) <= limit or len(lines) <= 1:
        return [text.strip()]

    parts: list[str] = []
    current: list[str] = []
    entry: str | None = None  # the job/project line the current bullets belong to
    for line in lines:
        starts_entry = _is_entry_line(line)
        # Start a new chunk at a new job/project, or when this one is full.
        if current and (starts_entry or len("\n".join(current + [line])) > limit):
            parts.append("\n".join(current).strip())
            current = [] if starts_entry or entry is None else [f"{entry} (continued)"]
        if starts_entry:
            entry = line.strip()
        current.append(line)
    if current:
        parts.append("\n".join(current).strip())
    return parts


def chunk_resume(resume_text: str) -> list[ResumeChunk]:
    """
    Returns section-labelled chunks in document order. Text before the first
    heading (name, title, contact line) becomes a "header" chunk. Falls back
    to size-split "full_text" chunks when no headings are found, so
    unstructured CVs are still indexed.
    """
    if not resume_text or not resume_text.strip():
        return []
    # CVs stored before PDF cleanup may still contain pdfminer's unmapped-glyph codes.
    resume_text = re.sub(r"\(cid:\d+\)", "•", resume_text)

    sections: list[tuple[str, list[str]]] = [("header", [])]
    for line in resume_text.splitlines():
        section = _heading_section(line)
        if section:
            sections.append((section, []))
        else:
            sections[-1][1].append(line)

    if len(sections) == 1:
        return [ResumeChunk("full_text", part) for part in _split_long(resume_text)]

    chunks: list[ResumeChunk] = []
    for section, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            chunks.extend(ResumeChunk(section, part) for part in _split_long(body))
    return chunks


def chunk_embedding_text(section: str, text: str) -> str:
    """What actually gets embedded: the section label gives the vector topical context."""
    return f"{section.replace('_', ' ').title()}: {text}"
