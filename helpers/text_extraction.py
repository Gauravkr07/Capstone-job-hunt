import re
import unicodedata
from functools import lru_cache

from helpers.constant import INDUSTRY_PATTERNS, SKILL_KEYWORDS

# Smart/curly quotes and dashes commonly introduced by copy-pasting from web
# pages or word processors -- normalized to their plain ASCII equivalents so
# downstream regex matching (label lines, "<Company> is a..." patterns, etc.)
# isn't silently defeated by a different quote character.
_QUOTE_AND_DASH_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
}


def sanitize_raw_text(text: str) -> str:
    """
    Strip/normalize copy-paste artifacts from raw job-posting or resume text
    before extraction: smart quotes/dashes -> ASCII, zero-width and other
    invisible control characters removed, non-breaking spaces -> regular
    spaces. Does not remove legitimate punctuation, accented characters (e.g.
    "CYURÆ"), or newlines -- only characters that are invisible or that would
    silently break exact-string/regex matching.
    """
    if not text:
        return text

    for smart_char, plain_char in _QUOTE_AND_DASH_MAP.items():
        text = text.replace(smart_char, plain_char)

    # Non-breaking space and other Unicode space variants -> regular space.
    text = "".join(" " if unicodedata.category(ch) == "Zs" and ch != " " else ch for ch in text)

    # Zero-width characters and other invisible-but-non-whitespace control
    # chars (category "Cf" = "format", e.g. zero-width space/joiner, BOM).
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")

    return text


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


@lru_cache(maxsize=None)
def _keyword_pattern(keyword: str) -> re.Pattern:
    """
    Compile a word-boundary-aware regex for a keyword. \b only applies where the
    keyword actually starts/ends on a word character -- a keyword like "c++" or
    "c#" already ends on a non-word char, so \b there would never match and
    silently break the pattern; skip the boundary marker in that case.
    """
    escaped = re.escape(keyword)
    prefix = r"\b" if keyword[0].isalnum() else ""
    suffix = r"\b" if keyword[-1].isalnum() else ""
    return re.compile(prefix + escaped + suffix, re.IGNORECASE)


def _keyword_in(keyword: str, text: str) -> bool:
    return _keyword_pattern(keyword).search(text) is not None


def extract_skills(text: str) -> list[str]:
    skills = []

    # SKILL_KEYWORDS is a nested structure: category -> skill alias -> phrase list.
    # Flatten that nested structure and ask whether any alias is found in the text body.
    for skill_group in SKILL_KEYWORDS.values():
        if not isinstance(skill_group, dict):
            continue

        for skill_name, patterns in skill_group.items():
            if any(_keyword_in(keyword, text) for keyword in patterns):
                skills.append(skill_name)

    return sorted(set(skills))


def infer_industry_type(text: str) -> str | None:
    matched = []

    for industry, patterns in INDUSTRY_PATTERNS.items():
        if any(_keyword_in(keyword, text) for keyword in patterns):
            matched.append(industry)

    if matched:
        return matched[0]

    return "general"


def find_experience_years(text: str) -> int | None:
    lowered = text.lower()
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:yr|yrs|years|year)\s*(?:of)?\s*experience",
        r"experience\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*(?:yr|yrs|years|year)",
        r"(\d+(?:\.\d+)?)\s*(?:yr|yrs|years|year)\s*(?:working|developing|in)",
        r"(\d+(?:\.\d+)?)\s*(?:\+?)\s*years",
    ]

    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            try:
                return int(float(match.group(1)))
            except (ValueError, TypeError):
                return None

    return None
