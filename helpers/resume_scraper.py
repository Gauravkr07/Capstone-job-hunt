import re
from typing import Any

from helpers.text_extraction import extract_skills, find_experience_years, infer_industry_type, normalize_text, sanitize_raw_text


def _scrape_skills(text: str) -> list[str]:
    skills = extract_skills(text)
    lowered = text.lower()

    # Fallback: pull common skill tokens from a "Skills" section
    skills_section_match = re.search(r"skills\s*[:\-](.*?)(?:experience|education|projects|summary|certifications|contact)", lowered, re.IGNORECASE | re.DOTALL)
    if skills_section_match:
        section = skills_section_match.group(1)
        tokens = [token.strip() for token in section.replace(",", " ").split()]
        for token in tokens:
            if len(token) > 2:
                skills.append(token)

    return sorted(set(skills))


def scrape_resume(resume_text: str) -> dict[str, Any]:
    """
    Build a basic profile summary out of the raw resume content.

    Returns resume_text (sanitized), skills, experience, and industry_type.
    """
    if not resume_text or not str(resume_text).strip():
        raise ValueError("Resume text is empty")

    resume_text = sanitize_raw_text(resume_text)
    normalized = normalize_text(resume_text)
    skills = _scrape_skills(normalized)
    experience = find_experience_years(normalized)
    industry = infer_industry_type(normalized)

    return {
        "resume_text": resume_text,
        "skills": skills,
        "experience": experience,
        "industry_type": industry,
    }
