import io
import re

from helpers.logger import get_logger

logger = get_logger("resume_file_parser")


def _extract_pdf_text(content: bytes) -> str:
    """
    pdfminer.six rebuilds the spaces between words from glyph positions. Many
    CV generators place each word separately without space characters, and
    simpler extractors (e.g. pypdf) then return "SoftwareEngineerwith5+years...",
    which wrecks both embeddings and keyword matching.
    """
    from pdfminer.high_level import extract_text

    text = extract_text(io.BytesIO(content))
    text = re.sub(r"\(cid:\d+\)", "•", text)  # unmapped glyphs -- in CVs almost always the bullet symbol
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)  # re-join words hyphenated across lines
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_docx_text(content: bytes) -> str:
    from docx import Document

    document = Document(io.BytesIO(content))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(paragraphs)


def extract_resume_text(content: bytes, filename: str | None = None, content_type: str | None = None) -> str:
    """
    Extract plain text from resume bytes. Detects PDF/DOCX by content-type or
    filename extension; falls back to treating the bytes as plain text/HTML.
    """
    lowered_name = (filename or "").lower()
    lowered_type = (content_type or "").lower()

    is_pdf = lowered_name.endswith(".pdf") or "pdf" in lowered_type
    is_docx = lowered_name.endswith(".docx") or "wordprocessingml" in lowered_type

    if is_pdf:
        logger.info("Extracting text from PDF resume (%s)", filename or "unnamed")
        return _extract_pdf_text(content)

    if is_docx:
        logger.info("Extracting text from DOCX resume (%s)", filename or "unnamed")
        return _extract_docx_text(content)

    logger.info("Treating resume content as plain text (%s)", filename or "unnamed")
    return content.decode("utf-8", errors="ignore")
