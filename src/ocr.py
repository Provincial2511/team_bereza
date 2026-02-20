from __future__ import annotations

import logging
from pathlib import Path

from pdf2image import convert_from_path
import pytesseract

logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_scanned_pdf(path: str) -> str:
    """
    Extract raw text from a scanned PDF clinical guideline using OCR.

    This function:
    - Converts each PDF page to a PIL image with ``pdf2image``.
    - Runs Tesseract OCR on each page with Russian language support.
    - Concatenates the text from all pages into a single string.
    - Normalizes excessive whitespace.

    Args:
        path: Path to the PDF file.

    Returns:
        A single string containing OCR text from all pages.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If OCR or PDF processing fails.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        pages = convert_from_path(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to convert PDF to images: {exc}") from exc

    page_texts: list[str] = []
    for index, page in enumerate(pages, start=1):
        logger.info("Running OCR on page %d", index)
        try:
            text = pytesseract.image_to_string(page, lang="rus")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OCR failed on page {index}: {exc}") from exc
        page_texts.append(text)

    combined = "\n\n".join(page_texts)
    # Normalize excessive whitespace while preserving basic spacing.
    normalized = " ".join(combined.split())
    return normalized

