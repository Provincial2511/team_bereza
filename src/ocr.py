from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import easyocr

logger = logging.getLogger(__name__)

# Module-level reader cache: avoids re-loading EasyOCR weights on repeated calls.
_READER_CACHE: dict[tuple[str, ...], easyocr.Reader] = {}


def _get_reader(languages: list[str]) -> easyocr.Reader:
    """Return a cached EasyOCR Reader for the given language list."""
    key = tuple(sorted(languages))
    if key not in _READER_CACHE:
        logger.info("Initializing EasyOCR reader for languages: %s", languages)
        _READER_CACHE[key] = easyocr.Reader(languages, gpu=False)
    return _READER_CACHE[key]


def _has_text_layer(pdf_path: str) -> bool:
    """Return True if the PDF contains a native selectable text layer."""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            if page.get_text().strip():
                return True
    return False


def _extract_text_native(pdf_path: str) -> str:
    """Extract text directly from a PDF that has a native text layer."""
    parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n\n".join(parts)


def _extract_text_ocr(pdf_path: str, languages: list[str]) -> str:
    """
    Extract text from a scanned PDF using EasyOCR.

    Each page is rendered to a high-resolution image via PyMuPDF,
    then passed to EasyOCR in paragraph mode.
    """
    reader = _get_reader(languages)
    parts: list[str] = []

    with fitz.open(pdf_path) as doc:
        total = len(doc)
        for idx, page in enumerate(doc, start=1):
            logger.info("Running OCR on page %d / %d", idx, total)
            pix = page.get_pixmap(dpi=300)

            # Convert PyMuPDF pixmap to a numpy array (RGB or RGBA).
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            # EasyOCR expects RGB; drop the alpha channel if present.
            if pix.n == 4:
                img = img[:, :, :3]

            results: list[str] = reader.readtext(img, detail=0, paragraph=True)
            parts.append("\n".join(results))

    return "\n\n".join(parts)


def _normalize(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph structure.

    - Collapses multiple spaces / tabs to a single space within a line.
    - Reduces runs of 3+ newlines to a paragraph break (two newlines).
    - Strips trailing whitespace from each line.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def extract_text_from_pdf(
    path: str,
    languages: Optional[list[str]] = None,
) -> str:
    """
    Extract text from a PDF file — native or scanned.

    Detection logic:
    - If the PDF has a selectable text layer → extract directly via PyMuPDF
      (fast, lossless, preserves structure).
    - Otherwise → render each page to a 300-DPI image and run EasyOCR.

    Args:
        path: Path to the PDF file.
        languages: BCP-47 / EasyOCR language codes for OCR fallback.
                   Defaults to ``["ru"]`` (Russian).

    Returns:
        Normalized extracted text with paragraph breaks preserved (``\\n\\n``
        between logical blocks).

    Raises:
        FileNotFoundError: If the PDF does not exist at *path*.
        RuntimeError: If text extraction fails for any reason.
    """
    if languages is None:
        languages = ["ru"]

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        if _has_text_layer(str(pdf_path)):
            logger.info("Native text layer detected in '%s'.", pdf_path.name)
            raw = _extract_text_native(str(pdf_path))
        else:
            logger.info(
                "No text layer found in '%s'; running EasyOCR.", pdf_path.name
            )
            raw = _extract_text_ocr(str(pdf_path), languages)
    except Exception as exc:
        raise RuntimeError(
            f"Text extraction failed for '{pdf_path}': {exc}"
        ) from exc

    return _normalize(raw)
