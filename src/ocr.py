from __future__ import annotations

import logging
import re
from collections import Counter
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


def get_pdf_title(pdf_path: str) -> str:
    """
    Return a human-readable document title for use in citations.

    Priority:
    1. PDF metadata ``title`` field (set by the document author).
    2. First non-empty line of page 0 that is long enough to be a real title.
    3. Empty string (caller should fall back to the filename).
    """
    with fitz.open(pdf_path) as doc:
        meta_title = (doc.metadata or {}).get("title", "").strip()
        if meta_title and len(meta_title) > 10:
            return meta_title
        # Heuristic: first substantive line on page 0 is often the document title.
        if len(doc) > 0:
            for line in doc[0].get_text(sort=True).splitlines():
                line = line.strip()
                if len(line) > 15 and not line.isdigit():
                    return line
    return ""


def _has_text_layer(pdf_path: str) -> bool:
    """Return True if the PDF contains a native selectable text layer."""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            if page.get_text().strip():
                return True
    return False


def _extract_text_native(pdf_path: str) -> str:
    """
    Extract text from a digital PDF (native text layer).

    Improvements over bare ``page.get_text()``:

    - ``sort=True``: PyMuPDF reads blocks in correct reading order
      (top-to-bottom, left-to-right), which handles most 2-column layouts.
    - ``TEXT_DEHYPHENATE``: re-joins words broken across line endings.
    - Header / footer removal: lines that appear verbatim on ≥30 % of pages
      (page numbers, document title in running header, ministry name, etc.)
      are stripped from every page before chunking.
    """
    flags = fitz.TEXT_DEHYPHENATE

    # First pass — collect raw lines from every page.
    raw_pages: list[list[str]] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            lines = [
                ln.strip()
                for ln in page.get_text(sort=True, flags=flags).splitlines()
                if ln.strip()
            ]
            raw_pages.append(lines)

    if not raw_pages:
        return ""

    # Identify repeated lines (headers / footers / page numbers).
    all_lines: list[str] = [ln for page_lines in raw_pages for ln in page_lines]
    counts = Counter(all_lines)
    repeat_threshold = max(3, len(raw_pages) * 0.30)
    noise_lines: set[str] = {
        ln
        for ln, cnt in counts.items()
        if cnt >= repeat_threshold or ln.isdigit()
    }

    # Second pass — rebuild page text without noise.
    parts: list[str] = []
    for page_lines in raw_pages:
        clean = [ln for ln in page_lines if ln not in noise_lines]
        if clean:
            parts.append("\n".join(clean))

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
