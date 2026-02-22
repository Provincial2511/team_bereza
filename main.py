from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List

import faiss
from docx import Document

from src.config import Config
from src.embeddings import TextEmbedder
from src.faiss_store import FaissStore
from src.generator import LocalGenerator
from src.ocr import extract_text_from_pdf
from src.parser import GuidelineParserStub


def load_patient_docx(path: str) -> str:
    """
    Load patient information from a DOCX file as a single text string.

    Args:
        path: Path to the DOCX file.

    Returns:
        Concatenated text from all non-empty paragraphs in the document.

    Raises:
        FileNotFoundError: If the DOCX file does not exist.
    """
    docx_path = Path(path)
    if not docx_path.is_file():
        raise FileNotFoundError(f"Patient DOCX file not found: {docx_path}")

    document = Document(str(docx_path))
    paragraphs: List[str] = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def main() -> None:
    """
    Run the minimal local RAG pipeline for clinical recommendation generation.

    Pipeline steps:
    1. Load or build FAISS index from a clinical guideline PDF.
    2. Optionally add new guidelines to an existing index.
    3. Embed patient DOCX text and retrieve the most relevant guideline sections.
    4. Generate a clinical recommendation with a local LLM.
    5. Save the result as a versioned JSON file.
    """
    cfg = Config()

    timestamp_start = datetime.now().isoformat()
    pipeline_start = time()
    stage_durations: Dict[str, float] = {}

    if cfg.mode not in ("doctor", "patient"):
        raise ValueError("cfg.mode must be 'doctor' or 'patient'")

    # --- FAISS index: load or build ---
    index_dir = Path(cfg.faiss_index_path)
    index_file = index_dir / "index.faiss"
    texts_file = index_dir / "texts.pkl"
    metadata_file = index_dir / "metadata.pkl"

    if index_file.exists() and texts_file.exists() and metadata_file.exists():
        print("Loading existing FAISS index from disk...")
        temp_index = faiss.read_index(str(index_file))
        dimension = temp_index.d
        del temp_index

        store = FaissStore(dimension=dimension)
        store.load(cfg.faiss_index_path)
        print(f"FAISS index loaded ({store.index.ntotal} vectors).\n")

        embedder = TextEmbedder(model_name=cfg.embedder_model)

        # Optional: add a new guideline PDF to the existing index.
        if cfg.add_new_guidelines and Path(cfg.new_guideline_pdf_path).exists():
            print("Processing new guideline PDF and adding to existing index...")
            try:
                new_text = extract_text_from_pdf(
                    cfg.new_guideline_pdf_path, languages=cfg.ocr_languages
                )
                new_parser = GuidelineParserStub(
                    chunk_size=cfg.chunk_size,
                    overlap=cfg.overlap,
                    model_name=cfg.embedder_model,
                )
                new_sections = new_parser.parse(new_text)
                if new_sections:
                    new_embeddings = embedder.embed_batch(new_sections)
                    new_metadata = [{} for _ in range(len(new_sections))]
                    store.add(new_sections, new_embeddings, new_metadata)
                    store.save(cfg.faiss_index_path)
                    print(
                        f"New guidelines added. Updated index: "
                        f"{store.index.ntotal} vectors.\n"
                    )
            except Exception as exc:
                print(f"Warning: Failed to add new guidelines: {exc}\n")
    else:
        print("FAISS index not found. Building from PDF...")

        # 1. OCR / text extraction.
        t0 = time()
        try:
            guideline_text = extract_text_from_pdf(
                cfg.guideline_pdf_path, languages=cfg.ocr_languages
            )
            print("Guideline text extracted.\n")
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return
        stage_durations["ocr"] = time() - t0

        if not guideline_text.strip():
            print("Error: text extraction returned empty string.")
            return

        # 2. Chunking.
        t0 = time()
        guideline_parser = GuidelineParserStub(
            chunk_size=cfg.chunk_size,
            overlap=cfg.overlap,
            model_name=cfg.embedder_model,
        )
        guideline_sections = guideline_parser.parse(guideline_text)
        print(f"Guideline split into {len(guideline_sections)} chunks.\n")
        stage_durations["chunking"] = time() - t0

        if not guideline_sections:
            print("Error: chunking produced no sections.")
            return

        # 3. Embed sections.
        t0 = time()
        embedder = TextEmbedder(model_name=cfg.embedder_model)
        section_embeddings = embedder.embed_batch(guideline_sections)
        print("Guideline sections embedded.\n")
        stage_durations["embedding"] = time() - t0

        if section_embeddings.size == 0:
            print("Error: embeddings are empty.")
            return

        n_items, dimension = section_embeddings.shape

        # 4. Build and persist FAISS index.
        store = FaissStore(dimension=dimension)
        empty_metadata = [{} for _ in range(n_items)]
        store.add(guideline_sections, section_embeddings, empty_metadata)
        store.save(cfg.faiss_index_path)
        print(f"FAISS index saved to '{cfg.faiss_index_path}'.\n")

    # --- Patient data ---
    try:
        patient_text = load_patient_docx(cfg.patient_docx_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    if not patient_text.strip():
        print("Error: patient DOCX text is empty.")
        return
    print("Patient DOCX loaded.\n")

    # --- Patient embedding ---
    t0 = time()
    patient_embedding = embedder.embed_text(patient_text)
    stage_durations["patient_embedding"] = time() - t0

    if patient_embedding.shape[1] != store.dimension:
        raise RuntimeError(
            f"Embedding dimension mismatch: patient={patient_embedding.shape[1]}, "
            f"guidelines={store.dimension}."
        )
    print("Patient text embedded.\n")

    # --- Retrieval ---
    t0 = time()
    search_results = store.search(patient_embedding[0], top_k=cfg.top_k)
    retrieved_sections: List[str] = [r["text"] for r in search_results]
    stage_durations["retrieval"] = time() - t0

    if not retrieved_sections:
        print("Error: FAISS returned no results.")
        return
    print(f"Retrieved {len(retrieved_sections)} guideline sections.\n")

    # --- Generation ---
    generator = LocalGenerator(model_name=cfg.generator_model, device=cfg.device)
    print("Generator initialized.\n")

    t0 = time()
    final_response = generator.generate(
        patient_text=patient_text,
        retrieved_sections=retrieved_sections,
        mode=cfg.mode,
        max_new_tokens=cfg.max_new_tokens,
    )
    stage_durations["generation"] = time() - t0
    print("Recommendation generated.\n")

    # --- Save versioned JSON ---
    timestamp_end = datetime.now().isoformat()
    total_duration = time() - pipeline_start

    response_dir = Path(cfg.response_dir)
    response_dir.mkdir(parents=True, exist_ok=True)

    version_pattern = re.compile(r"version_(\d+)\.json$")
    max_version = -1
    for fp in response_dir.iterdir():
        if fp.is_file():
            m = version_pattern.match(fp.name)
            if m:
                max_version = max(max_version, int(m.group(1)))

    next_version = max_version + 1
    response_metadata = {
        "version": next_version,
        "mode": cfg.mode,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "total_duration_seconds": round(total_duration, 3),
        "model_name": cfg.generator_model,
        "stage_durations": {k: round(v, 3) for k, v in stage_durations.items()},
        "final_response": final_response,
    }

    response_file = response_dir / f"version_{next_version}.json"
    with response_file.open("w", encoding="utf-8") as f:
        json.dump(response_metadata, f, ensure_ascii=False, indent=2)

    print(f"Response saved to '{response_file}'.\n")

    # --- Output ---
    print("=== Clinical Recommendation ===")
    print(final_response)


if __name__ == "__main__":
    main()
