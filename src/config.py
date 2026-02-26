from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Paths ---
    faiss_index_path: str = "data/faiss_index"
    guideline_path: str = "data/clinical_guideline"
    patient_docx_path: str = "data/input_example/case_example_2.docx"
    response_dir: str = "data/generator_response"
    new_guideline_pdf_path: str = "data/clinical_guideline/new_guideline.pdf"

    # --- Models ---
    embedder_model: str = "intfloat/multilingual-e5-small"
    generator_model: str = "Qwen/Qwen2-7B-Instruct"

    # --- OCR ---
    ocr_languages: list[str] = field(default_factory=lambda: ["ru"])

    # --- Chunking ---
    chunk_size: int = 500
    overlap: int = 50

    # --- Retrieval ---
    top_k: int = 8
    # L2 distance threshold for retrieved chunks (L2-normalized vectors,
    # so range is [0, 2]; lower = more similar).
    # 1.2 ≈ cosine similarity 0.28 — filters clearly irrelevant chunks.
    # If fewer than 2 chunks pass the threshold, all top_k are used as fallback.
    retrieval_score_threshold: float = 1.1

    # --- Generation ---
    max_new_tokens: int = 1500
    device: str = "cpu"
    mode: str = "doctor"  # "doctor" | "patient"

    # --- Flags ---
    add_new_guidelines: bool = False
