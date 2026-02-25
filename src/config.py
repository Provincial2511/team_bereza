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
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generator_model: str = "Qwen/Qwen2-7B-Instruct"

    # --- OCR ---
    ocr_languages: list[str] = field(default_factory=lambda: ["ru"])

    # --- Chunking ---
    chunk_size: int = 500
    overlap: int = 50

    # --- Retrieval ---
    top_k: int = 5

    # --- Generation ---
    max_new_tokens: int = 1024
    device: str = "cpu"
    mode: str = "doctor"  # "doctor" | "patient"

    # --- Flags ---
    add_new_guidelines: bool = False
