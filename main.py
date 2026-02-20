from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List

import faiss
import numpy as np
import torch
from docx import Document

from src.embeddings import TextEmbedder
from src.faiss_store import FaissStore
from src.generator import LocalGenerator
from src.ocr import extract_text_from_scanned_pdf
from src.parser import GuidelineParserStub


def load_patient_docx(path: str) -> str:
    """
    Load patient information from a DOCX file as a single text string.

    Args:
        path: Path to the DOCX file.

    Returns:
        Concatenated text from all paragraphs in the document.

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

    The pipeline performs the following steps:
    1. Check for existing FAISS index; load if present, otherwise:
       - OCR on scanned clinical guideline PDF.
       - Guideline text parsing into sections.
       - Embedding generation for guideline sections.
       - FAISS index construction and save to disk.
    2. Patient DOCX loading and embedding.
    3. Nearest-neighbor retrieval of guideline sections.
    4. Local LLM generation of a clinical recommendation.
    """
    # Start pipeline timing
    timestamp_start = datetime.now().isoformat()
    pipeline_start_time = time()
    stage_durations: Dict[str, float] = {}
    
    # FAISS index storage path (configurable)
    faiss_index_path = "data/faiss_index"
    guideline_pdf_path = "data/clinical_guideline/cg_30_5_lung_cancer.pdf"
    patient_docx_path = "data/input_example/case_example_2.docx"
    
    # Set to True to add new guidelines to existing index (if index exists)
    add_new_guidelines = False
    new_guideline_pdf_path = "data/clinical_guideline/new_guideline.pdf"
    
    # Model name for generation (will be set when generator is initialized)
    model_name = ""

    # Check if FAISS index exists
    index_dir = Path(faiss_index_path)
    index_file = index_dir / "index.faiss"
    texts_file = index_dir / "texts.pkl"
    metadata_file = index_dir / "metadata.pkl"

    if index_file.exists() and texts_file.exists() and metadata_file.exists():
        # Load existing index
        print("Loading existing FAISS index from disk...")
        # Read index to get dimension
        temp_index = faiss.read_index(str(index_file))
        dimension = temp_index.d
        del temp_index

        store = FaissStore(dimension=dimension)
        store.load(faiss_index_path)
        print(f"FAISS index loaded successfully ({store.index.ntotal} vectors).\n")

        # Initialize embedder for patient text embedding (needed later)
        embedder = TextEmbedder()

        # Optional: Add new guidelines to existing index
        if add_new_guidelines and Path(new_guideline_pdf_path).exists():
            print("Processing new guideline PDF and adding to existing index...")
            try:
                new_guideline_text = extract_text_from_scanned_pdf(new_guideline_pdf_path)
                new_guideline_parser = GuidelineParserStub(chunk_size=500, overlap=50)
                new_guideline_sections = new_guideline_parser.parse(new_guideline_text)
                if new_guideline_sections:
                    new_section_embeddings = embedder.embed_batch(new_guideline_sections)
                    new_empty_metadata = [{} for _ in range(len(new_guideline_sections))]
                    store.add(new_guideline_sections, new_section_embeddings, new_empty_metadata)
                    store.save(faiss_index_path)
                    print(f"New guidelines added. Updated index has {store.index.ntotal} vectors.\n")
            except Exception as exc:
                print(f"Warning: Failed to add new guidelines: {exc}\n")
    else:
        # Build new index from PDF
        print("FAISS index not found. Building new index from PDF...")
        # 1. Extract guideline text via OCR.
        ocr_start = time()
        try:
            guideline_text = extract_text_from_scanned_pdf(guideline_pdf_path)
            print("Guideline text extracted successfully.\n")
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return
        stage_durations["ocr"] = time() - ocr_start

        if not guideline_text or not guideline_text.strip():
            print("Error: OCR returned empty guideline text.")
            return

        # 2. Parse guideline text into sections.
        chunking_start = time()
        guideline_parser = GuidelineParserStub(chunk_size=500, overlap=50)
        guideline_sections = guideline_parser.parse(guideline_text)
        print("Guideline text parsed successfully.\n")
        stage_durations["chunking"] = time() - chunking_start

        if not guideline_sections:
            print("Error: Parsed guideline sections are empty.")
            return

        # 3. Initialize embedder and generate embeddings for guideline sections.
        embedding_start = time()
        embedder = TextEmbedder()
        section_embeddings = embedder.embed_batch(guideline_sections)
        print("Guideline sections embedded successfully.\n")
        stage_durations["embedding"] = time() - embedding_start

        if section_embeddings.size == 0:
            print("Error: Embeddings for guideline sections are empty.")
            return

        n_items, dimension = section_embeddings.shape

        # 4. Initialize FAISS store and add guideline sections.
        store = FaissStore(dimension=dimension)
        empty_metadata = [{} for _ in range(n_items)]
        store.add(guideline_sections, section_embeddings, empty_metadata)
        print("Guideline sections added to FAISS store successfully.\n")

        # Save FAISS index to disk
        store.save(faiss_index_path)
        print(f"FAISS index saved to {faiss_index_path}\n")

    # 5. Load patient DOCX and extract text.
    try:
        patient_text = load_patient_docx(patient_docx_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return
    print("Patient DOCX text loaded successfully.\n")

    if not patient_text or not patient_text.strip():
        print("Error: Patient DOCX text is empty.")
        return

    # 6. Generate embedding for patient text.
    patient_embedding_start = time()
    patient_embedding = embedder.embed_text(patient_text)
    print("Patient text embedded successfully.\n")
    stage_durations["patient_embedding"] = time() - patient_embedding_start
    
    if patient_embedding.shape[1] != store.dimension:
        raise RuntimeError(
            f"Embedding dimension mismatch between patient ({patient_embedding.shape[1]}) "
            f"and guidelines ({store.dimension})."
        )
    print("Patient embedding dimension check passed.\n")
    
    # 7. Retrieve most relevant guideline sections.
    retrieval_start = time()
    top_k = 5
    query_vector = patient_embedding[0]  # shape (dimension,)
    search_results = store.search(query_vector, top_k=top_k)
    retrieved_sections: List[str] = [result["text"] for result in search_results]
    print("Most relevant guideline sections retrieved successfully.\n")
    stage_durations["retrieval"] = time() - retrieval_start
    
    if not retrieved_sections:
        print("Error: No guideline sections retrieved from FAISS.")
        return

    # 8. Initialize local generator and produce final recommendation.
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    generator = LocalGenerator(
        model_name="Qwen/Qwen2-7B-Instruct",
        device=device,
    )
    model_name = generator.model_name
    print("Local generator initialized successfully.\n")
    
    generation_start = time()
    final_response = generator.generate(
        patient_text=patient_text,
        retrieved_sections=retrieved_sections,
        max_new_tokens=2048,
    )
    stage_durations["generation"] = time() - generation_start
    print("Final recommendation generated successfully.\n")
    
    # 9. Calculate total duration and prepare metadata
    timestamp_end = datetime.now().isoformat()
    total_duration_seconds = time() - pipeline_start_time
    
    # 10. Save response with versioning and metadata.
    response_dir = Path("data/generator_response")
    response_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the highest existing version number
    version_pattern = re.compile(r"version_(\d+)\.json$")
    max_version = -1
    
    for file_path in response_dir.iterdir():
        if file_path.is_file():
            match = version_pattern.match(file_path.name)
            if match:
                version_num = int(match.group(1))
                max_version = max(max_version, version_num)
    
    # Determine next version number
    next_version = max_version + 1
    
    # Prepare metadata dictionary
    response_metadata = {
        "version": next_version,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "total_duration_seconds": round(total_duration_seconds, 3),
        "model_name": model_name,
        "stage_durations": {k: round(v, 3) for k, v in stage_durations.items()},
        "final_response": final_response,
    }
    
    # Save response to JSON file
    response_file = response_dir / f"version_{next_version}.json"
    with response_file.open("w", encoding="utf-8") as f:
        json.dump(response_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Response saved to {response_file}\n")
    
    # 10. Output result.
    print("=== Clinical Recommendation ===")
    print(final_response)


if __name__ == "__main__":
    main()

