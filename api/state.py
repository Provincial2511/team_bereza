from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import faiss

from src.config import Config
from src.embeddings import TextEmbedder
from src.faiss_store import FaissStore
from src.generator import LocalGenerator
from src.ocr import extract_text_from_pdf
from src.parser import GuidelineParserStub

logger = logging.getLogger(__name__)


@dataclass
class Session:
    session_id: str
    patient_text: str
    retrieved_sections: list[str]
    mode: str
    history: list[dict] = field(default_factory=list)


class AppState:
    """
    Singleton holding all heavy ML objects loaded once at startup.

    Attributes:
        embedder:  TextEmbedder for patient text.
        store:     FaissStore with pre-built guideline index.
        generator: LocalGenerator (Qwen2-7B-Instruct).
        sessions:  In-memory session store keyed by UUID.
    """

    def __init__(self) -> None:
        self.cfg = Config()
        # DEVICE env var overrides config.py (e.g. DEVICE=cuda uvicorn ...)
        env_device = os.getenv("DEVICE")
        if env_device:
            self.cfg.device = env_device.lower()
        self.embedder: TextEmbedder | None = None
        self.store: FaissStore | None = None
        self.generator: LocalGenerator | None = None
        self.sessions: dict[str, Session] = {}

    def load(self) -> None:
        """Load all ML components. Called once during FastAPI lifespan startup."""
        logger.info("Loading TextEmbedder...")
        self.embedder = TextEmbedder(model_name=self.cfg.embedder_model)

        index_dir = Path(self.cfg.faiss_index_path)
        if not (index_dir / "index.faiss").exists():
            logger.warning(
                "FAISS index not found at '%s' — building automatically from guidelines...",
                self.cfg.faiss_index_path,
            )
            self._build_index()

        logger.info("Loading FAISS index from '%s'...", self.cfg.faiss_index_path)
        temp = faiss.read_index(str(index_dir / "index.faiss"))
        self.store = FaissStore(dimension=temp.d)
        self.store.load(self.cfg.faiss_index_path)
        del temp
        logger.info("FAISS index loaded (%d vectors).", self.store.index.ntotal)

        logger.info("Loading generator model '%s'...", self.cfg.generator_model)
        self.generator = LocalGenerator(
            model_name=self.cfg.generator_model,
            device=self.cfg.device,
        )
        logger.info("All components ready.")

    def _build_index(self) -> None:
        """
        Build the FAISS index from all PDFs in cfg.guideline_path.

        Called automatically by load() when the index is missing.
        Requires self.embedder to be initialised first.
        """
        guideline_dir = Path(self.cfg.guideline_path)
        if not guideline_dir.exists():
            raise RuntimeError(
                f"Guideline directory not found: '{guideline_dir}'. "
                "Place clinical guideline PDFs there and restart."
            )

        pdf_files = sorted(guideline_dir.glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError(
                f"No PDF files found in '{guideline_dir}'. "
                "Add clinical guideline PDFs and restart."
            )

        logger.info(
            "Building FAISS index from %d PDFs in '%s'...",
            len(pdf_files),
            guideline_dir,
        )

        parser = GuidelineParserStub(
            chunk_size=self.cfg.chunk_size,
            overlap=self.cfg.overlap,
            model_name=self.cfg.embedder_model,
        )

        all_sections: list[str] = []
        all_metadata: list[dict] = []

        for pdf_path in pdf_files:
            logger.info("OCR: %s", pdf_path.name)
            try:
                text = extract_text_from_pdf(
                    str(pdf_path), languages=self.cfg.ocr_languages
                )
            except Exception as exc:
                logger.warning("Failed to OCR '%s': %s", pdf_path.name, exc)
                continue

            sections = parser.parse(text)
            if not sections:
                logger.warning("No chunks produced from '%s'.", pdf_path.name)
                continue

            meta = [{"source": pdf_path.name} for _ in sections]
            all_sections.extend(sections)
            all_metadata.extend(meta)
            logger.info("  → %d chunks from '%s'.", len(sections), pdf_path.name)

        if not all_sections:
            raise RuntimeError(
                "No guideline sections extracted from any PDF. "
                "Check OCR languages and PDF quality."
            )

        logger.info("Embedding %d chunks...", len(all_sections))
        embeddings = self.embedder.embed_batch(all_sections)

        store = FaissStore(dimension=embeddings.shape[1])
        store.add(all_sections, embeddings, all_metadata)
        store.save(self.cfg.faiss_index_path)
        logger.info(
            "FAISS index built and saved to '%s' (%d vectors).",
            self.cfg.faiss_index_path,
            store.index.ntotal,
        )

    def create_session(
        self,
        patient_text: str,
        retrieved_sections: list[str],
        mode: str,
    ) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(
            session_id=session_id,
            patient_text=patient_text,
            retrieved_sections=retrieved_sections,
            mode=mode,
        )
        return session_id

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)


# Module-level singleton — imported by api/main.py
app_state = AppState()
