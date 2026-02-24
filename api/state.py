from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import faiss

from src.config import Config
from src.embeddings import TextEmbedder
from src.faiss_store import FaissStore
from src.generator import LocalGenerator

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
        self.embedder: TextEmbedder | None = None
        self.store: FaissStore | None = None
        self.generator: LocalGenerator | None = None
        self.sessions: dict[str, Session] = {}

    def load(self) -> None:
        """Load all ML components. Called once during FastAPI lifespan startup."""
        logger.info("Loading TextEmbedder...")
        self.embedder = TextEmbedder(model_name=self.cfg.embedder_model)

        logger.info("Loading FAISS index from '%s'...", self.cfg.faiss_index_path)
        index_dir = Path(self.cfg.faiss_index_path)
        if not (index_dir / "index.faiss").exists():
            raise RuntimeError(
                f"FAISS index not found at '{self.cfg.faiss_index_path}'. "
                "Run 'python main.py' first to build the index."
            )
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
