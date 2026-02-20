from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np


class FaissStore:
    """
    Minimal FAISS-based vector store for local RAG.

    Uses:
    - `faiss.IndexFlatL2` for vector similarity search.
    - In-memory lists for texts and metadata.
    - Separate on-disk files for index, texts, and metadata.
    """

    def __init__(self, dimension: int) -> None:
        """
        Initialize the FAISS store.

        Args:
            dimension: Dimensionality of the embedding vectors.
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive.")
        self.dimension: int = dimension
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Add texts, embeddings, and metadata to the store.

        Args:
            texts: List of text strings corresponding to embeddings.
            embeddings: Numpy array of shape (n_items, dimension).
            metadata: List of metadata dicts per item.

        Raises:
            ValueError: If lengths or dimensions do not match.
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("embeddings must be a numpy.ndarray.")

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array of shape (n_items, dimension).")

        n_items, dim = embeddings.shape
        if dim != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {dim}."
            )

        if len(texts) != n_items or len(metadata) != n_items:
            raise ValueError(
                "Lengths of texts, embeddings, and metadata must match."
            )

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbors of a query embedding.

        Args:
            query_embedding: 1D or 2D numpy array representing the query vector.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of dictionaries of the form:
                {
                    "text": str,
                    "metadata": dict,
                    "score": float  # L2 distance
                }

        Raises:
            ValueError: If the query dimension is invalid.
            RuntimeError: If the index is empty.
        """
        if self.index.ntotal == 0:
            raise RuntimeError("FAISS index is empty; add vectors before searching.")

        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("query_embedding must be a numpy.ndarray.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        elif query_embedding.ndim != 2:
            raise ValueError("query_embedding must be 1D or 2D numpy array.")

        _, dim = query_embedding.shape
        if dim != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, got {dim}."
            )

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        k = min(top_k, len(self.texts))
        distances, indices = self.index.search(query_embedding, k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append(
                {
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "score": float(dist),
                }
            )
        return results

    def save(self, path: str) -> None:
        """
        Save the FAISS index, texts, and metadata to disk.

        The given path is treated as a directory; files are stored as:
        - index:   <path>/index.faiss
        - texts:   <path>/texts.pkl
        - metadata:<path>/metadata.pkl

        Args:
            path: Directory path where the store should be saved.
        """
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)

        index_path = base / "index.faiss"
        texts_path = base / "texts.pkl"
        metadata_path = base / "metadata.pkl"

        faiss.write_index(self.index, str(index_path))

        with texts_path.open("wb") as f_texts:
            pickle.dump(self.texts, f_texts)

        with metadata_path.open("wb") as f_meta:
            pickle.dump(self.metadata, f_meta)

    def load(self, path: str) -> None:
        """
        Load the FAISS index, texts, and metadata from disk.

        The given path is treated as a directory; the method expects:
        - <path>/index.faiss
        - <path>/texts.pkl
        - <path>/metadata.pkl

        Args:
            path: Directory path from which the store should be loaded.

        Raises:
            FileNotFoundError: If any of the required files are missing.
            RuntimeError: If loaded index dimension does not match existing dimension.
        """
        base = Path(path)
        index_path = base / "index.faiss"
        texts_path = base / "texts.pkl"
        metadata_path = base / "metadata.pkl"

        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not texts_path.is_file():
            raise FileNotFoundError(f"Texts file not found: {texts_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        index = faiss.read_index(str(index_path))
        if index.d != self.dimension:
            raise RuntimeError(
                f"Loaded index dimension {index.d} does not match store dimension {self.dimension}."
            )

        self.index = index

        with texts_path.open("rb") as f_texts:
            self.texts = pickle.load(f_texts)

        with metadata_path.open("rb") as f_meta:
            self.metadata = pickle.load(f_meta)
