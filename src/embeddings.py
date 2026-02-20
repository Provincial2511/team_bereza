from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Generate dense vector embeddings for texts using a local sentence-transformers model.

    This class:
    - Loads a sentence-transformers model locally (no external APIs).
    - Generates embeddings for single texts or batches.
    - Normalizes embeddings to unit length (L2 normalization) for cosine similarity.
    - Returns embeddings as numpy arrays with dtype float32.

    Normalization is required because FAISS IndexFlatL2 can compute cosine similarity
    on L2-normalized vectors using the L2 distance metric. After normalization,
    L2 distance between normalized vectors is equivalent to cosine distance.

    Output shapes:
    - embed_text: (1, dimension)
    - embed_batch: (n_items, dimension)
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        """
        Initialize the text embedder with a local sentence-transformers model.

        Args:
            model_name: Name or path of the sentence-transformers model.
        """
        self.model: SentenceTransformer = SentenceTransformer(model_name, device="cpu")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string.

        Returns:
            Normalized embedding array of shape (1, dimension) with dtype float32.

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        embedding = self.model.encode(text, convert_to_numpy=True)
        embedding = embedding.astype(np.float32)
        embedding = embedding.reshape(1, -1)

        # L2 normalization: divide by L2 norm
        norm = np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        embedding = embedding / norm

        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input text strings.

        Returns:
            Normalized embedding array of shape (n_items, dimension) with dtype float32.
            Order matches the input list.

        Raises:
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty.")

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        # L2 normalization: divide by L2 norm for each row.
        norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        return embeddings

