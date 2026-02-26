from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Generate dense vector embeddings for texts using a local sentence-transformers model.

    Supports two families of models transparently:

    * Standard models (e.g. ``paraphrase-multilingual-MiniLM-L12-v2``):
      text is passed as-is to the encoder.

    * E5 models (e.g. ``intfloat/multilingual-e5-small``):
      the official E5 recipe requires asymmetric prefixes —
      ``"query: "`` for retrieval queries and ``"passage: "`` for
      indexed documents.  The correct prefix is applied automatically
      based on which method is called:

      - :meth:`embed_text`  → query embedding  (``"query: "`` prefix)
      - :meth:`embed_batch` → passage embeddings (``"passage: "`` prefix)

    All embeddings are L2-normalised so that ``IndexFlatL2`` over
    normalised vectors is equivalent to cosine similarity search.

    Output shapes:
    - embed_text:  (1, dimension)
    - embed_batch: (n_items, dimension)
    """

    # E5 model identifiers that require query/passage prefix injection.
    _E5_MODELS: frozenset[str] = frozenset(
        {
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
            "intfloat/multilingual-e5-large",
            "intfloat/multilingual-e5-small-instruct",
            "intfloat/multilingual-e5-base-instruct",
            "intfloat/e5-small",
            "intfloat/e5-base",
            "intfloat/e5-large",
            "intfloat/e5-small-v2",
            "intfloat/e5-base-v2",
            "intfloat/e5-large-v2",
        }
    )

    def __init__(
        self, model_name: str = "intfloat/multilingual-e5-small"
    ) -> None:
        """
        Initialize the text embedder.

        Args:
            model_name: HuggingFace model name or local path.
                        E5-family models are auto-detected and receive
                        the appropriate query/passage prefix at inference time.
        """
        self.model_name = model_name
        self._is_e5 = model_name in self._E5_MODELS
        self.model: SentenceTransformer = SentenceTransformer(model_name, device="cpu")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalise rows in-place and return the array."""
        norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single *query* string (used at retrieval time).

        For E5 models the ``"query: "`` prefix is added automatically.

        Args:
            text: Non-empty query string.

        Returns:
            Normalised embedding of shape ``(1, dimension)``, dtype float32.

        Raises:
            ValueError: If *text* is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        query = f"query: {text}" if self._is_e5 else text
        emb = self.model.encode(query, convert_to_numpy=True).astype(np.float32)
        return self._normalise(emb.reshape(1, -1))

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of *passage* strings (used when building the index).

        For E5 models the ``"passage: "`` prefix is added to every text
        automatically.

        Args:
            texts: Non-empty list of passage strings.

        Returns:
            Normalised embeddings of shape ``(n_items, dimension)``, dtype float32.
            Order matches the input list.

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty.")

        passages = [f"passage: {t}" for t in texts] if self._is_e5 else texts
        embs = self.model.encode(passages, convert_to_numpy=True).astype(np.float32)
        return self._normalise(embs)
