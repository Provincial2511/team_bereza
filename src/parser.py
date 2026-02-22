from __future__ import annotations

import json
import re
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GuidelineParser:
    """
    Parse raw OCR guideline text into structured sections using a local LLM.

    The model is expected to return strictly valid JSON containing
    the requested guideline sections.

    Note: This class is implemented but not used in the current main pipeline.
    The production flow uses GuidelineParserStub instead.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "cpu",
        max_new_tokens: int = 512,
    ) -> None:
        """
        Initialize the guideline parser with a local HuggingFace model.

        Args:
            model_name: Name or path of the HuggingFace model to use.
            device: Device identifier, e.g. ``"cpu"`` or ``"cuda"``.
            max_new_tokens: Maximum number of new tokens to generate.
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            self.device
        )

    def parse(self, text: str) -> Dict[str, str]:
        """
        Parse raw OCR text into structured guideline sections.

        The model is prompted to return JSON with the following keys:
        - Краткая информация
        - Диагностика
        - Лечение
        - Реабилитация
        - Профилактика
        - Организация ухода
        - Факторы прогноза

        Args:
            text: Raw OCR text of the clinical guideline.

        Returns:
            Dictionary containing structured sections.

        Raises:
            RuntimeError: If the model output cannot be parsed as JSON.
        """
        prompt = (
            "Тебе дан сырой текст клинической рекомендации.\n"
            "Извлеки следующие секции и верни только валидный JSON, с этими ключами:\n"
            "- Краткая информация\n"
            "- Диагностика\n"
            "- Лечение\n"
            "- Реабилитация\n"
            "- Профилактика\n"
            "- Организация ухода\n"
            "- Факторы прогноза\n\n"
            "Верни строго валидный JSON, с двойными кавычками и строковыми значениями.\n\n"
            f"Текст:\n{text}"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
            )

        generated = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        try:
            data: Dict[str, str] = json.loads(generated)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse JSON from model output: {generated}"
            ) from exc

        return data


class GuidelineParserStub:
    """
    Semantic chunker for clinical guideline text.

    Strategy:
    1. Split the text on paragraph breaks (two or more consecutive newlines).
    2. Group small paragraphs together into chunks of up to ``chunk_size`` tokens,
       accumulating text until adding the next paragraph would exceed the limit.
    3. Sub-divide paragraphs that individually exceed ``chunk_size`` using
       overlapping token windows (identical to the previous token-sliding approach).

    This preserves meaningful text units (paragraphs) wherever possible and
    falls back to raw token slicing only for oversized blocks.

    The tokenizer is used only for token counting and decoding; no model
    inference is performed.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """
        Args:
            chunk_size: Maximum number of tokens per output chunk.
            overlap: Number of tokens to overlap when sub-dividing oversized blocks.
            model_name: HuggingFace model name used to load the tokenizer.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def parse(self, text: str) -> List[str]:
        """
        Split guideline text into semantically coherent chunks.

        Args:
            text: Raw guideline text (may contain ``\\n\\n`` paragraph breaks).

        Returns:
            List of non-empty text chunk strings.
        """
        paragraphs = self._split_paragraphs(text)
        chunks: List[str] = []

        current_parts: List[str] = []
        current_tokens: int = 0

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para, add_special_tokens=False))

            if para_tokens > self.chunk_size:
                # Flush the current accumulation before handling the large block.
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                chunks.extend(self._token_chunks(para))

            elif current_tokens + para_tokens > self.chunk_size:
                # Current accumulation is full — flush and start a new group.
                chunks.append("\n\n".join(current_parts))
                current_parts = [para]
                current_tokens = para_tokens

            else:
                current_parts.append(para)
                current_tokens += para_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return [c for c in chunks if c.strip()]

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split text on two or more consecutive newlines."""
        parts = re.split(r"\n{2,}", text)
        return [p.strip() for p in parts if p.strip()]

    def _token_chunks(self, text: str) -> List[str]:
        """
        Split a single block into overlapping token windows.

        Used as a fallback for paragraphs that individually exceed chunk_size.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        stride = max(1, self.chunk_size - self.overlap)
        chunks: List[str] = []

        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = self.tokenizer.decode(
                tokens[start:end], skip_special_tokens=True
            )
            chunks.append(chunk_text)
            start += stride

        return chunks
