from __future__ import annotations

import json
from typing import Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch


class GuidelineParser:
    """
    Parse raw OCR guideline text into structured sections using a local LLM.

    The model is expected to return strictly valid JSON containing
    the requested guideline sections.
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
        - Diagnosis
        - Short information
        - Diagnostics
        - Treatment
        - Rehabilitation
        - Prevention
        - Organization of care
        - Prognosis factors

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

        # Attempt to parse JSON; raise a clear error if this fails.
        try:
            data: Dict[str, str] = json.loads(generated)
        except json.JSONDecodeError as exc:  # noqa: PERF203
            raise RuntimeError(
                f"Failed to parse JSON from model output: {generated}"
            ) from exc

        return data


class GuidelineParserStub:
    """
    Temporary stub parser for clinical guidelines.

    Splits the text into overlapping chunks by token count.
    Does not try to extract structured JSON.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """
        Args:
            chunk_size: Number of tokens per chunk.
            overlap: Number of tokens to overlap between consecutive chunks.
            model_name: Model name for tokenizer (can be lightweight).
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def parse(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Raw clinical guideline text.

        Returns:
            List of text chunks (strings).
        """
        # Encode text into tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: List[str] = []

        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            # Move start pointer with overlap
            start += self.chunk_size - self.overlap
            if start < 0:
                start = 0

        return chunks

