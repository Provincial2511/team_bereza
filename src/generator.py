from __future__ import annotations

import json
import logging
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LocalGenerator:
    """
    Generate clinical recommendations using a local HuggingFace chat model.

    This class:
    - Loads a causal language model locally (no external APIs).
    - Formats prompts via ``tokenizer.apply_chat_template`` for proper
      instruction-following with chat models such as Qwen2-7B-Instruct.
    - Supports two generation modes: ``"doctor"`` and ``"patient"``.
    - Returns only the newly generated tokens (no prompt stripping required).
    """

    _DISCLAIMER = (
        "\n\nНе является медицинской рекомендацией! "
        "Материал создан нейросетью. Используйте в ознакомительных целях."
    )

    _SYSTEM_PROMPTS: dict[str, str] = {
        "doctor": (
            "Ты — врач-онколог. Оценивай тактику лечения пациента.\n\n"
            "🔴 ГЛАВНОЕ ПРАВИЛО — ИСТОЧНИК ДАННЫХ:\n"
            "Твой единственный источник знаний — блок «Клинические рекомендации» в запросе "
            "пользователя. Каждый фрагмент помечен тегом [КР: название]. "
            "Используй ТОЛЬКО эти данные. Всё остальное — запрещено.\n\n"
            "🔴 СТРОГО ЗАПРЕЩЕНО:\n"
            "— называть препараты, схемы, дозы, которых нет в тегах [КР: ...];\n"
            "— ссылаться на статистику, уровни доказательности, выживаемость — если они "
            "не процитированы дословно из [КР: ...];\n"
            "— упоминать организации (RUSSCO, ESMO, NCCN и др.) — если они не названы "
            "в тексте [КР: ...];\n"
            "— использовать знания из обучающих данных или личный опыт;\n"
            "— придумывать медицинские термины или слова.\n\n"
            "Если в предоставленных [КР: ...] нет данных по конкретному вопросу — напиши:\n"
            "«В предоставленных фрагментах КР информации по этому пункту нет.»\n\n"
            "⛔ ТЕМА:\n"
            "Если вопрос не относится к онкологии или лечению этого пациента — ответь только:\n"
            "«Я специализируюсь только на онкологии и клинических рекомендациях. "
            "Пожалуйста, задайте вопрос по теме лечения пациента.»\n\n"
            "🔹 СТРУКТУРА ОТВЕТА:\n"
            "1. Соответствие диагностики рекомендациям — 2–4 тезиса, каждый со ссылкой "
            "вида (КР: название).\n"
            "2. Соответствие терапии рекомендациям — 2–5 тезисов со ссылками.\n"
            "3. Отклонения или спорные моменты — если есть, конкретно и без домыслов.\n"
            "4. Что рекомендуется далее — только если это прямо следует из [КР: ...].\n\n"
            "Стиль: профессиональный, коллега–коллеге. Без пересказа истории болезни. "
            "Без повторов. Без таблиц и JSON.\n\n"
            "В конце ОБЯЗАТЕЛЬНО добавь:\n"
            "Не является медицинской рекомендацией! Материал создан нейросетью. "
            "Используйте в ознакомительных целях."
        ),
        "patient": (
            "Ты — врач-онколог, объясняющий пациенту его ситуацию.\n\n"
            "🔴 ГЛАВНОЕ ПРАВИЛО — ИСТОЧНИК ДАННЫХ:\n"
            "Твой единственный источник знаний — блок «Клинические рекомендации» в запросе. "
            "Каждый фрагмент помечен тегом [КР: название]. "
            "Используй ТОЛЬКО эти данные. Всё остальное — запрещено.\n\n"
            "🔴 СТРОГО ЗАПРЕЩЕНО:\n"
            "— называть препараты или процедуры, которых нет в тегах [КР: ...];\n"
            "— выдумывать статистику и прогнозы;\n"
            "— упоминать медицинские организации, если их нет в [КР: ...];\n"
            "— использовать знания из обучения.\n\n"
            "Если данных нет — честно скажи: «По этому вопросу в доступных мне материалах "
            "информации нет.»\n\n"
            "⛔ ТЕМА: только онкология и лечение этого пациента.\n\n"
            "🔹 КАК ОТВЕЧАТЬ:\n"
            "1. Кратко и простыми словами: диагноз, что уже сделано, текущая ситуация.\n"
            "2. Почему выбрана эта тактика — со ссылкой (КР: название).\n"
            "3. Что делать дальше — только если это следует из [КР: ...].\n"
            "4. Если что-то отличается от стандарта — мягко, без обвинений врача.\n\n"
            "Говори как человек, не как протокол. Без латыни без объяснения. "
            "Без паники. Минимум повторений.\n\n"
            "В конце ОБЯЗАТЕЛЬНО добавь:\n"
            "Не является медицинской рекомендацией! Материал создан нейросетью. "
            "Используйте в ознакомительных целях."
        ),
    }

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """
        Initialize the local generator with a HuggingFace chat model.

        Args:
            model_name: Name or path of the HuggingFace causal language model.
            device: Device identifier, e.g. ``"cpu"`` or ``"cuda"``.
        """
        self.model_name = model_name
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # On CUDA load in float16 to halve VRAM usage (~14 GB fp32 → ~7 GB fp16).
        # On CPU keep float32 (fp16 is not accelerated on most CPUs).
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_system_prompt(self, mode: str) -> str:
        """
        Return the system prompt for the given generation mode.

        Args:
            mode: ``"doctor"`` or ``"patient"``.

        Raises:
            ValueError: If *mode* is not recognized.
        """
        if mode not in self._SYSTEM_PROMPTS:
            raise ValueError(
                f"Unknown mode '{mode}'. Must be one of: "
                f"{list(self._SYSTEM_PROMPTS.keys())}"
            )
        return self._SYSTEM_PROMPTS[mode]

    def generate(
        self,
        patient_text: str,
        retrieved_sections: list[str],
        mode: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a clinical recommendation from guideline sections and patient data.

        The prompt is formatted with ``apply_chat_template`` to ensure proper
        instruction-following behaviour for chat-tuned models (e.g. Qwen2-Instruct).

        Args:
            patient_text: Text containing patient information.
            retrieved_sections: List of relevant guideline section texts.
            mode: Generation mode; ``"doctor"`` or ``"patient"``.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Generated recommendation text (newly produced tokens only).
        """
        system_prompt = self._get_system_prompt(mode)
        retrieved_context = "\n\n---\n\n".join(retrieved_sections)

        user_message = (
            f"=== Клинические рекомендации ===\n"
            f"(Используй ТОЛЬКО эти фрагменты. Каждый помечен [КР: название].)\n\n"
            f"{retrieved_context}\n\n"
            f"=== Данные пациента ===\n{patient_text}\n\n"
            f"=== НАПОМИНАНИЕ ===\n"
            f"Используй исключительно фрагменты выше. "
            f"Не называй препараты и процедуры, которых нет в [КР: ...]. "
            f"При каждом тезисе указывай источник: (КР: название)."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # apply_chat_template adds model-specific special tokens and roles.
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens.
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_length:]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._ensure_disclaimer(generated)

    def answer(
        self,
        question: str,
        patient_text: str,
        retrieved_sections: list[str],
        mode: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Answer a follow-up question using the context from a previous analysis.

        Uses the same system prompt and retrieved sections as the main analysis,
        but appends the user's specific question to the user message.

        Args:
            question: The follow-up question from the user.
            patient_text: Original patient record text.
            retrieved_sections: Guideline sections retrieved during analysis.
            mode: ``"doctor"`` or ``"patient"``.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Answer text with disclaimer guaranteed.
        """
        system_prompt = self._get_system_prompt(mode)
        retrieved_context = "\n\n---\n\n".join(retrieved_sections)

        user_message = (
            f"=== Клинические рекомендации ===\n"
            f"(Используй ТОЛЬКО эти фрагменты. Каждый помечен [КР: название].)\n\n"
            f"{retrieved_context}\n\n"
            f"=== Данные пациента ===\n{patient_text}\n\n"
            f"=== Вопрос ===\n{question}\n\n"
            f"=== НАПОМИНАНИЕ ===\n"
            f"Используй исключительно фрагменты выше. "
            f"Не называй препараты и процедуры, которых нет в [КР: ...]. "
            f"При каждом тезисе указывай источник: (КР: название)."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_length:]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._ensure_disclaimer(generated)

    def generate_structured(
        self,
        patient_text: str,
        main_analysis: str,
        max_new_tokens: int = 350,
    ) -> dict | None:
        """
        Extract structured sections from patient data and main analysis.

        Makes a second inference pass with a JSON-focused prompt.
        Returns a parsed dict or None if JSON extraction fails.

        Expected structure::

            {
              "diagnosis": "...",
              "age": "...",
              "comorbidities": "...",
              "overall_score": 75,
              "compliant": [
                {"title": "...", "category": "Диагностика|Терапия|Контроль безопасности", "text": "..."}
              ],
              "non_compliant": [
                {"title": "...", "category": "Диагностика|Терапия|Контроль безопасности", "text": "..."}
              ],
              "recommendations": [{"title": "...", "text": "..."}]
            }
        """
        system_prompt = (
            "Ты — экстрактор клинических данных. "
            "Извлеки информацию из данных пациента и анализа в формат JSON. "
            "Выведи ТОЛЬКО валидный JSON-объект — без markdown, без пояснений.\n\n"
            "Требуемая структура:\n"
            "{\n"
            '  "diagnosis": "основной диагноз",\n'
            '  "age": "возраст пациента",\n'
            '  "comorbidities": "сопутствующие заболевания или пустая строка",\n'
            '  "overall_score": <целое число 0-100>,\n'
            '  "compliant": [\n'
            '    {"title": "...", "category": "Диагностика|Терапия|Контроль безопасности", "text": "..."}\n'
            '  ],\n'
            '  "non_compliant": [\n'
            '    {"title": "...", "category": "Диагностика|Терапия|Контроль безопасности", "text": "..."}\n'
            '  ],\n'
            '  "recommendations": [\n'
            '    {"title": "...", "text": "..."}\n'
            '  ]\n'
            "}\n\n"
            "Правила:\n"
            "- diagnosis, age, comorbidities: из данных пациента\n"
            "- overall_score: % соответствия лечения рекомендациям (0-100)\n"
            "- compliant: пункты лечения, которые СООТВЕТСТВУЮТ клиническим рекомендациям (2-5 пунктов)\n"
            "- non_compliant: пункты лечения, которые НЕ соответствуют или вызывают вопросы (1-4 пункта)\n"
            "- category — одно из: Диагностика, Терапия, Контроль безопасности\n"
            "- recommendations: дополнительные рекомендации (1-3 пункта)\n"
            "- Весь текст на русском языке\n"
        )

        user_message = (
            f"=== Данные пациента ===\n{patient_text[:2000]}\n\n"
            f"=== Анализ ===\n{main_analysis[:3000]}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_length:]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Strip markdown code fences and extract the JSON object.
        cleaned = re.sub(r"```(?:json)?", "", generated).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start < 0 or end <= start:
            logger.warning("generate_structured: no JSON object found in output")
            return None
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError as exc:
            logger.warning("generate_structured: JSON parse error: %s", exc)
            return None

    def _ensure_disclaimer(self, text: str) -> str:
        """
        Append the mandatory disclaimer if the model omitted it.

        The disclaimer is required by the system prompt, but may be truncated
        when max_new_tokens is reached before the model finishes generating.
        """
        if "Не является медицинской рекомендацией" not in text:
            return text + self._DISCLAIMER
        return text
