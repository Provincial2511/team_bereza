from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch


class LocalGenerator:
    """
    Generate clinical recommendations using a local HuggingFace transformer model.

    This class:
    - Loads a causal language model locally (no external APIs).
    - Constructs prompts explicitly from retrieved guideline sections and patient data.
    - Generates text continuations using the model.

    Prompt construction is explicit to maintain full control over the input format
    and ensure reproducibility. The prompt format combines:
    1. Retrieved clinical guideline sections (context)
    2. Patient information
    3. A clear instruction for the model
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """
        Initialize the local generator with a HuggingFace model.

        Args:
            model_name: Name or path of the HuggingFace causal language model.
            device: Device identifier, e.g. "cpu" or "cuda".
        """
        self.model_name = model_name
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        patient_text: str,
        retrieved_sections: list[str],
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a clinical recommendation based on guideline sections and patient data.

        Args:
            patient_text: Text containing patient information.
            retrieved_sections: List of relevant guideline section texts.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Generated recommendation text (prompt stripped).
        """
        retrieved_context = "\n\n".join(retrieved_sections)

        prompt = (
            "Ты клинический консультант.\n"
            "На основе следующих клинических рекомендаций:\n"
            f"{retrieved_context}\n\n"
            "И следующих данных пациента:\n"
            f"{patient_text}\n\n"
            "Сопоставь историю лечения с клиническими рекомендациями и предложи рекомендации для дальнейшего лечения.\n\n"
            "Ответ должен быть на русском языке."
        )

        inputs = self.tokenizer(
            prompt,
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

        # Decode full output
        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Strip the prompt to return only the generated continuation
        if full_output.startswith(prompt):
            generated = full_output[len(prompt) :].strip()
        else:
            # Fallback: decode only new tokens
            input_length = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][input_length:]
            generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return generated

