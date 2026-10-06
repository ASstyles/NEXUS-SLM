"""Prompt formatting and ChatML templating for NEXUS-SLM."""

from typing import Any, Dict, List, Optional
from src.utils.constants import SYSTEM_PROMPT, CATEGORIES, PRIORITIES, LANGUAGES


class PromptFormatter:
    """Formats customer support tickets into consistent instruction prompts."""

    @staticmethod
    def format_prompt(
        text: str,
        system_prompt: Optional[str] = None,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """
        Formats input ticket into structured conversational prompt.
        Uses tokenizer.apply_chat_template if available, otherwise standard ChatML.
        """
        sys_msg = system_prompt or SYSTEM_PROMPT

        user_content = (
            f"Support Ticket:\n\"{text}\"\n\n"
            f"Allowed categories: {', '.join(CATEGORIES)}\n"
            f"Allowed priorities: {', '.join(PRIORITIES)}\n"
            f"Allowed languages: {', '.join(LANGUAGES)}\n"
            "Return valid JSON only."
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content},
        ]

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Standard fallback ChatML format (native to Qwen / modern SLMs)
        return (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @staticmethod
    def format_training_example(
        text: str,
        target_json_str: str,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """Formats full training sample containing prompt + expected assistant output."""
        prompt = PromptFormatter.format_prompt(text, tokenizer=tokenizer)
        return f"{prompt}{target_json_str}<|im_end|>"
