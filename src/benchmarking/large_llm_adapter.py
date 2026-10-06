"""Pluggable adapter for general-purpose Large LLM benchmark comparison."""

import os
import time
from typing import Any, Dict, Optional
from src.data.schemas import TriagePrediction
from src.inference.parser import OutputParser
from src.models.prompt_formatter import PromptFormatter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LargeLLMAdapter:
    """
    Pluggable client adapter for external general-purpose LLMs
    (e.g. OpenAI GPT-4o-mini/GPT-4o, Anthropic Claude 3.5, Gemini, or local Ollama).
    Strict policy: If API credentials or local service are absent, returns status 'NOT RUN'.
    Never fabricates synthetic results.
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.parser = OutputParser()

    def is_available(self) -> bool:
        """Checks if required API key or endpoint is configured."""
        if self.provider == "ollama":
            import requests
            try:
                r = requests.get(f"{self.base_url}/api/tags", timeout=2)
                return r.status_code == 200
            except Exception:
                return False
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def predict(self, text: str) -> TriagePrediction:
        """
        Executes zero-shot triage query against the large model.
        Raises RuntimeError if provider is not configured.
        """
        if not self.is_available():
            raise RuntimeError(
                f"Large LLM benchmark is NOT RUN. "
                f"Missing API key or running service for provider '{self.provider}'."
            )

        start_time = time.perf_counter()
        raw_text = ""

        if self.provider == "openai":
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            prompt = PromptFormatter.format_prompt(text)
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]

        elif self.provider == "ollama":
            import requests
            prompt = PromptFormatter.format_prompt(text)
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data.get("response", "")

        else:
            raise NotImplementedError(f"Provider '{self.provider}' not yet configured.")

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        is_valid, parsed, err = self.parser.parse_and_validate(raw_text)

        if not is_valid or parsed is None:
            return TriagePrediction(
                category="general_query",
                priority="medium",
                language="english",
                requires_human=True,
                confidence=0.0,
                latency_ms=latency_ms,
                model_version=f"large-llm-{self.model_name}",
                raw_output=raw_text,
            )

        return TriagePrediction(
            category=parsed["category"],
            priority=parsed["priority"],
            language=parsed["language"],
            requires_human=parsed["requires_human"],
            confidence=0.95,
            latency_ms=latency_ms,
            model_version=f"large-llm-{self.model_name}",
            raw_output=raw_text,
        )
