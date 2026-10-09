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
    (e.g. OpenAI GPT-4o-mini/GPT-4o, Anthropic Claude 3.5, Google Gemini, Groq Llama-3.3, or local Ollama).
    Strict policy: If API credentials or local service are absent, returns status 'NOT RUN'.
    Never fabricates synthetic results.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # Auto-detect provider based on available environment variables if not specified
        if provider is None:
            if os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
                self.model_name = model_name or "gpt-4o-mini"
            elif os.environ.get("GEMINI_API_KEY"):
                self.provider = "gemini"
                self.model_name = model_name or "gemini-1.5-flash"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
                self.model_name = model_name or "claude-3-5-sonnet-20241022"
            elif os.environ.get("GROQ_API_KEY"):
                self.provider = "groq"
                self.model_name = model_name or "llama-3.3-70b-versatile"
            else:
                self.provider = "openai"
                self.model_name = model_name or "gpt-4o-mini"
        else:
            self.provider = provider.lower()
            self.model_name = model_name or "gpt-4o-mini"

        # Resolve API keys
        if self.provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        elif self.provider == "gemini":
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        elif self.provider == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        elif self.provider == "groq":
            self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

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

    def get_status_info(self) -> Dict[str, Any]:
        """Returns execution availability status and metadata."""
        avail = self.is_available()
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "available": avail,
            "status": "CONFIGURED" if avail else "NOT RUN (API Key Missing)",
            "reason": None if avail else f"No valid credentials found for provider '{self.provider}' (set {self.provider.upper()}_API_KEY or OLLAMA_BASE_URL).",
        }

    def predict(self, text: str) -> TriagePrediction:
        """
        Executes zero-shot triage query against the large model.
        Raises RuntimeError if provider is not configured.
        """
        if not self.is_available():
            raise RuntimeError(
                f"Large LLM benchmark is NOT RUN. "
                f"Missing API key or running service for provider '{self.provider}'. "
                f"Configure {self.provider.upper()}_API_KEY in environment or .env."
            )

        start_time = time.perf_counter()
        raw_text = ""

        prompt = PromptFormatter.format_prompt(text)

        if self.provider == "openai":
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]

        elif self.provider == "gemini":
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                },
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]

        elif self.provider == "anthropic":
            import requests
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.0,
            }
            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["content"][0]["text"]

        elif self.provider == "groq":
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            }
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]

        elif self.provider == "ollama":
            import requests
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
            raise NotImplementedError(f"Provider '{self.provider}' not supported.")

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
