"""Configuration loading and validation utility."""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def _expand_env_vars(obj: Any) -> Any:
    """Recursively replaces ${VAR_NAME} or $VAR_NAME in strings with environment variables."""
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    elif isinstance(obj, str):
        pattern = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default)

        return pattern.sub(replacer, obj)
    return obj


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Loads a YAML configuration file with environment variable expansion."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p.resolve()}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return _expand_env_vars(data)


from dataclasses import dataclass, field
from typing import List


@dataclass
class AppConfig:
    """Production application configuration with environment overrides and validation."""

    # Environment
    env: str = field(default_factory=lambda: os.getenv("NEXUS_ENV", "production").lower())
    debug: bool = field(default_factory=lambda: os.getenv("NEXUS_DEBUG", "false").lower() in ("true", "1", "yes"))

    # Active Model
    active_model_version: str = field(default_factory=lambda: os.getenv("ACTIVE_MODEL_VERSION", "nexus-qlora-v1"))
    adapter_path_override: Optional[str] = field(default_factory=lambda: os.getenv("NEXUS_ADAPTER_PATH"))

    # API & Networking
    host: str = field(default_factory=lambda: os.getenv("NEXUS_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("NEXUS_PORT", "8000")))
    allowed_origins: List[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
        ]
    )

    # Security & Administration
    admin_api_key: str = field(default_factory=lambda: os.getenv("NEXUS_ADMIN_KEY", "nexus-admin-secret-key-change-in-prod"))
    enforce_admin_auth: bool = field(
        default_factory=lambda: os.getenv("NEXUS_ENFORCE_ADMIN_AUTH", "true").lower() in ("true", "1", "yes")
    )
    pii_redaction_enabled: bool = field(
        default_factory=lambda: os.getenv("NEXUS_PII_REDACTION", "true").lower() in ("true", "1", "yes")
    )
    prompt_injection_guard_enabled: bool = field(
        default_factory=lambda: os.getenv("NEXUS_PROMPT_INJECTION_GUARD", "true").lower() in ("true", "1", "yes")
    )

    # Traffic, Latency & Concurrency SLAs
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv("NEXUS_RATE_LIMIT_PER_MINUTE", "120")))
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("NEXUS_REQUEST_TIMEOUT_SECONDS", "30.0"))
    )
    concurrency_limit: int = field(default_factory=lambda: int(os.getenv("NEXUS_CONCURRENCY_LIMIT", "25")))
    max_payload_length: int = field(default_factory=lambda: int(os.getenv("NEXUS_MAX_PAYLOAD_CHARS", "4000")))

    # Telemetry & Monitoring
    telemetry_file: str = field(default_factory=lambda: os.getenv("TELEMETRY_LOG_FILE", "monitoring/telemetry.jsonl"))

    def is_production(self) -> bool:
        return self.env == "production"

    def is_test(self) -> bool:
        return self.env == "test"


_config_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Singleton getter for application configuration."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance


def reload_config() -> AppConfig:
    """Forces reloading of configuration from environment."""
    global _config_instance
    _config_instance = AppConfig()
    return _config_instance
