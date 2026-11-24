"""Configuration loading and validation utility."""

import os
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

# Known insecure, placeholder, and default keys that must never be permitted in production
KNOWN_INSECURE_KEYS = {
    "nexus-admin-secret-key-change-in-prod",
    "admin",
    "secret",
    "password",
    "changeme",
    "12345678",
    "admin123",
    "default",
    "nexus",
    "root",
}


def generate_secure_admin_key() -> str:
    """Generates a cryptographically strong 256-bit URL-safe token using secrets module."""
    return secrets.token_urlsafe(32)


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
    host: str = field(
        default_factory=lambda: os.getenv("NEXUS_HOST", os.getenv("API_HOST", "0.0.0.0"))
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("NEXUS_PORT", os.getenv("API_PORT", "8000")))
    )
    allowed_origins: List[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
        ]
    )
    trusted_proxies: List[str] = field(
        default_factory=lambda: [
            p.strip()
            for p in os.getenv("NEXUS_TRUSTED_PROXIES", "127.0.0.1,::1,localhost").split(",")
            if p.strip()
        ]
    )

    # Security & Administration
    admin_api_key: str = field(default_factory=lambda: os.getenv("NEXUS_ADMIN_KEY", ""))
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

    # Distributed Rate Limiting & Persistence
    rate_limiting_backend: str = field(
        default_factory=lambda: os.getenv("NEXUS_RATE_LIMITING_BACKEND", "memory").lower()
    )
    redis_url: Optional[str] = field(
        default_factory=lambda: os.getenv("REDIS_URL") or os.getenv("NEXUS_REDIS_URL")
    )

    def __post_init__(self):
        # In test or development environments, provide a default isolated key if unspecified
        if not self.admin_api_key:
            if self.is_test():
                self.admin_api_key = "nexus-test-environment-key-min-32-chars-long"
            elif self.env in ("development", "dev", "local"):
                self.admin_api_key = "nexus-dev-environment-key-min-32-chars-long"

        self.validate()

    def validate(self) -> None:
        """Enforces strict production security constraints."""
        if self.is_production():
            if not self.enforce_admin_auth:
                raise ValueError(
                    "Security configuration error: Administrator authentication cannot be disabled "
                    "in production mode (NEXUS_ENFORCE_ADMIN_AUTH must be true)."
                )
            if not self.admin_api_key or not self.admin_api_key.strip():
                raise ValueError(
                    "Security configuration error: Mandatory administrative secret (NEXUS_ADMIN_KEY) "
                    "is missing in production. Generate a strong random secret (min 32 characters) "
                    "using src.utils.config.generate_secure_admin_key()."
                )
            key_clean = self.admin_api_key.strip().lower()
            if key_clean in KNOWN_INSECURE_KEYS or "change-in-prod" in key_clean:
                raise ValueError(
                    "Security configuration error: NEXUS_ADMIN_KEY matches a known insecure or "
                    "placeholder default secret. Set a unique, securely generated random secret."
                )
            if len(self.admin_api_key.strip()) < 32:
                raise ValueError(
                    f"Security configuration error: NEXUS_ADMIN_KEY is too short ({len(self.admin_api_key.strip())} chars). "
                    "Production requires >= 32 characters of entropy."
                )

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
