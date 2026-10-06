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
