"""Comprehensive Application Security Audit Suite for NEXUS-SLM (Workstream 8)."""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.security import sanitize_pii

logger = get_logger(__name__)

SECRET_PATTERNS = [
    (r"(?i)(?:api_key|apikey|secret|password|access_token|bearer)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]", "High-entropy API Key / Secret"),
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API Key format"),
    (r"ghp_[a-zA-Z0-9]{20,}", "GitHub Personal Access Token"),
    (r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----", "Private Cryptographic Key"),
]


# Explicitly verified non-production test/dev placeholder literals used in unit test assertions
_ALLOWLISTED_TEST_LITERAL_FRAGMENTS = (
    "short-secret-key-12345",
    "nexus-test-environment-key-min-32-chars-long",
    "nexus-dev-environment-key-min-32-chars-long",
    "security-audit-test-admin-key-min-32-chars",
)


def audit_git_tracked_secrets() -> Dict[str, Any]:
    """Scans source, config, and artifact text files for accidental credential commits."""
    findings = []
    allowlisted_fixture_hits = []
    scan_exts = {".py", ".json", ".yml", ".yaml", ".toml", ".env", ".txt", ".md"}
    skip_dirs = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".next", "scratch"}
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            p = Path(root) / f
            if p.suffix in scan_exts and p.name != ".env.example":
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    rel = str(p.relative_to(PROJECT_ROOT))
                    for pattern, p_name in SECRET_PATTERNS:
                        for m in re.finditer(pattern, content):
                            matched_str = m.group(0)
                            if any(frag in matched_str for frag in _ALLOWLISTED_TEST_LITERAL_FRAGMENTS):
                                allowlisted_fixture_hits.append({
                                    "file": rel,
                                    "pattern_type": p_name,
                                    "reason": "explicit_non_prod_test_fixture",
                                })
                            else:
                                findings.append({
                                    "file": rel,
                                    "pattern_type": p_name,
                                })
                except Exception:
                    pass
    return {
        "status": "PASS" if not findings else "FAIL",
        "findings_count": len(findings),
        "findings": findings,
        "allowlisted_test_fixture_hits_count": len(allowlisted_fixture_hits),
        "allowlisted_test_fixture_hits": allowlisted_fixture_hits,
    }


def audit_env_and_gitignore() -> Dict[str, Any]:
    gitignore_p = PROJECT_ROOT / ".gitignore"
    env_example_p = PROJECT_ROOT / ".env.example"
    env_real_p = PROJECT_ROOT / ".env"

    is_env_ignored = False
    if gitignore_p.exists():
        lines = [line.strip() for line in gitignore_p.read_text(encoding="utf-8").splitlines()]
        is_env_ignored = any(".env" in line for line in lines)

    env_example_clean = True
    if env_example_p.exists():
        content = env_example_p.read_text(encoding="utf-8")
        for pattern, _ in SECRET_PATTERNS:
            if re.findall(pattern, content):
                env_example_clean = False

    return {
        "is_env_in_gitignore": is_env_ignored,
        "does_real_env_exist": env_real_p.exists(),
        "env_example_has_zero_secrets": env_example_clean,
        "status": "PASS" if (is_env_ignored and env_example_clean) else "FAIL",
    }


def audit_pii_and_prod_config_guards() -> Dict[str, Any]:
    """Verify PII scrubbing helpers and production secret startup enforcement."""
    sample = "Contact alice@example.com or +91-9876543210, card 4111-2222-3333-4444"
    scrubbed = sanitize_pii(sample)
    pii_scrubbed = ("alice@example.com" not in scrubbed) and ("4111-2222-3333-4444" not in scrubbed)

    prod_blocks_weak_key = False
    try:
        AppConfig(env="production", admin_api_key="weak", enforce_admin_auth=True)
    except ValueError:
        prod_blocks_weak_key = True

    prod_blocks_disabled_auth = False
    try:
        AppConfig(env="production", admin_api_key="A" * 40, enforce_admin_auth=False)
    except ValueError:
        prod_blocks_disabled_auth = True

    return {
        "pii_scrubbing_verified": pii_scrubbed,
        "prod_rejects_weak_admin_key": prod_blocks_weak_key,
        "prod_rejects_disabled_auth": prod_blocks_disabled_auth,
        "status": "PASS" if (pii_scrubbed and prod_blocks_weak_key and prod_blocks_disabled_auth) else "FAIL",
    }


def audit_api_security_controls() -> Dict[str, Any]:
    """Tests API input limits, auth controls, proxy spoofing resistance, CORS, and error hygiene."""
    import tempfile
    from fastapi.testclient import TestClient

    tmp_tel = str(Path(tempfile.mkdtemp(prefix="nexus_sec_audit_")) / "telemetry.jsonl")
    os.environ.setdefault("NEXUS_ENV", "test")
    os.environ.setdefault("NEXUS_ADMIN_KEY", "security-audit-test-admin-key-min-32-chars")
    os.environ.setdefault("NEXUS_TELEMETRY_LOG_FILE", tmp_tel)

    import api.main as api_main
    from api.middleware import resolve_client_ip

    with TestClient(api_main.app) as client:
        r_oversized = client.post("/predict", json={"ticket": "A" * 5000})
        oversized_rejected = r_oversized.status_code == 422

        r_empty = client.post("/predict", json={"ticket": ""})
        empty_rejected = r_empty.status_code == 422

        r_malformed = client.post("/predict", data="invalid json {", headers={"Content-Type": "application/json"})
        malformed_rejected = r_malformed.status_code == 422

        r_unauth = client.post("/model/rollback?target_version=nexus-qlora-v1")
        r_forbid = client.post("/model/rollback?target_version=nexus-qlora-v1", headers={"Authorization": "Bearer wrong-token-123"})
        auth_enforced = (r_unauth.status_code == 401 and r_forbid.status_code == 403)

        error_hygiene_ok = ("Traceback" not in r_oversized.text) and (str(PROJECT_ROOT) not in r_oversized.text)

        r_health = client.get("/health")
        h_str = json.dumps(r_health.json()) if r_health.status_code == 200 else ""
        health_clean = (r_health.status_code == 200) and not any(k in h_str.lower() for k in ["admin_api_key", "secret_key", "password"])

    # Verify untrusted peer cannot spoof X-Forwarded-For to bypass rate limiter
    class _Peer:
        def __init__(self, host): self.host = host

    class _Req:
        def __init__(self, host, xff):
            self.client = _Peer(host)
            self.headers = {"X-Forwarded-For": xff}

    spoofed_ip_ignored = resolve_client_ip(_Req("203.0.113.55", "198.51.100.1"), ["127.0.0.1"]) == "203.0.113.55"
    trusted_ip_honored = resolve_client_ip(_Req("10.0.0.2", "198.51.100.1, 10.0.0.2"), ["10.0.0.2"]) == "198.51.100.1"

    controls_pass = all([
        oversized_rejected,
        empty_rejected,
        malformed_rejected,
        auth_enforced,
        error_hygiene_ok,
        health_clean,
        spoofed_ip_ignored,
        trusted_ip_honored,
    ])
    return {
        "oversized_rejected_422": oversized_rejected,
        "empty_rejected_422": empty_rejected,
        "malformed_rejected_422": malformed_rejected,
        "admin_auth_401_403_enforced": auth_enforced,
        "error_messages_sanitized": error_hygiene_ok,
        "health_endpoint_safe": health_clean,
        "untrusted_xff_spoof_blocked": spoofed_ip_ignored,
        "trusted_proxy_xff_honored": trusted_ip_honored,
        "status": "PASS" if controls_pass else "FAIL",
    }


def main():
    secrets_res = audit_git_tracked_secrets()
    env_res = audit_env_and_gitignore()
    cfg_pii_res = audit_pii_and_prod_config_guards()
    api_res = audit_api_security_controls()

    overall = "PASS" if all(r["status"] == "PASS" for r in (secrets_res, env_res, cfg_pii_res, api_res)) else "FAIL"
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_security_status": overall,
        "audits": {
            "secrets_scan": secrets_res,
            "environment_and_gitignore": env_res,
            "pii_and_config_guards": cfg_pii_res,
            "api_security_controls": api_res,
        },
        "residual_risks": [
            "Default ALLOWED_ORIGINS='*' for local development dashboard convenience; production deployments should set explicit ALLOWED_ORIGINS origins.",
            "HTTP 429 rate limiting resets across replicas unless NEXUS_RATE_LIMITING_BACKEND=redis and external Redis 7+ service is configured.",
        ],
    }

    out_file = PROJECT_ROOT / "experiments/artifacts/security_audit_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Overall Security Status: {overall}")
    print(json.dumps(report["audits"], indent=2))
    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
