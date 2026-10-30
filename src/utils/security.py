"""Security utilities: PII sanitization and prompt-injection guardrails for NEXUS-SLM."""

import re
from dataclasses import dataclass
from typing import Optional, Tuple


# Regex patterns for sensitive PII identification
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
PHONE_REGEX = re.compile(r"(?:\+91[\-\s]?)?[6-9]\d{9}\b|\b(?:\d{3}[-\s]?\d{3}[-\s]?\d{4})\b")
CARD_REGEX = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b")
UPI_REGEX = re.compile(r"\b[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}\b")

# Prompt injection patterns (adversarial jailbreak attempts & delimiter injections)
INJECTION_PATTERNS = [
    re.compile(r"(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|directives?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\s+[a-z0-9_\-\s]+agent", re.IGNORECASE),
    re.compile(r"(?:developer\s+mode|dan\s+mode|jailbreak)", re.IGNORECASE),
    re.compile(r"<\s*\|\s*im_start\s*\|", re.IGNORECASE),
    re.compile(r"<\s*\|\s*im_end\s*\|", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"reveal\s+(?:the\s+)?(?:system\s+)?prompt", re.IGNORECASE),
    re.compile(r"print\s+(?:the\s+)?(?:system\s+)?prompt", re.IGNORECASE),
    re.compile(r"bypass\s+(?:all\s+)?safety\s+filters?", re.IGNORECASE),
]


@dataclass(frozen=True)
class PromptInjectionAnalysis:
    """Outcome of adversarial prompt-injection inspection."""
    is_injection_attempt: bool
    matched_pattern: Optional[str] = None
    risk_level: str = "none"  # "none", "medium", "critical"


def sanitize_pii(text: str) -> str:
    """Redacts personally identifiable information (emails, phone numbers, cards, UPI IDs).

    Preserves sentence syntax and semantic intent for customer support classification.
    """
    if not text:
        return ""

    sanitized = CARD_REGEX.sub("[CARD_REDACTED]", text)
    sanitized = EMAIL_REGEX.sub("[EMAIL_REDACTED]", sanitized)
    sanitized = UPI_REGEX.sub("[UPI_REDACTED]", sanitized)
    sanitized = PHONE_REGEX.sub("[PHONE_REDACTED]", sanitized)
    return sanitized


def inspect_prompt_injection(text: str) -> PromptInjectionAnalysis:
    """Detects adversarial prompt-injection, delimiter escape, or jailbreak patterns.

    Returns risk assessment without mutating user input.
    """
    if not text:
        return PromptInjectionAnalysis(is_injection_attempt=False)

    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            matched_str = match.group(0)
            # High-risk control token injection
            if "im_start" in matched_str.lower() or "im_end" in matched_str.lower():
                risk = "critical"
            else:
                risk = "high"
            return PromptInjectionAnalysis(
                is_injection_attempt=True,
                matched_pattern=matched_str,
                risk_level=risk,
            )

    return PromptInjectionAnalysis(is_injection_attempt=False, risk_level="none")
