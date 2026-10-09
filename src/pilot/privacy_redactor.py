"""Privacy, PII redaction, and secret sanitization module for NEXUS-SLM."""

import re
from typing import Dict, Tuple

# Compiled regex patterns for sensitive customer identifiers and secrets
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
CVV_REGEX = re.compile(r"\b(cvv|cvc|security code)[:=\s]+(\d{3,4})\b", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?(\d{1,3}))?[-.\s]?(\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b")
API_KEY_REGEX = re.compile(r"\b(?:bearer|token|api[_-]?key|secret)[:=\s]+([A-Za-z0-9_\-\.]{16,})\b", re.IGNORECASE)
PASSWORD_REGEX = re.compile(r"\b(?:password|pwd|passcode)[:=\s]+([^\s,]{4,})\b", re.IGNORECASE)


class PrivacyRedactor:
    """Sanitizes ticket content by redacting PII, payment credentials, and secrets prior to inference/logging."""

    @classmethod
    def redact_sensitive_data(cls, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Replaces sensitive personal and financial identifiers with safety tokens.
        Returns (sanitized_text, redaction_counts).
        """
        redactions = {
            "credit_card": 0,
            "cvv": 0,
            "email": 0,
            "phone": 0,
            "api_key": 0,
            "password": 0,
        }

        # 1. Redact CVV / CVC
        def _cvv_sub(match):
            redactions["cvv"] += 1
            return f"{match.group(1)}: [CVV_REDACTED]"
        sanitized = CVV_REGEX.sub(_cvv_sub, text)

        # 2. Redact Credit / Debit Card Numbers
        def _card_sub(match):
            digits = re.sub(r"\D", "", match.group(0))
            if 13 <= len(digits) <= 16:
                redactions["credit_card"] += 1
                return f"[CARD_REDACTED_ENDING_{digits[-4:]}]"
            return match.group(0)
        sanitized = CREDIT_CARD_REGEX.sub(_card_sub, sanitized)

        # 3. Redact API Keys / Bearer Tokens
        def _key_sub(match):
            redactions["api_key"] += 1
            return "[SECRET_TOKEN_REDACTED]"
        sanitized = API_KEY_REGEX.sub(_key_sub, sanitized)

        # 4. Redact Passwords
        def _pwd_sub(match):
            redactions["password"] += 1
            return "password: [PASSWORD_REDACTED]"
        sanitized = PASSWORD_REGEX.sub(_pwd_sub, sanitized)

        # 5. Redact Email Addresses
        def _email_sub(match):
            redactions["email"] += 1
            return "[EMAIL_REDACTED]"
        sanitized = EMAIL_REGEX.sub(_email_sub, sanitized)

        # 6. Redact Phone Numbers
        def _phone_sub(match):
            redactions["phone"] += 1
            return "[PHONE_REDACTED]"
        sanitized = PHONE_REGEX.sub(_phone_sub, sanitized)

        return sanitized, redactions
