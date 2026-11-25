"""Deterministic, safety-aware human escalation policy for NEXUS-SLM triage."""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils.constants import CATEGORIES, LANGUAGES, PRIORITIES
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Devanagari + ASCII safe word-boundary anchors (handles Unicode combining vowel marks like धुआं / डिलीवरी)
_WB_L = r"(?<![\w\u0900-\u097F])"
_WB_R = r"(?![\w\u0900-\u097F])"

PHYSICAL_SAFETY_PATTERN = re.compile(
    rf"{_WB_L}(?:sparking|sparks|smoking|smoke|fire|electric\s*shock|explosion|flames|overheated|धुआं|आग|चिंगारी|जलने|विस्फोट|करंट){_WB_R}",
    re.IGNORECASE,
)

# Keywords that warrant mandatory deterministic human review
FRAUD_KEYWORDS = [
    "fraud", "unauthorized", "unauthoris", "unauthorised", "stolen", "hacked", "hack", "hacker",
    "phishing", "scam", "compromised", "fake transaction", "धोखाधड़ी", "हैक", "चोरी",
    "skimming", "kisi aur ne", "unrecognized charge", "unrecognized charges",
    "unauthorized charge", "unauthorized charges", "card debited without otp",
    "fraudulent", "otp nahi aaya", "without consent", "without my consent",
    "without authorisation", "without authorization",
    "someone placed", "bina otp", "अनधिकृत", "बिना ओटीपी"
]

FINANCIAL_DISPUTE_KEYWORDS = [
    "kat gaye", "kat gaya", "cut ho gaya", "cut gaya", "cut ho gya", "cut ho gyi",
    "paise cut", "amount cut", "account se cut", "kat gya", "kat gye",
    "charged twice", "double bill", "double charge", "money twice", "double debit",
    "debited twice", "deducted from", "money deducted", "paise kat", "paisa kat",
    "wrongly debited", "gateway timeout", "amount debited", "money got deducted",
    "cvv", "atm debit", "पैसे कट", "कट गया", "पैसे कटने", "पैसे दो बार कट", "kat gaye par", "kat gaya par",
    "billed twice", "wrongly billed", "incorrectly billed",
    "discrepancy in billing", "tax-exempt"
]

ACCOUNT_SECURITY_KEYWORDS = [
    "account locked", "account suspended", "security reasons", "suspicious activity",
    "locked for security", "unauthorized login", "password attempts", "account blocked",
    "otp sms", "otp receive nahi", "otp nahi aa", "sms receive nahi",
    "authenticator", "backup codes", "mfa", "locked profile",
    "accessed my account", "someone accessed", "unauthorized access", "account accessed",
    "kisi ne login kiya", "suspicious login", "account compromised", "unrecognized device",
    "अकाउंट ब्लॉक", "ओटीपी फोन पर आ ही नहीं"
]

DELIVERY_INTEGRITY_KEYWORDS = [
    "empty box", "counterfeit", "fake branded", "fake product", "marked delivered but",
    "marked packet delivered", "never received", "parcel missing", "stolen package",
    "package tampered", "tampered", "seal broken", "seal tooti", "seal toota", "open box",
    "refurbished", "missing item", "gayab", "खाली डिब्बा", "सील टूटी"
]

LEGAL_CLOSURE_KEYWORDS = [
    "delete my account", "account permanently delete", "terminate my user profile",
    "gdpr", "erase my data", "erase all personal data", "right to be forgotten",
    "legal action", "lawyer", "consumer court", "पुलिस", "court",
    "attorney", "lawsuit", "complaint to authority", "police report",
    "advocate", "legal notice", "breach of contract", "formal legal",
    "स्थायी रूप से बंद", "निजी जानकारी मिटा"
]

AMBIGUOUS_INDICATORS = [
    "not sure if", "either or", "confused whether", "or maybe", "not clear if",
    "not sure whether", "two issues", "multiple problems", "either refund",
    "both issues", "cancel karu ya refund", "dono option"
]

PROMPT_INJECTION_KEYWORDS = [
    "system instruction", "system override", "ignore all previous", "ignore previous instructions",
    "ignore the instructions", "system prompt", "</user", "<system", "<instruction>",
    "admin override", "jailbreak", "override:"
]

# Unicode range covering Devanagari script (Hindi)
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF]")

# Non-fraud compounds containing substrings like 'hack'
NON_FRAUD_COMPOUND_REGEX = re.compile(
    r"\b(hackathon|hack-a-thon|hackathons|hack-a-thons|hacker\s*rank|life\s*hack|growth\s*hack)\b",
    re.IGNORECASE,
)


class HumanEscalationPolicy:
    """
    Enforces deterministic safety rules for human escalation routing.
    Protects against model hallucination, overconfidence, and safety-critical oversights,
    while minimizing unnecessary human queue overload.
    """

    CONFIDENCE_THRESHOLD = 0.85

    @classmethod
    def evaluate_escalation(
        cls,
        text: str,
        category: str,
        priority: str,
        language: str,
        raw_confidence: Optional[float] = None,
        is_parse_error: bool = False,
        inference_mode: str = "neural",
        model_requires_human: Optional[bool] = None,
        enforce_devanagari_fallback: bool = True,
    ) -> Tuple[bool, str, Optional[float], str, str]:
        """
        Determines whether the ticket requires human escalation and returns:
        (requires_human: bool, escalation_reason: str, calibrated_confidence: Optional[float],
         category: str, priority: str)
        """
        t = text.strip()
        t_lower = t.lower()
        word_count = len(t_lower.split())

        final_category = category
        final_priority = priority

        # 0. Prompt Injection / Adversarial bypass safeguard
        from src.utils.security import inspect_prompt_injection
        injection_check = inspect_prompt_injection(text)
        is_injection = injection_check.is_injection_attempt or any(k in t_lower for k in PROMPT_INJECTION_KEYWORDS)
        if is_injection:
            final_priority = "critical"
            return True, "adversarial_prompt_injection_detected", None, final_category, final_priority

        # 1. Context-aware Fraud detection safeguard
        # Disregard harmless compounds like 'hack-a-thon' or 'hackathon'
        is_non_fraud_compound = bool(NON_FRAUD_COMPOUND_REGEX.search(t_lower))
        is_fraud = False
        if not is_non_fraud_compound:
            is_fraud = (category == "fraud_report") or any(k in t_lower for k in FRAUD_KEYWORDS)

        if is_fraud:
            final_category = "fraud_report"
            final_priority = "critical"
            return True, "safety_critical_fraud_detection", None, final_category, final_priority

        # 1.1 Physical safety hazard safeguard (smoking, sparking, fire) with Unicode-safe boundaries
        if PHYSICAL_SAFETY_PATTERN.search(t_lower):
            final_priority = "critical"
            return True, "physical_safety_hazard_detected", None, final_category, final_priority

        # 2. Native Devanagari Hindi / Mixed-script fallback
        has_devanagari = bool(DEVANAGARI_REGEX.search(text))
        if has_devanagari and enforce_devanagari_fallback:
            return True, "native_hindi_devanagari_unreliable_tokenization", None, final_category, final_priority

        # 3. Account closure / Legal safeguard
        is_legal = (final_category == "account_closure") or any(k in t_lower for k in LEGAL_CLOSURE_KEYWORDS)
        if is_legal:
            final_category = "account_closure"
            return True, "legal_or_account_closure_action", None, final_category, final_priority

        # 4. Parse or validation failure (for non-Devanagari inputs)
        if is_parse_error:
            return True, "parse_or_schema_validation_failure", None, final_category, final_priority

        # 5. Critical priority tickets
        if final_priority == "critical":
            return True, "critical_urgency_escalation", None, final_category, final_priority

        # 6. Short or context-sparse inputs (<= 3 words)
        if word_count <= 3:
            return True, "short_input_context_sparsity", None, final_category, final_priority

        # 7. Ambiguous or conflicting multi-intent tickets
        is_ambiguous = any(ind in t_lower for ind in AMBIGUOUS_INDICATORS)
        conflicting_intents = (
            ("refund" in t_lower and "cancel" in t_lower and "subscription" not in t_lower)
            or ("delete" in t_lower and "login" in t_lower)
            or ("bill" in t_lower and "crash" in t_lower)
        )
        if is_ambiguous or conflicting_intents:
            return True, "ambiguous_ticket_requires_human_clarification", None, final_category, final_priority

        # 8. High-impact financial dispute & unexpected double debits
        is_routine_invoice = bool(
            re.search(r"\b(?:tax\s+invoice|gst\s+invoice|monthly\s+invoice|invoice\s+copy|receipt\s+for\s+reimbursement|annual\s+statement)\b", t_lower)
        ) and not any(k in t_lower for k in ["wrongly", "unauthorized", "debited", "stolen", "double"])
        is_financial_dispute = (not is_routine_invoice) and any(k in t_lower for k in FINANCIAL_DISPUTE_KEYWORDS)
        if is_financial_dispute:
            return True, "high_impact_financial_dispute", None, final_category, final_priority

        # 9. Account security / lockout safeguard
        is_account_lockout = any(k in t_lower for k in ACCOUNT_SECURITY_KEYWORDS)
        if is_account_lockout:
            return True, "account_security_lockout", None, final_category, final_priority

        # 10. Delivery integrity / theft / counterfeit safeguard
        is_delivery_risk = any(k in t_lower for k in DELIVERY_INTEGRITY_KEYWORDS)
        if is_delivery_risk:
            return True, "delivery_theft_or_tampering", None, final_category, final_priority

        # 11. Heuristic fallback mode
        if inference_mode != "neural":
            if final_priority in ["high", "critical"] or final_category in ["payment_failure", "delivery_issue"]:
                return True, "heuristic_mode_high_impact_rule", None, final_category, final_priority
            # Unambiguous routine tickets in heuristic mode can route safely without forced fallback escalation
            return False, "automated_routing_permitted", None, final_category, final_priority

        # 12. Low / Uncertain / Unavailable Confidence Check (Neural inference)
        if raw_confidence is None:
            return True, "confidence_unavailable_conservative_escalation", None, final_category, final_priority
        if raw_confidence < cls.CONFIDENCE_THRESHOLD:
            return True, "unreliable_confidence_below_threshold", None, final_category, final_priority

        # 13. Model-requested escalation
        if model_requires_human is True:
            return True, "model_predicted_human_escalation", None, final_category, final_priority

        return False, "automated_routing_permitted", None, final_category, final_priority
