"""Token-aware, Unicode-safe contextual heuristic rules and category precedence for NEXUS-SLM."""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils.constants import CATEGORIES, LANGUAGES, PRIORITIES

# Unicode range covering Devanagari script (Hindi)
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF]")

# Explicit Hinglish Romanized Hindi stopwords (anchored with word boundaries)
HINGLISH_STOPWORDS = {
    "mera", "meri", "mere", "mera", "hai", "hain", "nahi", "nhi", "nahin",
    "kat", "gaya", "gaye", "gayi", "gya", "gyi", "gye",
    "bhai", "karo", "kijiye", "karna", "karein", "paise", "paisa", "rupaye",
    "kyun", "kyu", "kya", "kaise", "kahan", "kab",
    "mujhe", "humko", "humein", "aapka", "apka", "tumhara", "unka",
    "chahiye", "karunga", "aaya", "aayi", "aaye", "mil", "raha", "rahi", "rahe",
    "wapas", "bhejo", "dekh", "suno", "samjho", "lekin", "par", "bhi"
}

# Regex pattern matching exact Hinglish tokens
HINGLISH_TOKEN_PATTERN = re.compile(
    r"\b(" + "|".join(HINGLISH_STOPWORDS) + r")\b",
    re.IGNORECASE
)

# Harmless compound patterns containing substrings like "hack"
NON_FRAUD_COMPOUND_PATTERN = re.compile(
    r"\b(hackathon|hack-a-thon|hackathons|hack-a-thons|hacker\s*rank|life\s*hack|growth\s*hack)\b",
    re.IGNORECASE
)

# Genuine fraud and security risk patterns
FRAUD_CONTEXT_PATTERN = re.compile(
    r"\b(fraud|fraudulent|phishing|scam|skimming|fake\s*transaction)\b|"
    r"\b(?:unauthorized|unrecognised|unrecognized)\s+(?:charge|charges|transaction|transactions|debit|debits|payment|payments|login|access)\b|"
    r"\b(?:card|account|details|money|otp)\s+(?:stolen|compromised|leaked|hacked)\b|"
    r"\b(?:hacked|stolen|compromised)\s+(?:my\s+)?(?:account|card|profile|login|funds|money)\b|"
    r"\bsomeone\s+(?:hacked|accessed|stole)\b",
    re.IGNORECASE
)


def detect_language(text: str) -> str:
    """
    Accurately classifies ticket language as 'hindi', 'hinglish', or 'english'.
    Avoids brittle substring checks:
    - 'camera' is NOT Hinglish simply because of 'mera'.
    - 'chair' is NOT Hinglish simply because of 'hai'.
    """
    if not text or not text.strip():
        return "english"

    # 1. Native Devanagari Hindi check
    if DEVANAGARI_REGEX.search(text):
        return "hindi"

    # 2. Token-anchored Hinglish check
    # Require at least 1 genuine Hinglish token match at strict word boundaries
    tokens = HINGLISH_TOKEN_PATTERN.findall(text)
    if tokens:
        return "hinglish"

    # 3. Default to English
    return "english"


def classify_heuristic_category_and_priority(text: str) -> Tuple[str, str, bool]:
    """
    Context-aware category and priority classification with strict precedence hierarchy.
    Returns: (category, priority, requires_human)
    """
    if not text or not text.strip():
        return "general_query", "medium", False

    t = text.strip()
    t_lower = t.lower()

    # Precedence 1: Explicit Prompt Injection Safeguard
    from src.utils.security import inspect_prompt_injection
    inj = inspect_prompt_injection(t)
    if inj.is_injection_attempt:
        return "general_query", "critical", True

    # Precedence 2: Context-aware Fraud & Account Compromise
    if not NON_FRAUD_COMPOUND_PATTERN.search(t_lower):
        if "धोखाधड़ी" in t_lower or "चोरी" in t_lower or "unauthorized" in t_lower or "unrecognised" in t_lower or "unrecognized" in t_lower or FRAUD_CONTEXT_PATTERN.search(t_lower):
            return "fraud_report", "critical", True

    # Precedence 3: Subscription Management & Cancellation
    # Differentiates 'cancel subscription' from 'cancel order'
    if re.search(r"\b(?:cancel|stop|pause|terminate|end|renew|renewal|renewed|manage)\s+(?:my\s+)?(?:subscription|membership|plan)\b", t_lower) or \
       re.search(r"\b(?:subscription|membership)\s+(?:cancel|cancellation|renewal|auto-renew|charge|fee)\b", t_lower) or \
       re.search(r"\b(?:auto-renew|auto-renewal|सब्सक्रिप्शन)\b", t_lower):
        return "subscription", "medium", False

    # Precedence 4: Order Cancellation
    if re.search(r"\b(?:cancel|cancelling|canceling|cancellation|रद्द|निरस्त)\s+(?:my\s+)?(?:order|item|package|parcel|shipment|booking)\b", t_lower) or \
       re.search(r"\border\s+(?:cancel|cancellation|canceling)\b", t_lower) or \
       re.search(r"\b(?:cancel\s+karo|cancel\s+karna\s+hai)\b", t_lower):
        return "order_cancel", "high", False

    # Precedence 5: Account Closure & Privacy / GDPR Erasure
    if re.search(r"\b(?:delete|close|erase|remove)\s+(?:my\s+)?(?:account|profile|data|membership)\b", t_lower) or \
       re.search(r"\b(?:account\s+deletion|delete\s+my\s+account|gdpr|right\s+to\s+be\s+forgotten|अकाउंट\s*डिलीट)\b", t_lower):
        return "account_closure", "medium", True

    # Precedence 6: Password Reset
    if re.search(r"\b(?:password|passcode|पासवर्ड)\b", t_lower) and \
       re.search(r"\b(?:forgot|reset|change|update|new\s+password|भूला|बदलना)\b", t_lower):
        return "password_reset", "medium", False

    # Precedence 7: Account Access & Login Issues
    if re.search(r"\b(?:login|log\s*in|sign\s*in|otp|blocked|suspended|account\s*locked|access\s*denied|लॉगिन|सस्पेंडेड)\b", t_lower):
        return "account_access", "high", True

    # Precedence 8: Payment Failure & Gateway Deductions
    if re.search(r"(?:payment\s+fail|payment\s+failed|amount\s+debited|money\s+deducted|kat\s+gaya|kat\s+gaye|kat\s+gya|cut\s+ho\s+gaya|cut\s+ho\s+gya|cut\s+gaya|paise\s+kat|paise\s+cut|account\s+se\s+cut|account\s+se\s+kat|upi\s+fail|gateway\s+error|पेमेंट|पैसे\s*कट|कट\s*गया|कट\s*गए|कट\s*गई)", t_lower) or \
       re.search(r"(?:debited\s+without|money\s+cut|amount\s+cut|payment\s+कट)", t_lower) or \
       ("payment" in t_lower and ("कट" in t_lower or "cut" in t_lower or "fail" in t_lower)):
        return "payment_failure", "high", True

    # Precedence 9: Refund Request
    if re.search(r"\b(?:refund|money\s+back|return\s+money|wapas\s+chahiye|रिफंड|पैसे\s*वापस)\b", t_lower):
        return "refund_request", "medium", False

    # Precedence 10: Order Tracking
    if re.search(r"\b(?:track|tracking|where\s+is\s+my|status\s+of\s+my|parcel\s+location|पार्सल|कहाँ)\b", t_lower):
        return "order_tracking", "medium", False

    # Precedence 11: Delivery Issues
    if re.search(r"\b(?:delivery|courier|delayed\s+delivery|not\s+delivered|wrong\s+address|package\s+missing|package\s+damaged|डिलीवरी)\b", t_lower):
        return "delivery_issue", "high", True

    # Precedence 12: Billing & Invoice Inquiries
    if re.search(r"\b(?:bill|billing|invoice|receipt|tax\s+invoice|bill\s+amount|बिल|इनवॉइस)\b", t_lower):
        return "billing", "low", False

    # Precedence 13: Technical & System Errors
    if re.search(r"\b(?:crash|crashed|crashing|bug|server\s+error|500\s+error|404\s+error|glitch|app\s+not\s+working|क्रैश|एरर)\b", t_lower):
        return "technical_issue", "high", False

    # Precedence 14: Product Complaints (Hardware, Electronics, Physical Items)
    if re.search(r"\b(?:defective|broken|damaged|faulty|not\s+working|poor\s+quality|counterfeit|घटिया|खराब|टूटा)\b", t_lower) or \
       re.search(r"\b(?:camera|chair|phone|device|screen|laptop|earphone|shoe|shirt|watch)\s+(?:has\s+a\s+problem|is\s+broken|not\s+working|is\s+defective|is\s+damaged)\b", t_lower):
        return "product_complaint", "high", False

    # Default general query
    return "general_query", "medium", False


def get_fallback_prediction(text: str) -> Dict[str, Any]:
    """
    Deterministic rule-based fallback when neural weights are not loaded.
    Used for instant initialization, testing, and cold-start fallback.
    """
    lang = detect_language(text)
    cat, prio, req_human = classify_heuristic_category_and_priority(text)

    return {
        "category": cat,
        "priority": prio,
        "language": lang,
        "requires_human": req_human,
        "confidence": None,
    }
