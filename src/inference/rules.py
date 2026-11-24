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
    r"\b(?:unauthorized|unrecognised|unrecognized|unauthorised)\s+(?:charge|charges|transaction|transactions|debit|debits|payment|payments|login|access)\b|"
    r"\bwithout\s+(?:my\s+)?(?:authorisation|authorization|consent|permission|otp)\b|"
    r"\b(?:card|account|details|money|otp)\s+(?:stolen|compromised|leaked|hacked)\b|"
    r"\b(?:hacked|stolen|compromised)\s+(?:my\s+)?(?:account|card|profile|login|funds|money)\b|"
    r"\bsomeone\s+(?:hacked|accessed|stole|debited)\b",
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


def determine_contextual_priority(text: str, category: str) -> str:
    """
    Assigns priority based on objective operational impact, urgency, and safety risk:
    - Critical: Active financial/identity compromise, physical safety hazards, legal notices.
    - High: Severe payment disruption (money debited), account lockout, urgent cancellations.
    - Medium: Routine service workflows, tracking, password resets, plan changes.
    - Low: Non-disputed tax invoices, store hours, general inquiries.
    """
    t_lower = text.lower()

    # 1. Critical Operational Hazards
    if category == "fraud_report":
        return "critical"
    if re.search(r"\b(sparking|smoking|fire|electric\s*shock|explosion|flames|overheated|धुआं|आग)\b", t_lower):
        return "critical"
    if re.search(
        r"\b(unauthorized|unrecognised|unrecognized|unauthorised|stolen|hacked|fake\s*transaction)\b|"
        r"\b(?:without|bina)\s+(?:my\s+)?(?:authorisation|authorization|consent|otp|permission)\b|"
        r"(?:अनधिकृत|धोखाधड़ी|चोरी|बिना\s*ओटीपी|खाता\s*फ्रीज)",
        t_lower
    ):
        if not NON_FRAUD_COMPOUND_PATTERN.search(t_lower):
            return "critical"

    # 2. High Operational Impact
    if category in ("payment_failure", "account_access", "order_cancel"):
        return "high"
    if re.search(
        r"\b(?:payment\s+fail|amount\s+debited|money\s+deducted|debited\s+without|charged\s+twice|double\s+charge|kat\s+gaya|kat\s+gaye|kat\s+gya|cut\s+ho\s+gaya|paise\s+kat|पैसे\s*कट|कट\s*गया|कट\s*गए)\b|"
        r"\b(?:locked\s+out|account\s+locked|suspended|2fa|authenticator|access\s+denied|लॉगिन\s*नहीं|ब्लॉक)\b|"
        r"\b(?:cancel|cancelling|cancellation)\s+(?:my\s+)?(?:order|item|shipment)\b|"
        r"\b(?:tampered|seal\s+broken|open\s+box|empty\s+box|seal\s+tooti)\b|"
        r"\b(?:crash|crashed|500\s+error|freezes|white\s+screen|क्रैश)\b",
        t_lower
    ):
        return "high"

    # 3. Low Operational Impact (Routine Informational & Non-disputed Invoices)
    if category in ("general_query", "billing"):
        if (
            re.search(r"\b(?:tax\s+invoice|gst|invoice\s+copy|receipt|download\s+invoice|reimbursement|बिल|इनवॉइस)\b", t_lower)
            or re.search(r"\b(?:hours|timings|working\s+hours|store\s+hours|helpline|faq|student\s+discount|voucher|hackathon|guide)\b", t_lower)
            or re.search(r"\b(?:कब\s*खुलता|हेल्पलाइन|समय|मार्गदर्शिका)\b", t_lower)
        ):
            return "low"
        if category == "billing" and not any(k in t_lower for k in ["dispute", "charged", "deducted", "wrongly", "kat gaya"]):
            return "low"
        if category == "general_query":
            return "low"

    # 4. Standard workflows
    if category == "product_complaint":
        return "high" if any(k in t_lower for k in ["damaged", "broken", "toota", "not working"]) else "medium"
    if category == "delivery_issue":
        return "high" if any(k in t_lower for k in ["tampered", "missing", "not delivered", "gayab", "nahi mila"]) else "medium"
    if category == "technical_issue":
        return "high" if any(k in t_lower for k in ["crash", "500 error", "server error", "crashing"]) else "medium"

    return "medium"


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
            prio = determine_contextual_priority(t, "fraud_report")
            return "fraud_report", prio, True

    # Precedence 3: Subscription Management & Cancellation
    if re.search(r"\b(?:cancel|stop|pause|terminate|end|renew|renewal|renewed|manage)\s+(?:my\s+)?(?:subscription|membership|plan)\b", t_lower) or \
       re.search(r"\b(?:subscription|membership)\s+(?:cancel|cancellation|renewal|auto-renew|charge|fee)\b", t_lower) or \
       re.search(r"\b(?:auto-renew|auto-renewal|सब्सक्रिप्शन)\b", t_lower):
        prio = determine_contextual_priority(t, "subscription")
        return "subscription", prio, False

    # Precedence 4: Order Cancellation
    if re.search(r"\b(?:cancel|cancelling|canceling|cancellation|रद्द|निरस्त)\s+(?:my\s+)?(?:order|item|package|parcel|shipment|booking)\b", t_lower) or \
       re.search(r"\border\s+(?:cancel|cancellation|canceling)\b", t_lower) or \
       re.search(r"\b(?:cancel\s+karo|cancel\s+karna\s+hai)\b", t_lower):
        prio = determine_contextual_priority(t, "order_cancel")
        return "order_cancel", prio, False

    # Precedence 5: Account Closure & Privacy / GDPR Erasure
    if re.search(r"\b(?:delete|close|erase|remove)\s+(?:my\s+)?(?:account|profile|data|membership)\b", t_lower) or \
       re.search(r"\b(?:account\s+deletion|delete\s+my\s+account|gdpr|right\s+to\s+be\s+forgotten|अकाउंट\s*डिलीट)\b", t_lower):
        prio = determine_contextual_priority(t, "account_closure")
        return "account_closure", prio, True

    # Precedence 6: Password Reset
    if re.search(r"\b(?:password|passcode|पासवर्ड)\b", t_lower) and \
       re.search(r"\b(?:forgot|reset|change|update|new\s+password|भूला|बदलना)\b", t_lower):
        prio = determine_contextual_priority(t, "password_reset")
        return "password_reset", prio, False

    # Precedence 7: Account Access & Login Issues
    if re.search(r"\b(?:login|log\s*in|sign\s*in|otp|blocked|suspended|account\s*locked|access\s*denied|लॉगिन|सस्पेंडेड)\b", t_lower):
        prio = determine_contextual_priority(t, "account_access")
        return "account_access", prio, True

    # Precedence 8: Payment Failure & Gateway Deductions
    if re.search(r"(?:payment\s+fail|payment\s+failed|amount\s+debited|money\s+deducted|kat\s+gaya|kat\s+gaye|kat\s+gya|cut\s+ho\s+gaya|cut\s+ho\s+gya|cut\s+gaya|paise\s+kat|paise\s+cut|account\s+se\s+cut|account\s+se\s+kat|upi\s+fail|gateway\s+error|पेमेंट|पैसे\s*कट|कट\s*गया|कट\s*गए|कट\s*गई)", t_lower) or \
       re.search(r"(?:debited\s+without|money\s+cut|amount\s+cut|payment\s+कट)", t_lower) or \
       ("payment" in t_lower and ("कट" in t_lower or "cut" in t_lower or "fail" in t_lower)):
        prio = determine_contextual_priority(t, "payment_failure")
        return "payment_failure", prio, True

    # Precedence 9: Refund Request
    if re.search(r"\b(?:refund|money\s+back|return\s+money|wapas\s+chahiye|रिफंड|पैसे\s*वापस)\b", t_lower):
        prio = determine_contextual_priority(t, "refund_request")
        return "refund_request", prio, False

    # Precedence 10: Order Tracking
    if re.search(r"\b(?:track|tracking|where\s+is\s+my|status\s+of\s+my|parcel\s+location|पार्सल|कहाँ)\b", t_lower):
        prio = determine_contextual_priority(t, "order_tracking")
        return "order_tracking", prio, False

    # Precedence 11: Delivery Issues
    if re.search(r"\b(?:delivery|courier|delayed\s+delivery|not\s+delivered|wrong\s+address|package\s+missing|package\s+damaged|डिलीवरी)\b", t_lower):
        prio = determine_contextual_priority(t, "delivery_issue")
        return "delivery_issue", prio, True

    # Precedence 12: Billing & Invoice Inquiries
    if re.search(r"\b(?:bill|billing|invoice|receipt|tax\s+invoice|bill\s+amount|बिल|इनवॉइस)\b", t_lower):
        prio = determine_contextual_priority(t, "billing")
        return "billing", prio, False

    # Precedence 13: Technical & System Errors
    if re.search(r"\b(?:crash|crashed|crashing|bug|server\s+error|500\s+error|404\s+error|glitch|app\s+not\s+working|क्रैश|एरर)\b", t_lower):
        prio = determine_contextual_priority(t, "technical_issue")
        return "technical_issue", prio, False

    # Precedence 14: Product Complaints (Hardware, Electronics, Physical Items)
    if re.search(r"\b(?:defective|broken|damaged|faulty|not\s+working|poor\s+quality|counterfeit|घटिया|खराब|टूटा)\b", t_lower) or \
       re.search(r"\b(?:camera|chair|phone|device|screen|laptop|earphone|shoe|shirt|watch)\s+(?:has\s+a\s+problem|is\s+broken|not\s+working|is\s+defective|is\s+damaged)\b", t_lower):
        prio = determine_contextual_priority(t, "product_complaint")
        return "product_complaint", prio, False

    # Default general query
    prio = determine_contextual_priority(t, "general_query")
    return "general_query", prio, False


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
