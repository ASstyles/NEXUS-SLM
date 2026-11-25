"""Token-aware, Unicode-safe contextual heuristic rules and category precedence for NEXUS-SLM."""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils.constants import CATEGORIES, LANGUAGES, PRIORITIES

# Unicode range covering Devanagari script (Hindi)
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF]")

_WB_L = r"(?<![\w\u0900-\u097F])"
_WB_R = r"(?![\w\u0900-\u097F])"

# Explicit Hinglish Romanized Hindi stopwords (anchored with strict word boundaries)
HINGLISH_STOPWORDS = {
    "mera", "meri", "mere", "hai", "hain", "nahi", "nhi", "nahin",
    "kat", "gaya", "gaye", "gayi", "gya", "gyi", "gye",
    "bhai", "karo", "kijiye", "karna", "karein", "krna", "kardo", "kro", "karwao", "karwana",
    "paise", "paisa", "rupaye", "kyun", "kyu", "kya", "kaise", "kahan", "kab", "kis",
    "mujhe", "humko", "humein", "aapka", "apka", "tumhara", "unka", "pichle", "dijiye",
    "chahiye", "karunga", "aaya", "aayi", "aaye", "mil", "raha", "rahi", "rahe", "rha", "rhi", "rhe",
    "hua", "hui", "hoke", "wapas", "bhejo", "dekh", "suno", "samjho", "lekin", "par", "bhi",
    "kripya", "turant", "jaldi", "dikha", "gayab", "tooti", "toota", "bhul", "bhool",
    "sthiti", "kshatigrast", "prapt", "ruka", "atka", "atki", "dabba", "awaaz",
    "dabate", "samajh", "mangoo", "rehta", "khula", "hatao", "saman"
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
    if re.search(rf"{_WB_L}(?:sparking|sparks|smoking|smoke|fire|electric\s*shock|explosion|flames|overheated|धुआं|आग|चिंगारी|जलने|विस्फोट|करंट){_WB_R}", t_lower):
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
        r"\b(?:payment\s+fail|amount\s+debited|money\s+deducted|debited\s+without|charged\s+twice|money\s+twice|double\s+(?:charge|debit)|kat\s+gaya|kat\s+gaye|kat\s+gya|cut\s+ho\s+gaya|paise\s+kat)\b|"
        r"\b(?:locked\s+out|account\s+locked|suspended|2fa|authenticator|access\s+denied|account\s+blocked)\b|"
        r"\b(?:cancel|cancelling|cancellation)\s+(?:my\s+)?(?:.*?\s+)?(?:order|ord|item|shipment|booking)\b|"
        r"\b(?:tampered|seal\s+broken|open\s+box|empty\s+box|seal\s+tooti)\b|"
        r"\b(?:crash|crashed|500\s+error|freezes|white\s+screen|app\s+hang)\b|"
        rf"{_WB_L}(?:पैसे\s*कट|कट\s*गया|कट\s*गए|दो\s*बार\s*कट|लॉगिन\s*नहीं|ब्लॉक|क्रैश|500\s*एरर|खाली\s*डिब्बा|सी[ल]?\s*टूटी){_WB_R}",
        t_lower
    ):
        return "high"

    # 3. Low Operational Impact (Routine Informational & Non-disputed Invoices)
    if category in ("general_query", "billing"):
        if (
            re.search(rf"{_WB_L}(?:tax\s+invoice|gst(?:\s+tax)?\s*invoice|gst|invoice\s+(?:pdf|copy)|receipt|download\s+invoice|reimbursement|बिल|इनवॉइस){_WB_R}", t_lower)
            or re.search(rf"{_WB_L}(?:hours|operating\s+hours|timings|working\s+hours|store\s+hours|helpline|faq|student\s+(?:discount|hackathon)|voucher|hackathon|guide|before\s+ordering|generally\s+work|समय|समय-सीमा|खुलता|खरीदने\s*से\s*पहले){_WB_R}", t_lower)
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
        return "high" if any(k in t_lower for k in ["tampered", "missing", "not delivered", "gayab", "nahi mila", "खाली डिब्बा", "सील टूटी", "kshatigrast"]) else "medium"
    if category == "technical_issue":
        return "high" if any(k in t_lower for k in ["crash", "500 error", "500 एरर", "server error", "crashing", "क्रैश", "app hang"]) else "medium"

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

    # Precedence 2.5: Benign pre-purchase FAQ / Operating hours negative controls
    if (
        re.search(r"\b(?:before\s+ordering|operating\s+hours|store\s+timings|generally\s+work|what\s+are\s+your\s+(?:standard\s+)?customer\s+support)\b", t_lower)
        or re.search(rf"{_WB_L}(?:रविवार\s*को\s*भी\s*खुला|समय\s*बताएं|खरीदने\s*से\s*पहले|समय-सीमा\s*कितने\s*दिन|service\s+center\s+sunday\s+ko\s+khula){_WB_R}", t_lower)
    ) and not any(k in t_lower for k in ["hacked", "unauthorized", "stolen", "debited", "fraud"]):
        prio = determine_contextual_priority(t, "general_query")
        return "general_query", prio, False

    # Precedence 3: Subscription Management & Cancellation
    if re.search(r"\b(?:cancel|cancelling|canceling|stop|pause|terminate|end|renew|renewal|renewed|manage)\s+(?:my\s+)?(?:.*?\s+)?(?:subscription|membership|plan)\b", t_lower) or \
       re.search(r"\b(?:subscription|membership)\s+(?:cancel|cancellation|renewal|auto[-\s]?renew|charge|fee)\b", t_lower) or \
       re.search(rf"{_WB_L}(?:auto[-\s]?renew|auto[-\s]?renewal|सब्सक्रिप्शन|सदस्यता){_WB_R}", t_lower):
        prio = determine_contextual_priority(t, "subscription")
        return "subscription", prio, False

    # Precedence 4: Order Cancellation
    if re.search(r"\b(?:cancel|cancelling|canceling|cancellation)\s+(?:my\s+)?(?:.*?\s+)?(?:order|ord|item|package|parcel|shipment|booking)\b", t_lower) or \
       re.search(r"\b(?:order|ord)\s+(?:cancel|cancellation|canceling)\b", t_lower) or \
       re.search(r"\b(?:cancel\s+(?:kar|karo|kr|kardo|karna\s+hai)|radd\s+(?:kar|karo))\b", t_lower) or \
       re.search(rf"{_WB_L}(?:ऑर्डर\s*रद्द|रद्द\s*करें|रद्द\s*कर|निरस्त\s*कर){_WB_R}", t_lower):
        prio = determine_contextual_priority(t, "order_cancel")
        return "order_cancel", prio, False

    # Precedence 5: Account Closure & Privacy / GDPR Erasure
    if re.search(r"\b(?:delete|close|erase|remove|terminate)\s+(?:my\s+)?(?:user\s+)?(?:account|profile|data|membership|personal\s+records)\b", t_lower) or \
       re.search(r"\b(?:account\s+deletion|delete\s+my\s+account|account\s+permanently\s+delete|erase\s+all\s+personal\s+data|gdpr|right\s+to\s+be\s+forgotten)\b", t_lower) or \
       re.search(rf"{_WB_L}(?:अकाउंट\s*डिलीट|खाता\s*(?:स्थायी\s*रूप\s*से\s*)?बंद|निजी\s*जानकारी\s*मिटा){_WB_R}", t_lower):
        prio = determine_contextual_priority(t, "account_closure")
        return "account_closure", prio, True

    # Precedence 6: Password Reset
    if re.search(rf"{_WB_L}(?:password|passcode|pwd|paswrd|पासवर्ड){_WB_R}", t_lower) and \
       re.search(rf"{_WB_L}(?:forgot|forget|reset|change|update|new\s+password|expir[ed]+|bhul|bhool|भूल|भूला|बदलना|रीसेट){_WB_R}", t_lower):
        prio = determine_contextual_priority(t, "password_reset")
        return "password_reset", prio, False

    # Precedence 7: Account Access & Login Issues
    if re.search(rf"{_WB_L}(?:login|log\s*in|sign\s*in|otp|2fa|two[-\s]?factor|authenticator|blocked|suspended|locked\s+profile|account\s*locked|access\s*denied|लॉगिन|सस्पेंडेड|ओटीपी|अकाउंट\s*ब्लॉक){_WB_R}", t_lower):
        prio = determine_contextual_priority(t, "account_access")
        return "account_access", prio, True

    # Precedence 8: Payment Failure & Gateway Deductions
    if re.search(rf"{_WB_L}(?:payment\s+fail|payment\s+failed|charged.*?twice|money\s+twice|double\s+debit|amount\s+debited|money\s+deducted|kat\s+gaya|kat\s+gaye|kat\s+gya|cut\s+ho\s+gaya|cut\s+ho\s+gya|cut\s+gaya|paise\s+kat|paise\s+cut|account\s+se\s+cut|account\s+se\s+kat|upi\s+fail|gateway\s+error|पेमेंट|पैसे.*?(?:कट|दो\s*बार)|कट\s*गया|कट\s*गए|कट\s*गई|order\s+fail\s+dikha){_WB_R}", t_lower) or \
       re.search(r"(?:debited\s+without|money\s+cut|amount\s+cut|payment\s+कट)", t_lower) or \
       ("payment" in t_lower and ("कट" in t_lower or "cut" in t_lower or "fail" in t_lower)):
        prio = determine_contextual_priority(t, "payment_failure")
        return "payment_failure", prio, True

    # Precedence 9: Refund Request (prioritize defective delivery/tampered over generic refund only when delivery/physical hazard dominates)
    if re.search(rf"{_WB_L}(?:refund|rfnd|money\s+back|return\s+money|wapas\s+chahiye|refund.*?credit|रिफंड|पैसे\s*वापस|रुपये\s*वापस){_WB_R}", t_lower):
        prio = determine_contextual_priority(t, "refund_request")
        return "refund_request", prio, False

    # Precedence 10: Order Tracking
    if re.search(rf"{_WB_L}(?:track|tracking|where\s+is\s+my|status\s+of\s+my|parcel\s+location|when.*?(?:reach|reaches|arrive|deliver)|stuck\s+in|ruka\s+hua|ट्रैकिंग|पार्सल.*?(?:कहाँ|नहीं\s*पहुंचा|रुका)){_WB_R}", t_lower):
        prio = determine_contextual_priority(t, "order_tracking")
        return "order_tracking", prio, False

    # Precedence 11: Delivery Issues
    if re.search(rf"{_WB_L}(?:delivery|delvry|courier|delayed\s+delivery|not\s+delivered|wrong\s+address|package\s+missing|package\s+damaged|tampered|empty\s+box|seal\s+tooti|seal\s+broken|kshatigrast|saman\s+gayab|डिलीवरी|खाली\s*डिब्बा|सील\s*टूटी|सामान\s*गायब){_WB_R}", t_lower):
        prio = determine_contextual_priority(t, "delivery_issue")
        return "delivery_issue", prio, True

    # Precedence 12: Billing & Invoice Inquiries
    if re.search(rf"{_WB_L}(?:bill|billing|invoice|receipt|tax\s+invoice|gst\s+(?:tax\s+)?invoice|bill\s+amount|बिल|इनवॉइस){_WB_R}", t_lower):
        prio = determine_contextual_priority(t, "billing")
        return "billing", prio, False

    # Precedence 13: Technical & System Errors
    if re.search(rf"{_WB_L}(?:crash|crashed|crashing|force[-\s]?closing|app\s+hang|bug|server\s+error|500\s+(?:error|एरर)|404\s+error|glitch|app\s+not\s+working|क्रैश|एरर|खराबी|काम\s*नहीं\s*कर\s*रही){_WB_R}", t_lower):
        prio = determine_contextual_priority(t, "technical_issue")
        return "technical_issue", prio, False

    # Precedence 14: Product Complaints (Hardware, Electronics, Physical Items)
    if re.search(rf"{_WB_L}(?:defective|broken|damaged|faulty|not\s+working|poor\s+quality|counterfeit|wobble|wobbles|leaks|slow\s+awaaz|awaaz|घटिया|खराब|टूटा|टूटी|अटक\s*गया|धुआं|चिंगारी){_WB_R}", t_lower) or \
       re.search(r"\b(?:camera|chair|phone|device|screen|laptop|earphone|headphones|mixer|blender|heater|charger|shoe|shirt|watch)\s+(?:has\s+a\s+problem|is\s+broken|not\s+working|is\s+defective|is\s+damaged)\b", t_lower):
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
