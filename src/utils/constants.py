"""NEXUS-SLM Constants, Taxonomies, and System Prompts."""

from typing import List

# 14 Standard Support Categories
CATEGORIES: List[str] = [
    "payment_failure",
    "refund_request",
    "account_access",
    "password_reset",
    "order_cancel",
    "order_tracking",
    "subscription",
    "billing",
    "technical_issue",
    "fraud_report",
    "delivery_issue",
    "account_closure",
    "product_complaint",
    "general_query",
]

# 4 Standard Priorities
PRIORITIES: List[str] = [
    "low",
    "medium",
    "high",
    "critical",
]

# 3 Supported Language Modalities
LANGUAGES: List[str] = [
    "english",
    "hindi",
    "hinglish",
]

# Standard System Prompt for NEXUS-SLM
SYSTEM_PROMPT: str = (
    "You are NEXUS-SLM, an enterprise multilingual support intelligence model. "
    "Analyze the customer ticket and classify it into the predefined taxonomy. "
    "Output ONLY a valid JSON object with keys: 'category', 'priority', 'language', 'requires_human'. "
    "Do not include any preamble, markdown code blocks, or explanations outside the JSON."
)

JSON_SCHEMA_DESCRIPTION: str = """{
  "category": "one of [payment_failure, refund_request, account_access, password_reset, order_cancel, order_tracking, subscription, billing, technical_issue, fraud_report, delivery_issue, account_closure, product_complaint, general_query]",
  "priority": "one of [low, medium, high, critical]",
  "language": "one of [english, hindi, hinglish]",
  "requires_human": true or false
}"""

# Standard Error Classification Categories
ERROR_TYPES: List[str] = [
    "similar_categories",
    "language_ambiguity",
    "hinglish_ambiguity",
    "typos",
    "rare_class",
    "long_input",
    "short_input",
    "priority_error",
    "language_error",
    "invalid_json",
    "schema_missing_fields",
    "other",
]
