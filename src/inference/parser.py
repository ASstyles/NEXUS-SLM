"""Strict JSON output parser and schema validator for NEXUS-SLM."""

import json
import re
from typing import Any, Dict, Optional, Tuple
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES


class StructuredOutputValidationError(Exception):
    """Raised when model generation fails schema or taxonomy validation."""
    pass


class OutputParser:
    """Extracts, parses, and strictly validates structured JSON from SLM generations."""

    def __init__(self):
        self.categories = set(CATEGORIES)
        self.priorities = set(PRIORITIES)
        self.languages = set(LANGUAGES)

    @staticmethod
    def extract_json_string(raw_text: str) -> Optional[str]:
        """Extracts JSON substring, stripping markdown code fences and surrounding text."""
        if not raw_text:
            return None

        # 1. Check for markdown json block
        code_fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if code_fence_match:
            return code_fence_match.group(1).strip()

        # 2. Find outermost curly braces
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return raw_text[start_idx : end_idx + 1].strip()

        return None

    def parse_and_validate(self, raw_text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Parses raw generation string and validates strict schema and taxonomy constraints.
        Returns:
            (is_valid: bool, parsed_data: Optional[Dict], error_message: Optional[str])

        Rule: Never silently fabricate missing fields.
        """
        json_str = self.extract_json_string(raw_text)
        if not json_str:
            return False, None, "No JSON object found in model output"

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as jde:
            return False, None, f"JSON syntax error: {str(jde)}"

        if not isinstance(data, dict):
            return False, None, f"Expected JSON dictionary, got {type(data).__name__}"

        # Strict Field Check: No missing fields allowed
        required_fields = ["category", "priority", "language", "requires_human"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return False, data, f"Missing required fields: {missing_fields}"

        # Taxonomy validations
        cat = str(data["category"]).strip().lower()
        if cat not in self.categories:
            return False, data, f"Invalid category '{cat}'. Must be one of: {sorted(list(self.categories))}"

        prio = str(data["priority"]).strip().lower()
        if prio not in self.priorities:
            return False, data, f"Invalid priority '{prio}'. Must be one of: {sorted(list(self.priorities))}"

        lang = str(data["language"]).strip().lower()
        if lang not in self.languages:
            return False, data, f"Invalid language '{lang}'. Must be one of: {sorted(list(self.languages))}"

        if not isinstance(data["requires_human"], bool):
            # Check string representation if boolean was stringified
            if isinstance(data["requires_human"], str) and data["requires_human"].lower() in ["true", "false"]:
                data["requires_human"] = (data["requires_human"].lower() == "true")
            else:
                return False, data, f"Field 'requires_human' must be boolean, got {type(data['requires_human']).__name__}"

        # Standardize cleaned values in dict
        data["category"] = cat
        data["priority"] = prio
        data["language"] = lang

        return True, data, None
