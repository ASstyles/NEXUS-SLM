"""Data validator and text normalizer for NEXUS-SLM."""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from pydantic import ValidationError
from src.data.schemas import TicketRecord, ValidationResult
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES


class DataValidator:
    """Validates raw records, normalizes text, and audits schema adherence."""

    HINGLISH_KEYWORDS = {
        "mera", "meri", "mere", "hai", "hain", "nahi", "nhi", "kya", "karo",
        "kijiye", "kar", "gaya", "gayi", "hoga", "aaya", "aayi", "tha", "thi",
        "ho", "raha", "rahi", "kaise", "kab", "kaha", "kyun", "bhai", "sir",
        "plz", "paise", "kat", "order", "bataye", "chahiye", "lekin", "magar"
    }

    def __init__(self, categories: Optional[List[str]] = None,
                 priorities: Optional[List[str]] = None,
                 languages: Optional[List[str]] = None):
        self.categories = set(categories or CATEGORIES)
        self.priorities = set(priorities or PRIORITIES)
        self.languages = set(languages or LANGUAGES)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Cleans and standardizes raw ticket text."""
        if not text:
            return ""
        # Unicode normalization (NFC ensures consistent Devanagari representation)
        text = unicodedata.normalize("NFC", text)
        # Replace non-breaking spaces and irregular whitespace
        text = re.sub(r"[\r\t\f\v]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def validate_language_heuristic(self, text: str, declared_language: str) -> Tuple[bool, Optional[str]]:
        """Heuristic check to flag severe language mismatches."""
        has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))

        if declared_language == "hindi":
            if not has_devanagari:
                return False, "Declared 'hindi' but text contains no Devanagari characters"
        elif declared_language == "english":
            if has_devanagari:
                return False, "Declared 'english' but text contains Devanagari characters"
            # Check if text is full of Hinglish keywords
            words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
            overlap = words.intersection(self.HINGLISH_KEYWORDS)
            if len(overlap) >= 3:
                return False, f"Declared 'english' but text contains prominent Hinglish words: {list(overlap)[:3]}"
        elif declared_language == "hinglish":
            if has_devanagari:
                # Some mixed Devanagari-Roman can exist in code-switched, but standard Hinglish is Roman Hindi
                pass
            words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
            overlap = words.intersection(self.HINGLISH_KEYWORDS)
            if len(overlap) == 0 and len(words) > 3:
                return True, "Hinglish sample has no common marker words (flagged as subtle Hinglish)"

        return True, None

    def validate_record(self, record_data: Dict[str, Any]) -> Tuple[Optional[TicketRecord], ValidationResult]:
        """Validates a single dictionary record against schema and taxonomy."""
        errors: List[str] = []
        warnings: List[str] = []
        rec_id = str(record_data.get("id", "unknown"))

        # 1. Normalize text if present
        if "text" in record_data and isinstance(record_data["text"], str):
            record_data["text"] = self.normalize_text(record_data["text"])
        else:
            errors.append("Field 'text' is missing or not a string")

        # 2. Check Taxonomy constraints
        cat = record_data.get("category")
        if cat not in self.categories:
            errors.append(f"Invalid category '{cat}'. Expected one of: {sorted(list(self.categories))}")

        prio = record_data.get("priority")
        if prio not in self.priorities:
            errors.append(f"Invalid priority '{prio}'. Expected one of: {sorted(list(self.priorities))}")

        lang = record_data.get("language")
        if lang not in self.languages:
            errors.append(f"Invalid language '{lang}'. Expected one of: {sorted(list(self.languages))}")
        elif "text" in record_data and record_data["text"]:
            lang_ok, lang_warn = self.validate_language_heuristic(record_data["text"], lang)
            if not lang_ok:
                errors.append(f"Language mismatch: {lang_warn}")
            elif lang_warn:
                warnings.append(lang_warn)

        if "requires_human" not in record_data or not isinstance(record_data["requires_human"], bool):
            errors.append("Field 'requires_human' must be a boolean (True/False)")

        # 3. Pydantic Model Validation
        ticket: Optional[TicketRecord] = None
        if not errors:
            try:
                ticket = TicketRecord(**record_data)
            except ValidationError as ve:
                for err in ve.errors():
                    errors.append(f"{err['loc']}: {err['msg']}")

        result = ValidationResult(
            valid=(len(errors) == 0),
            record_id=rec_id,
            errors=errors,
            warnings=warnings,
        )
        return ticket, result
