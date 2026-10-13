"""Unit tests for OutputParser and strict JSON schema enforcement."""

from src.inference.parser import OutputParser


def test_parser_clean_json():
    raw = '{"category": "payment_failure", "priority": "high", "language": "hinglish", "requires_human": true}'
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is True
    assert data["category"] == "payment_failure"
    assert data["requires_human"] is True
    assert err is None


def test_parser_markdown_fenced_json():
    raw = """Here is the classified ticket:
```json
{
  "category": "refund_request",
  "priority": "medium",
  "language": "english",
  "requires_human": false
}
```
Hope this helps!"""
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is True
    assert data["category"] == "refund_request"
    assert data["requires_human"] is False


def test_parser_rejects_missing_fields():
    raw = '{"category": "billing", "priority": "low"}'  # missing language and requires_human
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is False
    assert "Missing required fields" in err


def test_parser_rejects_invalid_taxonomy():
    raw = '{"category": "invalid_cat", "priority": "high", "language": "english", "requires_human": true}'
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is False
    assert "Invalid category" in err


def test_parser_normalizes_language_aliases():
    """Verify OutputParser normalizes language aliases like roman_hindi to canonical hinglish."""
    raw = '{"category": "order_tracking", "priority": "medium", "language": "roman_hindi", "requires_human": false}'
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is True
    assert data["language"] == "hinglish"
    assert data["_raw_language"] == "roman_hindi"


def test_parser_normalizes_category_aliases():
    """Verify OutputParser deterministically maps unambiguous aliases to canonical taxonomy."""
    raw = '{"category": "refund", "priority": "medium", "language": "en", "requires_human": false}'
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is True
    assert data["category"] == "refund_request"
    assert data["language"] == "english"
    assert data["_raw_category"] == "refund"
    assert data["_raw_language"] == "en"


def test_parser_preserves_raw_unnormalized_fields():
    """Verify raw parsed values are preserved separately from canonicalized values."""
    raw = '{"category": "order_cancellation", "priority": "high", "language": "hi", "requires_human": true}'
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is True
    assert data["category"] == "order_cancel"
    assert data["_raw_category"] == "order_cancellation"
    assert data["language"] == "hindi"
    assert data["_raw_language"] == "hi"

