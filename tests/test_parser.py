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
