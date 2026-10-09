# NEXUS-SLM Inference Contract & API Reference

## 1. Strict Inference API Contract

All inference requests must adhere strictly to the schema specification. Arbitrary or unvalidated text is deterministically rejected.

### 1.1 Request Specification (`POST /predict`)

```json
{
  "ticket": "customer support ticket text"
}
```

#### Validation Rules:
- **Field Name**: `ticket` (required string). Note: `text` is accepted as an alias for backward compatibility.
- **Length Constraint**: Min length 1 character (stripped), Max length 4,000 characters.
- **Content Constraint**: Must not be empty or whitespace-only.
- **Encoding**: UTF-8 compliant. Native Hindi (Devanagari), Hinglish, Roman Hindi, English, and emojis supported.
- **Rejection Status**: Requests violating schema constraints receive deterministic `HTTP 422 Unprocessable Content`.

---

### 1.2 Response Specification

```json
{
  "category": "payment_failure",
  "priority": "high",
  "language": "hinglish",
  "requires_human": false,
  "confidence": 0.7498,
  "latency_ms": 1824.5,
  "model_version": "nexus-qlora-v1",
  "inference_mode": "neural",
  "request_id": "8f31b26c-d278-4394-8255-a4eb182098d1",
  "calibrated_confidence": 0.7498,
  "escalation_reason": "automated_routing_permitted"
}
```

---

## 2. Allowed Taxonomy Enums

### 2.1 Categories (14 Classes)
```text
payment_failure     refund_request      account_access      password_reset
order_cancel        order_tracking      subscription        billing
technical_issue     fraud_report        delivery_issue      account_closure
product_complaint   general_query
```

### 2.2 Urgency Priorities (4 Classes)
```text
low       medium       high       critical
```

### 2.3 Language Modalities (3 Classes)
```text
english   hindi   hinglish
```

---

## 3. Human Escalation Policy Safeguards

The system never relies solely on raw model generation for safety-critical decisions. The `HumanEscalationPolicy` enforces deterministic routing rules:

| Trigger Condition | Safeguard Action | Reason Tag |
|---|---|---|
| **Fraud Keywords** (`fraud`, `unauthorized`, `hacked`, `scam`, etc.) | Force `category = fraud_report`, `priority = critical`, `requires_human = True` | `safety_critical_fraud_detection` |
| **Legal / Account Deletion** (`delete account`, `gdpr`, `court`, etc.) | Force `category = account_closure`, `requires_human = True` | `legal_or_account_closure_action` |
| **Critical Priority** | Force `requires_human = True` | `critical_urgency_escalation` |
| **Short / Sparse Input** ($\le 3$ words) | Force `requires_human = True`, penalize confidence | `short_input_context_sparsity` |
| **Low Calibrated Confidence** ($< 0.65$) | Force `requires_human = True` | `low_calibrated_confidence` |
| **JSON Parse / Validation Failure** | Force `category = general_query`, `requires_human = True`, `confidence = 0.0` | `parse_or_schema_validation_failure` |

---

## 4. Representative Request & Response Examples

### Example 1: English Payment Issue
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticket": "My payment was charged but my order was not confirmed."}'
```
**Response (HTTP 200)**:
```json
{
  "category": "payment_failure",
  "priority": "high",
  "language": "english",
  "requires_human": false,
  "confidence": 0.7499,
  "latency_ms": 1812.3,
  "model_version": "nexus-qlora-v1",
  "inference_mode": "neural",
  "escalation_reason": "automated_routing_permitted"
}
```

### Example 2: Hinglish Code-Switched Ticket
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticket": "Mera payment kat gaya but order confirm nahi hua."}'
```
**Response (HTTP 200)**:
```json
{
  "category": "payment_failure",
  "priority": "high",
  "language": "hinglish",
  "requires_human": false,
  "confidence": 0.7498,
  "latency_ms": 1845.1,
  "model_version": "nexus-qlora-v1",
  "inference_mode": "neural"
}
```

### Example 3: Safety-Critical Fraud Ticket (Deterministic Escalation)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticket": "Someone hacked into my account and transferred $2,000 without my authorization!"}'
```
**Response (HTTP 200)**:
```json
{
  "category": "fraud_report",
  "priority": "critical",
  "language": "english",
  "requires_human": true,
  "confidence": 0.95,
  "escalation_reason": "safety_critical_fraud_detection"
}
```

### Example 4: Malformed Request / Rejection
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticket": "   "}'
```
**Response (HTTP 422)**:
```json
{
  "error": "Request validation failed",
  "error_code": "INVALID_INPUT_SCHEMA",
  "detail": "Ticket text cannot be empty or whitespace only.",
  "request_id": "c803ea40-3f11-477c-a496-c16f272a5a2e",
  "timestamp": "2026-10-09T21:40:00Z"
}
```
