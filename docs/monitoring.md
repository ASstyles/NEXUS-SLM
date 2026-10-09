# NEXUS-SLM Production Monitoring & Data Drift Architecture

## 1. Observability Principles

Enterprise deployment requires comprehensive observability while strictly respecting user privacy:
1. **Privacy By Design**: Customer ticket text is **never** written to production telemetry logs by default (`LOG_SENSITIVE_TEXT=false`).
2. **Deterministic Correlation**: Every inference request is assigned a unique UUID `request_id` passed back in the HTTP response and recorded in the audit logs.
3. **Continuous Tracking**: All telemetry is appended in high-throughput JSONL format (`monitoring/telemetry.jsonl`).

---

## 2. Telemetry Event Schema

```json
{
  "timestamp": "2026-10-09T21:44:12.301294Z",
  "request_id": "9d023b34-d309-4ce8-b5e1-840250df7a69",
  "status": "success",
  "category": "payment_failure",
  "priority": "high",
  "language": "hinglish",
  "requires_human": false,
  "confidence": 0.7498,
  "latency_ms": 1824.5,
  "model_version": "nexus-qlora-v1",
  "output_validity": true,
  "ticket_length": 48,
  "is_error": false,
  "error_type": null,
  "error_msg": null
}
```

---

## 3. Operational Monitoring Metrics (`GET /metrics`)

The monitoring engine calculates live aggregations across in-flight requests:
- **Total Request Count** & **Throughput (req/sec)**
- **Error Rate**: Percentage of requests encountering 5xx/4xx failures.
- **Latency Percentiles**: P50, P90, P95, and P99 latency tracking.
- **JSON Validity Rate**: Percentage of model generations that conform to strict schema without fallback intervention.
- **Human Escalation Rate**: Proportion of incoming tickets routed to human representatives.
- **Confidence Distribution**: Mean, standard deviation, min, and max calibrated confidence scores.
- **Class & Language Distributions**: Real-time traffic breakdowns across the 14 categories and 3 modalities.

---

## 4. Data Drift Detection Engine (`GET /monitoring/drift`)

### 4.1 Methodology
The drift detector monitors changes in language modalities, category balance, and ticket length using the **Population Stability Index (PSI)**:

$$\text{PSI} = \sum_{i=1}^{K} \left( P_{\text{actual}}(i) - P_{\text{baseline}}(i) \right) \times \ln\left( \frac{P_{\text{actual}}(i)}{P_{\text{baseline}}(i)} \right)$$

### 4.2 Baseline Reference Distribution
- **Language**: English (45%), Hinglish (35%), Hindi (20%).
- **Category**: Uniform across 14 classes (7.14% per category).
- **Average Ticket Length**: 55.0 characters.

### 4.3 Trigger Thresholds & Operational Response

| PSI Value | Status | Action Required |
|---|---|---|
| $\text{PSI} < 0.10$ | **Normal** | Stable distribution. No intervention. |
| $0.10 \le \text{PSI} < 0.25$ | **Moderate Shift** | Monitor queue. Investigate regional traffic trends. |
| $\text{PSI} \ge 0.25$ | **Significant Drift** | **Drift Alert triggered.** Flag sample for audit. Never automatically retrain. |

---

## 5. Active Degradation Alerts (`GET /monitoring/alerts`)

The monitoring engine raises automated operational alerts under the following empirical conditions:

1. **High Error Rate Alert**: `error_rate > 5.0%` (Severity: `CRITICAL`).
2. **High Latency Alert**: `P95 latency > 3000 ms` (Severity: `WARNING`).
3. **Low JSON Validity Alert**: `json_validity_rate < 90.0%` (Severity: `CRITICAL`).
4. **High Escalation Alert**: `human_escalation_rate > 60.0%` (Severity: `WARNING`).
