"""Production-grade privacy-preserving telemetry, monitoring metrics, and data drift detection."""

import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter
import numpy as np


class TelemetryLogger:
    """Records operational request metrics without storing sensitive customer ticket content."""

    # Baseline reference distributions from locked training/evaluation data
    BASELINE_LANGUAGES = {"english": 0.45, "hinglish": 0.35, "hindi": 0.20}
    BASELINE_CATEGORIES = {
        "payment_failure": 0.0714,
        "refund_request": 0.0714,
        "account_access": 0.0714,
        "password_reset": 0.0714,
        "order_cancel": 0.0714,
        "order_tracking": 0.0714,
        "subscription": 0.0714,
        "billing": 0.0714,
        "technical_issue": 0.0714,
        "fraud_report": 0.0714,
        "delivery_issue": 0.0714,
        "account_closure": 0.0714,
        "product_complaint": 0.0714,
        "general_query": 0.0714,
    }
    BASELINE_AVG_TICKET_LEN = 55.0  # characters

    def __init__(
        self,
        log_file: str = "monitoring/telemetry.jsonl",
        enabled: bool = True,
        log_ticket_text: bool = False,
    ):
        self.log_file = Path(log_file)
        self.enabled = enabled
        self.log_ticket_text = log_ticket_text
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_request(
        self,
        category: str,
        priority: str,
        language: str,
        requires_human: bool,
        confidence: float,
        latency_ms: float,
        model_version: str,
        request_id: Optional[str] = None,
        is_error: bool = False,
        error_type: Optional[str] = None,
        error_msg: Optional[str] = None,
        output_validity: bool = True,
        ticket_length: int = 0,
        text: Optional[str] = None,
    ) -> str:
        """Appends request metrics event to JSONL telemetry log without sensitive text by default."""
        if not self.enabled:
            return request_id or str(uuid.uuid4())

        rid = request_id or str(uuid.uuid4())
        entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": rid,
            "status": "error" if is_error else "success",
            "category": category,
            "priority": priority,
            "language": language,
            "requires_human": requires_human,
            "confidence": round(float(confidence), 4),
            "latency_ms": round(float(latency_ms), 2),
            "model_version": model_version,
            "output_validity": output_validity,
            "ticket_length": ticket_length,
            "is_error": is_error,
            "error_type": error_type,
            "error_msg": error_msg,
        }

        # Privacy guard: NEVER log ticket text unless explicitly configured
        if self.log_ticket_text and text:
            entry["ticket_text"] = text

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

        return rid

    def _read_records(self) -> List[Dict[str, Any]]:
        """Reads all records from the telemetry log file."""
        records = []
        if not self.log_file.exists():
            return records
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculates aggregated telemetry statistics, latency percentiles, and operational alerts."""
        records = self._read_records()
        total = len(records)
        if total == 0:
            return {
                "total_requests": 0,
                "error_count": 0,
                "error_rate": 0.0,
                "mean_latency_ms": 0.0,
                "latency_percentiles": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0},
                "human_escalation_count": 0,
                "human_escalation_rate": 0.0,
                "json_validation_failures": 0,
                "json_validity_rate": 1.0,
                "confidence_distribution": {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0},
                "categories": {},
                "languages": {},
                "priorities": {},
                "active_alerts": [],
            }

        errors = 0
        json_failures = 0
        latencies = []
        confidences = []
        cat_counter: Counter = Counter()
        lang_counter: Counter = Counter()
        prio_counter: Counter = Counter()
        human_req_count = 0

        for r in records:
            if r.get("is_error"):
                errors += 1
            if not r.get("output_validity", True):
                json_failures += 1
            if r.get("latency_ms") is not None:
                latencies.append(float(r["latency_ms"]))
            if r.get("confidence") is not None:
                confidences.append(float(r["confidence"]))
            cat_counter[r.get("category", "unknown")] += 1
            lang_counter[r.get("language", "unknown")] += 1
            prio_counter[r.get("priority", "unknown")] += 1
            if r.get("requires_human"):
                human_req_count += 1

        lat_arr = np.array(latencies) if latencies else np.array([0.0])
        conf_arr = np.array(confidences) if confidences else np.array([0.0])

        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))

        error_rate = round(errors / total, 4)
        json_validity_rate = round((total - json_failures) / total, 4)
        human_esc_rate = round(human_req_count / total, 4)

        # Operational alerts
        active_alerts = []
        if error_rate > 0.05:
            active_alerts.append({
                "severity": "CRITICAL",
                "alert": "high_error_rate",
                "value": error_rate,
                "threshold": 0.05,
                "message": f"Error rate ({error_rate*100:.1f}%) exceeds 5% threshold."
            })
        if p95 > 3000.0:
            active_alerts.append({
                "severity": "WARNING",
                "alert": "high_p95_latency",
                "value": p95,
                "threshold": 3000.0,
                "message": f"P95 latency ({p95:.1f}ms) exceeds 3000ms threshold."
            })
        if json_validity_rate < 0.90:
            active_alerts.append({
                "severity": "CRITICAL",
                "alert": "low_json_validity",
                "value": json_validity_rate,
                "threshold": 0.90,
                "message": f"JSON validity ({json_validity_rate*100:.1f}%) below 90% threshold."
            })
        if human_esc_rate > 0.60:
            active_alerts.append({
                "severity": "WARNING",
                "alert": "high_human_escalation_rate",
                "value": human_esc_rate,
                "threshold": 0.60,
                "message": f"Human escalation rate ({human_esc_rate*100:.1f}%) exceeds 60% threshold."
            })

        return {
            "total_requests": total,
            "error_count": errors,
            "error_rate": error_rate,
            "mean_latency_ms": round(float(np.mean(lat_arr)), 2),
            "latency_percentiles": {
                "p50": round(p50, 2),
                "p90": round(p90, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2),
            },
            "human_escalation_count": human_req_count,
            "human_escalation_rate": human_esc_rate,
            "json_validation_failures": json_failures,
            "json_validity_rate": json_validity_rate,
            "confidence_distribution": {
                "mean": round(float(np.mean(conf_arr)), 4),
                "std": round(float(np.std(conf_arr)), 4),
                "min": round(float(np.min(conf_arr)), 4),
                "max": round(float(np.max(conf_arr)), 4),
            },
            "categories": dict(cat_counter),
            "languages": dict(lang_counter),
            "priorities": dict(prio_counter),
            "active_alerts": active_alerts,
        }

    def compute_data_drift(self) -> Dict[str, Any]:
        """
        Calculates Population Stability Index (PSI) and distribution shifts against baseline reference.
        PSI = sum((actual% - expected%) * ln(actual% / expected%))
        Rules:
        PSI < 0.10: No significant shift
        0.10 <= PSI < 0.25: Moderate shift, observe
        PSI >= 0.25: Significant shift, trigger drift alert for investigation
        """
        records = self._read_records()
        total = len(records)
        if total < 10:
            return {
                "status": "INSUFFICIENT_DATA",
                "sample_count": total,
                "min_samples_required": 10,
                "language_psi": 0.0,
                "category_psi": 0.0,
                "drift_detected": False,
                "alerts": [],
            }

        # Language distribution
        lang_counts = Counter(r.get("language") for r in records if r.get("language"))
        observed_lang_dist = {k: lang_counts.get(k, 0) / total for k in self.BASELINE_LANGUAGES}

        lang_psi = 0.0
        for k, p_expected in self.BASELINE_LANGUAGES.items():
            p_actual = max(observed_lang_dist.get(k, 0.001), 0.001)
            lang_psi += (p_actual - p_expected) * math.log(p_actual / p_expected)

        # Category distribution
        cat_counts = Counter(r.get("category") for r in records if r.get("category"))
        cat_psi = 0.0
        for k, p_expected in self.BASELINE_CATEGORIES.items():
            p_actual = max(cat_counts.get(k, 0) / total, 0.001)
            cat_psi += (p_actual - p_expected) * math.log(p_actual / p_expected)

        # Ticket length drift
        lengths = [r.get("ticket_length", 0) for r in records if r.get("ticket_length", 0) > 0]
        avg_len = float(np.mean(lengths)) if lengths else self.BASELINE_AVG_TICKET_LEN
        len_shift_pct = round(((avg_len - self.BASELINE_AVG_TICKET_LEN) / self.BASELINE_AVG_TICKET_LEN) * 100, 2)

        alerts = []
        if lang_psi >= 0.25:
            alerts.append({
                "type": "LANGUAGE_DISTRIBUTION_DRIFT",
                "psi": round(lang_psi, 4),
                "threshold": 0.25,
                "message": f"Significant language distribution shift (PSI={lang_psi:.3f} >= 0.25).",
            })
        if cat_psi >= 0.25:
            alerts.append({
                "type": "CATEGORY_DISTRIBUTION_DRIFT",
                "psi": round(cat_psi, 4),
                "threshold": 0.25,
                "message": f"Significant category distribution shift (PSI={cat_psi:.3f} >= 0.25).",
            })
        if abs(len_shift_pct) > 50.0:
            alerts.append({
                "type": "TICKET_LENGTH_DRIFT",
                "shift_pct": len_shift_pct,
                "threshold": 50.0,
                "message": f"Ticket length drifted by {len_shift_pct}% from baseline.",
            })

        drift_detected = len(alerts) > 0

        return {
            "status": "DRIFT_ALERT" if drift_detected else "NORMAL",
            "sample_count": total,
            "language_psi": round(lang_psi, 4),
            "category_psi": round(cat_psi, 4),
            "observed_language_distribution": {k: round(v, 4) for k, v in observed_lang_dist.items()},
            "baseline_language_distribution": self.BASELINE_LANGUAGES,
            "average_ticket_length": round(avg_len, 2),
            "baseline_ticket_length": self.BASELINE_AVG_TICKET_LEN,
            "ticket_length_shift_pct": len_shift_pct,
            "drift_detected": drift_detected,
            "alerts": alerts,
        }
