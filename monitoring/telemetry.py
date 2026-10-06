"""Lightweight privacy-preserving telemetry and inference monitoring."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter


class TelemetryLogger:
    """Records operational request metrics without storing sensitive customer ticket content."""

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
        is_error: bool = False,
        error_msg: Optional[str] = None,
        text: Optional[str] = None,
    ) -> None:
        """Appends request metrics event to JSONL telemetry log."""
        if not self.enabled:
            return

        entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "priority": priority,
            "language": language,
            "requires_human": requires_human,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "model_version": model_version,
            "is_error": is_error,
            "error_msg": error_msg,
        }

        if self.log_ticket_text and text:
            entry["ticket_text"] = text

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculates aggregated telemetry statistics from recorded requests."""
        if not self.log_file.exists():
            return {
                "total_requests": 0,
                "error_count": 0,
                "mean_latency_ms": 0.0,
                "categories": {},
                "languages": {},
                "priorities": {},
            }

        total = 0
        errors = 0
        latencies = []
        cat_counter: Counter = Counter()
        lang_counter: Counter = Counter()
        prio_counter: Counter = Counter()
        human_req_count = 0

        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    total += 1
                    if data.get("is_error"):
                        errors += 1
                    if "latency_ms" in data and data["latency_ms"] is not None:
                        latencies.append(float(data["latency_ms"]))
                    cat_counter[data.get("category", "unknown")] += 1
                    lang_counter[data.get("language", "unknown")] += 1
                    prio_counter[data.get("priority", "unknown")] += 1
                    if data.get("requires_human"):
                        human_req_count += 1
                except Exception:
                    pass

        mean_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        p50 = round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0.0

        return {
            "total_requests": total,
            "error_count": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "mean_latency_ms": mean_lat,
            "p50_latency_ms": p50,
            "human_escalation_count": human_req_count,
            "human_escalation_rate": round(human_req_count / total, 4) if total else 0.0,
            "categories": dict(cat_counter),
            "languages": dict(lang_counter),
            "priorities": dict(prio_counter),
        }
