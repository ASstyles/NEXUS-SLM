# NEXUS-SLM Experiment Tracking & Evaluation Artifact Standard

## 1. Specification & Schema Purpose
To guarantee cryptographic reproducibility, eliminate unversioned metric claims, and prevent discrepancy recurrence, all model evaluations, benchmark reproductions, and offline validation runs across NEXUS-SLM must conform to the unified Experiment Tracking Standard.

Every evaluation run must output a machine-readable JSON artifact adhering to the schema specified below and reside in `experiments/results/`.

---

## 2. Machine-Readable Schema Specification

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "NEXUS-SLM Experiment Run Artifact",
  "type": "object",
  "required": [
    "metadata",
    "sample_count",
    "category_metrics",
    "priority_metrics",
    "language_metrics",
    "escalation_metrics",
    "critical_safety_metrics",
    "composite_metrics"
  ],
  "properties": {
    "metadata": {
      "type": "object",
      "required": [
        "dataset_key",
        "file",
        "sha256",
        "policy_version",
        "engine",
        "code_commit_sha",
        "timestamp_utc"
      ],
      "properties": {
        "dataset_key": {"type": "string", "enum": ["synthetic_locked_test", "human_eval_adjudicated", "synthetic_val_clean"]},
        "file": {"type": "string"},
        "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        "policy_version": {"type": "string", "enum": ["baseline_policy_v1", "refined_policy_v2"]},
        "engine": {"type": "string"},
        "code_commit_sha": {"type": "string"},
        "timestamp_utc": {"type": "string", "format": "date-time"},
        "description": {"type": "string"}
      }
    },
    "sample_count": {"type": "integer", "minimum": 1},
    "category_metrics": {
      "type": "object",
      "required": ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"],
      "properties": {
        "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
        "macro_precision": {"type": "number", "minimum": 0, "maximum": 1},
        "macro_recall": {"type": "number", "minimum": 0, "maximum": 1},
        "macro_f1": {"type": "number", "minimum": 0, "maximum": 1},
        "weighted_f1": {"type": "number", "minimum": 0, "maximum": 1}
      }
    },
    "priority_metrics": {
      "type": "object",
      "required": ["accuracy", "macro_f1", "weighted_f1", "per_class_recall"],
      "properties": {
        "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
        "macro_f1": {"type": "number", "minimum": 0, "maximum": 1},
        "weighted_f1": {"type": "number", "minimum": 0, "maximum": 1},
        "per_class_recall": {
          "type": "object",
          "required": ["low", "medium", "high", "critical"]
        }
      }
    },
    "language_metrics": {
      "type": "object",
      "required": ["overall_accuracy", "breakdown"]
    },
    "escalation_metrics": {
      "type": "object",
      "required": ["overall_escalation_rate", "unnecessary_escalation_rate"]
    },
    "critical_safety_metrics": {
      "type": "object",
      "required": ["critical_recall", "critical_false_negatives", "safety_violation_count"],
      "properties": {
        "critical_recall": {"type": "number", "minimum": 0, "maximum": 1},
        "critical_false_negatives": {"type": "integer", "minimum": 0},
        "safety_violation_count": {"type": "integer", "minimum": 0}
      }
    },
    "composite_metrics": {
      "type": "object",
      "required": ["exact_4field_match_rate"]
    }
  }
}
```

---

## 3. Policy Version Disambiguation Rules

Every evaluation artifact MUST specify `policy_version` explicitly:
1. `baseline_policy_v1`: Static category-to-priority dictionary lookup (`STATIC_PRIORITY_MAP_V1`) and unconditional fallback escalation upon heuristic classification (`requires_human = True` when `confidence is None`).
2. `refined_policy_v2`: Multi-factor contextual priority rubric (`determine_contextual_priority`) taking into account urgency indicators, financial/security triggers, and routine inquiry offsets, coupled with 7-tier escalation rules permitting routine non-risky heuristic inquiries to execute via `automated_routing_permitted`.

---

## 4. Benchmark Dataset Identifiers & Locked Hashes

| Dataset Key | Partition Path | Sample Size | Canonical SHA-256 |
|---|---|---|---|
| `synthetic_locked_test` | `data/v2_clean_template_grouped/test_clean.jsonl` | 746 | `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b` |
| `human_eval_adjudicated` | `data/human_eval/annotation_template.jsonl` | 184 | `e26d4eec99ddd247d3037652466df0024100b0cc85b9ca48e0fa32ed300923d3` |
| `synthetic_val_clean` | `data/v2_clean_template_grouped/val_clean.jsonl` | 744 | `88c3e8ec9ab5ee2a7fe018260a92548480a875a59334fe8923a4b7fcf7164cf3` |

Modifying `test_clean.jsonl` is strictly prohibited under project data governance policies.
