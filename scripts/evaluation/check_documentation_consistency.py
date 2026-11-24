#!/usr/bin/env python3
"""Automated consistency validator for NEXUS-SLM evaluation artifacts and documentation.

Verifies that:
1. All JSON evaluation artifacts in experiments/results/ conform to the Experiment Tracking Standard schema.
2. Machine-readable metrics match documented canonical values within floating-point tolerance (eps = 0.005).
3. Critical safety invariants (critical_recall == 1.0, critical_false_negatives == 0) hold across all runs.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"

EXPECTED_FILES = [
    "eval_human_holdout_baseline_v1.json",
    "eval_human_holdout_refined_v2.json",
    "eval_synthetic_test_baseline_v1.json",
    "eval_synthetic_test_refined_v2.json"
]

CANONICAL_TARGETS = {
    "eval_human_holdout_refined_v2.json": {
        "sample_count": 184,
        "category_metrics.accuracy": 0.6033,
        "priority_metrics.accuracy": 0.5978,
        "priority_metrics.macro_f1": 0.6519,
        "language_metrics.overall_accuracy": 0.9728,
        "escalation_metrics.overall_escalation_rate": 0.5870,
        "critical_safety_metrics.critical_recall": 1.0,
        "critical_safety_metrics.critical_false_negatives": 0,
    },
    "eval_human_holdout_baseline_v1.json": {
        "sample_count": 184,
        "escalation_metrics.overall_escalation_rate": 1.0,
        "critical_safety_metrics.critical_recall": 1.0,
        "critical_safety_metrics.critical_false_negatives": 0,
    },
    "eval_synthetic_test_refined_v2.json": {
        "sample_count": 746,
        "category_metrics.accuracy": 0.5751,
        "language_metrics.overall_accuracy": 0.8123,
        "escalation_metrics.overall_escalation_rate": 0.7064,
        "critical_safety_metrics.critical_recall": 1.0,
        "critical_safety_metrics.critical_false_negatives": 0,
    },
    "eval_synthetic_test_baseline_v1.json": {
        "sample_count": 746,
        "escalation_metrics.overall_escalation_rate": 1.0,
        "critical_safety_metrics.critical_recall": 1.0,
        "critical_safety_metrics.critical_false_negatives": 0,
    }
}

def get_nested(d: Dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    val = d
    for p in parts:
        val = val[p]
    return val

def validate_artifact(file_path: Path) -> List[str]:
    errors = []
    if not file_path.exists():
        return [f"Missing expected artifact: {file_path.name}"]
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Required top-level keys
    required_keys = [
        "metadata", "sample_count", "category_metrics", "priority_metrics",
        "language_metrics", "escalation_metrics", "critical_safety_metrics", "composite_metrics"
    ]
    for k in required_keys:
        if k not in data:
            errors.append(f"{file_path.name}: missing required top-level key '{k}'")
            
    # Check targets if defined
    name = file_path.name
    if name in CANONICAL_TARGETS:
        for path, expected in CANONICAL_TARGETS[name].items():
            try:
                actual = get_nested(data, path)
                if isinstance(expected, float):
                    if abs(actual - expected) > 0.005:
                        errors.append(f"{name} [{path}]: expected {expected:.4f}, got {actual:.4f}")
                else:
                    if actual != expected:
                        errors.append(f"{name} [{path}]: expected {expected}, got {actual}")
            except KeyError:
                errors.append(f"{name}: path '{path}' not found in artifact")
                
    # Strict invariant checks
    crit_recall = data.get("critical_safety_metrics", {}).get("critical_recall", 0.0)
    crit_fn = data.get("critical_safety_metrics", {}).get("critical_false_negatives", -1)
    if crit_recall < 0.999:
        errors.append(f"{name}: CRITICAL SAFETY INVARIANT VIOLATION - critical recall is {crit_recall} (< 1.0)")
    if crit_fn != 0:
        errors.append(f"{name}: CRITICAL SAFETY INVARIANT VIOLATION - critical false negatives is {crit_fn} (!= 0)")
        
    return errors

def main():
    all_errors = []
    print(f"Checking evaluation artifacts in {RESULTS_DIR}...")
    
    for filename in EXPECTED_FILES:
        artifact_path = RESULTS_DIR / filename
        errs = validate_artifact(artifact_path)
        if errs:
            all_errors.extend(errs)
        else:
            print(f"  [PASS] {filename}")
            
    if all_errors:
        print("\nERRORS DETECTED:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
        
    print("\nAll evaluation artifacts match schema, targets, and critical safety invariants.")
    sys.exit(0)

if __name__ == "__main__":
    main()
