"""Automated pytest suite for evaluation artifact and documentation consistency."""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"

EXPECTED_ARTIFACTS = [
    "eval_human_holdout_baseline_v1.json",
    "eval_human_holdout_refined_v2.json",
    "eval_synthetic_test_baseline_v1.json",
    "eval_synthetic_test_refined_v2.json"
]

@pytest.mark.parametrize("artifact_name", EXPECTED_ARTIFACTS)
def test_evaluation_artifact_exists_and_valid(artifact_name):
    artifact_path = RESULTS_DIR / artifact_name
    assert artifact_path.exists(), f"Artifact {artifact_name} must exist"
    
    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "metadata" in data
    assert "sample_count" in data
    assert "category_metrics" in data
    assert "priority_metrics" in data
    assert "critical_safety_metrics" in data
    
    # Zero critical false negatives invariant
    crit_metrics = data["critical_safety_metrics"]
    assert crit_metrics["critical_recall"] == 1.0, f"{artifact_name}: critical recall must be 1.0"
    assert crit_metrics["critical_false_negatives"] == 0, f"{artifact_name}: critical false negatives must be 0"

def test_human_eval_v2_metrics_match_canonical_sprint_report():
    v2_path = RESULTS_DIR / "eval_human_holdout_refined_v2.json"
    with open(v2_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["sample_count"] == 184
    assert abs(data["priority_metrics"]["accuracy"] - 0.5978) < 0.005
    assert abs(data["priority_metrics"]["macro_f1"] - 0.6519) < 0.005
    assert abs(data["escalation_metrics"]["overall_escalation_rate"] - 0.5870) < 0.005
    assert abs(data["language_metrics"]["overall_accuracy"] - 0.9728) < 0.005

def test_baseline_v1_policy_forces_100_percent_escalation():
    v1_human_path = RESULTS_DIR / "eval_human_holdout_baseline_v1.json"
    with open(v1_human_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["escalation_metrics"]["overall_escalation_rate"] == 1.0
