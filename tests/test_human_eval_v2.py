"""Tests for Human Evaluation V2 governance, Cohen's Kappa, deduplication, and leakage prevention."""

import json
from pathlib import Path

from scripts.data.human_eval_v2_manager import (
    compute_cohens_kappa,
    init_scaffold,
    prepare_blind_packets,
    validate_and_adjudicate,
)


def test_cohens_kappa_exact_and_chance_agreement():
    """Verify unweighted and linear-weighted Cohen's Kappa math."""
    res_perfect = compute_cohens_kappa(["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"])
    assert res_perfect["raw_agreement"] == 1.0
    assert res_perfect["kappa"] == 1.0

    # Symmetric 2x2 disagreement: [[20, 5], [10, 15]] -> Po=0.70, Pe=(25*30 + 25*20)/2500=0.50 -> K=0.40
    la = ["x"] * 25 + ["y"] * 25
    lb = ["x"] * 20 + ["y"] * 5 + ["x"] * 10 + ["y"] * 15
    res_2x2 = compute_cohens_kappa(la, lb, ["x", "y"])
    assert res_2x2["raw_agreement"] == 0.7
    assert res_2x2["kappa"] == 0.4

    # Linear weighted ordinal priority kappa
    weights = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    res_w = compute_cohens_kappa(
        ["low", "medium", "high", "critical"],
        ["low", "high", "high", "critical"],
        ["low", "medium", "high", "critical"],
        linear_weights=weights,
    )
    assert res_w["status"] == "OK"
    assert res_w["weighted_kappa"] > 0.7


def test_human_eval_v2_scaffold_is_honestly_pending():
    """Repo data/human_eval_v2 must start with >=300 sealed slots and honest PENDING status."""
    v2_dir = Path("data/human_eval_v2")
    assert (v2_dir / "allocation_plan_v2.json").exists()
    assert (v2_dir / "annotation_guidelines_v2.0.0.json").exists()
    assert (v2_dir / "consent_and_privacy_checklist.json").exists()
    assert (v2_dir / "intake_collection_template.jsonl").exists()

    plan = json.loads((v2_dir / "allocation_plan_v2.json").read_text(encoding="utf-8"))
    assert plan["status"] == "PENDING"
    assert plan["partitions"]["sealed_final_eval"]["target_count"] >= 300
    assert plan["governance_rules"]["v1_184_scenarios_excluded_from_cohort"] is True

    rep = json.loads(Path("experiments/artifacts/human_eval_v2_validation_status.json").read_text(encoding="utf-8"))
    assert rep["status"] == "PENDING"
    assert rep["counts"]["target_minimum_sealed"] == 300


def test_blind_validation_rejects_self_pairing_leakage_and_unresolved(tmp_path):
    """Ensure validator catches identical annotators, model IDs, duplicates, and V1/test overlap."""
    d = tmp_path / "hv2"
    init_scaffold(d)

    intake_p = d / "intake_collection_template.jsonl"
    rows = [json.loads(l) for l in intake_p.read_text(encoding="utf-8").splitlines() if l.strip()][:3]
    rows[0]["text"] = "Unique payment issue never seen before on mobile wallet 99182"
    rows[1]["text"] = "Unique payment issue never seen before on mobile wallet 99182"  # duplicate of row 0
    rows[2]["text"] = "Existing exposed scenario from test"
    intake_p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    exposed_file = tmp_path / "exposed.jsonl"
    exposed_file.write_text(json.dumps({"text": "Existing exposed scenario from test"}) + "\n", encoding="utf-8")

    pa = d / "blind_annotator_a_packet.jsonl"
    pb = d / "blind_annotator_b_packet.jsonl"
    prepare_blind_packets(intake_p, pa, pb)

    # Fill annotations with one self-paired (invalid), one disagreement, and one agreement
    a_recs = [json.loads(l) for l in pa.read_text(encoding="utf-8").splitlines() if l.strip()]
    b_recs = [json.loads(l) for l in pb.read_text(encoding="utf-8").splitlines() if l.strip()]

    for r in a_recs:
        r.update({"category": "payment_failure", "priority": "high", "language": "english", "requires_human": True, "annotator_id": "human_rev_A"})
    for idx, r in enumerate(b_recs):
        r.update({
            "category": "billing" if idx == 0 else "payment_failure",
            "priority": "high",
            "language": "english",
            "requires_human": True,
            "annotator_id": "human_rev_A" if idx == 2 else "human_rev_B",
        })

    pa.write_text("\n".join(json.dumps(r) for r in a_recs) + "\n", encoding="utf-8")
    pb.write_text("\n".join(json.dumps(r) for r in b_recs) + "\n", encoding="utf-8")

    res = validate_and_adjudicate(intake_p, pa, pb, exposed_paths=[exposed_file])
    assert res["status"] == "PENDING"
    assert len(res["data_quality"]["exact_duplicate_pairs"]) == 1
    assert len(res["data_quality"]["cross_dataset_leakage_hits"]) == 1
    assert any("blind violation" in e for e in res["data_quality"]["provenance_errors"])
    assert res["data_quality"]["disagreements_unresolved"] == 1
