"""Reproducible governance, blind annotation, Cohen's Kappa, and validation manager for Human Eval V2.

Enforces independent human evaluation protocols:
- Strict separation between development-exposed scenarios (e.g., 184 V1 scenarios) and V2 partitions.
- Dual-blind independent annotation (Annotator A != Annotator B) with provenance metadata.
- Exact and near-duplicate detection + cross-dataset leakage checks.
- Third-party adjudication queue and resolution merging.
- Honest PENDING state enforcement when external human authoring/annotations are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES

GUIDELINE_VERSION = "2.0.0"

REQUIRED_CHALLENGE_TAGS = [
    "standard",
    "mixed_language",
    "ambiguous",
    "negative_control",
    "multi_intent",
    "financial_security_risk",
    "physical_safety_hazard",
    "typo_unusual_phrasing",
]

PRIORITY_ORDINAL = {p: idx for idx, p in enumerate(PRIORITIES)}

EXISTING_EXPOSED_DATASETS = [
    PROJECT_ROOT / "data/v2_clean_template_grouped/train_clean.jsonl",
    PROJECT_ROOT / "data/v2_clean_template_grouped/val_clean.jsonl",
    PROJECT_ROOT / "data/v2_clean_template_grouped/test_clean.jsonl",
    PROJECT_ROOT / "data/human_eval/annotation_template.jsonl",
    PROJECT_ROOT / "data/evaluation/high_risk_challenge_set.jsonl",
    PROJECT_ROOT / "data/train/train.jsonl",
    PROJECT_ROOT / "data/val/val.jsonl",
    PROJECT_ROOT / "data/test/test.jsonl",
]


def normalize_text(text: str) -> str:
    """Normalize text for exact and n-gram near-duplicate comparison across scripts."""
    lowered = text.lower().strip()
    cleaned = re.sub(r"[^\w\u0900-\u097f\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def token_jaccard(a: str, b: str) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_cohens_kappa(
    labels_a: Sequence[Any],
    labels_b: Sequence[Any],
    possible_labels: Optional[Sequence[Any]] = None,
    linear_weights: Optional[Dict[Any, int]] = None,
) -> Dict[str, Any]:
    """Compute unweighted or linearly weighted Cohen's Kappa with raw agreement rate."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return {
            "n": 0,
            "raw_agreement": None,
            "kappa": None,
            "status": "INSUFFICIENT_DATA",
        }

    n = len(labels_a)
    raw_agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    if possible_labels is None:
        cats = sorted(set(labels_a) | set(labels_b), key=str)
    else:
        cats = list(possible_labels)

    if len(cats) <= 1:
        return {
            "n": n,
            "raw_agreement": round(raw_agree, 4),
            "kappa": 1.0 if raw_agree == 1.0 else 0.0,
            "status": "DEGENERATE_SINGLE_CLASS",
        }

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)

    if linear_weights is None:
        pe = sum((count_a[c] / n) * (count_b[c] / n) for c in cats)
        if abs(1.0 - pe) < 1e-12:
            kappa = 1.0 if raw_agree == 1.0 else 0.0
        else:
            kappa = (raw_agree - pe) / (1.0 - pe)
        return {
            "n": n,
            "raw_agreement": round(raw_agree, 4),
            "expected_agreement": round(pe, 4),
            "kappa": round(kappa, 4),
            "status": "OK",
        }

    max_dist = max(linear_weights.values()) - min(linear_weights.values())
    if max_dist <= 0:
        raise ValueError("linear_weights must span at least two distinct ordinal levels")

    obs_weighted_disagree = sum(
        abs(linear_weights[a] - linear_weights[b]) / max_dist
        for a, b in zip(labels_a, labels_b)
    ) / n
    exp_weighted_disagree = sum(
        (count_a[ci] / n) * (count_b[cj] / n) * (abs(linear_weights[ci] - linear_weights[cj]) / max_dist)
        for ci in cats
        for cj in cats
    )
    if exp_weighted_disagree < 1e-12:
        kappa_w = 1.0 if obs_weighted_disagree == 0.0 else 0.0
    else:
        kappa_w = 1.0 - (obs_weighted_disagree / exp_weighted_disagree)

    return {
        "n": n,
        "raw_agreement": round(raw_agree, 4),
        "weighted_kappa": round(kappa_w, 4),
        "status": "OK",
    }


def build_allocation_slots() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Create balanced intake slot allocation: 320 sealed_final_eval + 80 dev_calibration_reserved."""
    # Ensure at least 22 scenarios per category in sealed_final_eval (14 * 22 = 308 -> +12 safety/edge = 320)
    slots: List[Dict[str, Any]] = []
    langs_sealed = (["english"] * 100) + (["hindi"] * 80) + (["hinglish"] * 90) + (["mixed_language"] * 50)
    prios_sealed = (["low"] * 70) + (["medium"] * 110) + (["high"] * 95) + (["critical"] * 45)
    challenge_cycle = [
        "standard",
        "standard",
        "mixed_language",
        "ambiguous",
        "negative_control",
        "multi_intent",
        "financial_security_risk",
        "physical_safety_hazard",
        "typo_unusual_phrasing",
    ]

    for i in range(320):
        target_cat = CATEGORIES[i % len(CATEGORIES)]
        target_lang = langs_sealed[(i * 7) % len(langs_sealed)]
        # Override challenge to cover mixed_language when lang stratum is mixed_language
        challenge = "mixed_language" if target_lang == "mixed_language" else challenge_cycle[i % len(challenge_cycle)]
        if target_cat in ("fraud_report", "payment_failure", "account_access") and i % 3 == 0:
            challenge = "financial_security_risk"
        target_prio = prios_sealed[(i * 11) % len(prios_sealed)]
        if challenge == "physical_safety_hazard":
            target_prio = "critical"
        elif challenge == "negative_control" and target_prio == "critical":
            target_prio = "low"

        slots.append(
            {
                "slot_id": f"hv2_sealed_{i + 1:04d}",
                "partition": "sealed_final_eval",
                "target_category_hint": target_cat,
                "target_priority_hint": target_prio,
                "target_language_stratum": target_lang,
                "challenge_stratum": challenge,
                "text": None,
                "source_channel": "PENDING_INDEPENDENT_HUMAN_AUTHORING",
                "author_id": None,
                "authored_at_utc": None,
                "consent_verified": None,
                "pii_redacted": None,
                "annotator_a": None,
                "annotator_b": None,
                "adjudication": None,
                "guideline_version": GUIDELINE_VERSION,
                "status": "PENDING_INDEPENDENT_AUTHORING",
            }
        )

    for j in range(80):
        target_cat = CATEGORIES[j % len(CATEGORIES)]
        target_lang = ["english", "hindi", "hinglish", "mixed_language"][j % 4]
        target_prio = PRIORITIES[j % len(PRIORITIES)]
        challenge = challenge_cycle[j % len(challenge_cycle)]
        slots.append(
            {
                "slot_id": f"hv2_dev_{j + 1:04d}",
                "partition": "dev_calibration_reserved",
                "target_category_hint": target_cat,
                "target_priority_hint": target_prio,
                "target_language_stratum": target_lang,
                "challenge_stratum": challenge,
                "text": None,
                "source_channel": "PENDING_INDEPENDENT_HUMAN_AUTHORING",
                "author_id": None,
                "authored_at_utc": None,
                "consent_verified": None,
                "pii_redacted": None,
                "annotator_a": None,
                "annotator_b": None,
                "adjudication": None,
                "guideline_version": GUIDELINE_VERSION,
                "status": "PENDING_INDEPENDENT_AUTHORING",
            }
        )

    plan = {
        "schema_version": "human_eval_v2.0",
        "guideline_version": GUIDELINE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PENDING",
        "governance_rules": {
            "independent_human_authoring_required": True,
            "model_assisted_ground_truth_prohibited": True,
            "minimum_sealed_scenarios": 300,
            "planned_sealed_scenarios": 320,
            "planned_dev_calibration_scenarios": 80,
            "minimum_independent_annotators": 2,
            "blind_annotation_required": True,
            "adjudication_required_for_all_disagreements": True,
            "v1_184_scenarios_excluded_from_cohort": True,
            "sealed_partition_never_used_for_tuning": True,
        },
        "partitions": {
            "sealed_final_eval": {
                "target_count": 320,
                "category_distribution_target": dict(Counter(s["target_category_hint"] for s in slots if s["partition"] == "sealed_final_eval")),
                "priority_distribution_target": dict(Counter(s["target_priority_hint"] for s in slots if s["partition"] == "sealed_final_eval")),
                "language_stratum_target": dict(Counter(s["target_language_stratum"] for s in slots if s["partition"] == "sealed_final_eval")),
                "challenge_stratum_target": dict(Counter(s["challenge_stratum"] for s in slots if s["partition"] == "sealed_final_eval")),
            },
            "dev_calibration_reserved": {
                "target_count": 80,
                "purpose": "Post-authoring threshold/calibration validation separate from sealed_final_eval",
            },
        },
    }
    return plan, slots


def build_guidelines_doc() -> Dict[str, Any]:
    return {
        "guideline_version": GUIDELINE_VERSION,
        "effective_date_utc": datetime.now(timezone.utc).isoformat(),
        "taxonomy": {
            "categories": CATEGORIES,
            "priorities": PRIORITIES,
            "languages": LANGUAGES + ["mixed_language"],
            "language_mapping_note": (
                "During final model scoring against the 3 production language enums (english, hindi, hinglish), "
                "'mixed_language' packets track explicit Devanagari+Romanized code-switching separately in human governance "
                "and map to 'hinglish' only when evaluating legacy 3-class model outputs, while retaining mixed_language breakdowns."
            ),
        },
        "priority_rubric": {
            "critical": "Active fraud/unauthorized account takeover, physical safety hazard/fire/injury/shock, or immediate regulatory threat.",
            "high": "Money debited without order confirmation, account locked out completely, missing/tampered package after delivered scan, defective product safety issue.",
            "medium": "Standard order tracking delay, refund timeline inquiry, password reset friction, billing invoice discrepancy.",
            "low": "General inquiry, feature question, positive feedback, address/email preference update.",
        },
        "escalation_rubric": {
            "requires_human_true": [
                "Financial loss / double debit / failed transaction awaiting reconciliation",
                "Unauthorized account access / fraud / phishing / identity theft",
                "Physical safety hazard (burn, shock, smoke, allergic reaction, child hazard)",
                "Package tampered/empty box or false delivered status",
                "Legal threat / regulatory complaint / GDPR privacy demand",
                "Genunely ambiguous multi-intent dispute requiring account action",
            ],
            "requires_human_false_negative_controls": [
                "Negative control tickets mentioning high-risk keywords safely (e.g., 'Can I change my password later?', 'How do fraud alerts work in general?')",
                "Routine low-stakes queries solvable via self-service FAQ",
            ],
        },
        "multi_intent_rule": "Assign primary category to the highest-risk or financially/legally blocking intent; record secondary intents in metadata.",
        "blind_protocol_rules": [
            "No model predictions, heuristic outputs, or target hints may appear in annotator blind packets.",
            "Annotator A and Annotator B must be distinct verified human reviewers.",
            "Every disagreement on category, priority, language, or requires_human enters adjudication_queue.jsonl for third-party resolution.",
        ],
    }


def build_consent_checklist() -> Dict[str, Any]:
    return {
        "programme": "NEXUS-SLM Live Customer Intake & Independent Human Evaluation Governance",
        "version": GUIDELINE_VERSION,
        "status": "PENDING_LEGAL_AND_DPA_SIGNOFF",
        "mandatory_prerequisites_before_real_customer_collection": [
            {"id": "DPA-01", "requirement": "Executed Data Processing Agreement (DPA) / Customer Consent Notice covering support-ticket quality auditing", "completed": False},
            {"id": "PII-02", "requirement": "Automated + manual PII scrubbing (full names, phone numbers, card PANs, bank account numbers, government IDs, exact addresses)", "completed": False},
            {"id": "LAW-03", "requirement": "Documented lawful basis under applicable privacy framework (e.g., GDPR Art. 6 / DPDP Act India)", "completed": False},
            {"id": "RET-04", "requirement": "Defined data retention & erasure SLA and access audit logging", "completed": False},
            {"id": "OPT-05", "requirement": "Verified customer opt-out mechanism and zero retention of raw unredacted secrets in Git", "completed": False},
        ],
        "prohibition": "Synthetic or developer-written scenarios must NEVER be labeled as real customer production tickets.",
    }


def init_scaffold(base_dir: Path) -> Dict[str, Any]:
    base_dir.mkdir(parents=True, exist_ok=True)
    plan, slots = build_allocation_slots()
    guidelines = build_guidelines_doc()
    consent = build_consent_checklist()

    (base_dir / "allocation_plan_v2.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (base_dir / "annotation_guidelines_v2.0.0.json").write_text(json.dumps(guidelines, indent=2), encoding="utf-8")
    (base_dir / "consent_and_privacy_checklist.json").write_text(json.dumps(consent, indent=2), encoding="utf-8")

    with open(base_dir / "intake_collection_template.jsonl", "w", encoding="utf-8") as f:
        for slot in slots:
            f.write(json.dumps(slot, ensure_ascii=False) + "\n")

    # Empty placeholder files documenting exact schema for blind dual annotation & adjudication
    (base_dir / "blind_annotator_a_packet.jsonl").write_text("", encoding="utf-8")
    (base_dir / "blind_annotator_b_packet.jsonl").write_text("", encoding="utf-8")
    (base_dir / "adjudication_log_v2.jsonl").write_text("", encoding="utf-8")

    return {
        "directory": str(base_dir),
        "total_slots": len(slots),
        "sealed_slots": sum(1 for s in slots if s["partition"] == "sealed_final_eval"),
        "dev_slots": sum(1 for s in slots if s["partition"] == "dev_calibration_reserved"),
        "status": "PENDING",
    }


def _load_existing_exposed_texts(extra_paths: Optional[Sequence[Path]] = None) -> List[Tuple[str, str]]:
    """Load all existing train/val/test and dev-exposed 184 scenarios to enforce strict separation."""
    paths = list(extra_paths) if extra_paths is not None else EXISTING_EXPOSED_DATASETS
    corpus: List[Tuple[str, str]] = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            txt = rec.get("text")
            if isinstance(txt, str) and txt.strip():
                corpus.append((p.name, txt.strip()))
    return corpus


def prepare_blind_packets(intake_path: Path, out_a: Path, out_b: Path) -> Dict[str, Any]:
    """Generate clean blind packets stripping hints, author metadata, and model outputs."""
    records = [json.loads(l) for l in intake_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    authored = [r for r in records if isinstance(r.get("text"), str) and r["text"].strip()]
    count_written = 0
    with open(out_a, "w", encoding="utf-8") as fa, open(out_b, "w", encoding="utf-8") as fb:
        for rec in authored:
            blind_item = {
                "slot_id": rec["slot_id"],
                "partition": rec.get("partition", "sealed_final_eval"),
                "text": rec["text"].strip(),
                "guideline_version": GUIDELINE_VERSION,
                "category": None,
                "priority": None,
                "language": None,
                "requires_human": None,
                "challenge_tags": [],
                "annotator_id": None,
                "annotated_at_utc": None,
                "notes": "",
            }
            line = json.dumps(blind_item, ensure_ascii=False) + "\n"
            fa.write(line)
            fb.write(line)
            count_written += 1

    return {
        "authored_records_exported": count_written,
        "annotator_a_packet": str(out_a),
        "annotator_b_packet": str(out_b),
        "annotator_a_sha256": sha256_file(out_a),
        "annotator_b_sha256": sha256_file(out_b),
    }


def validate_and_adjudicate(
    intake_path: Path,
    ann_a_path: Path,
    ann_b_path: Path,
    adjudication_path: Optional[Path] = None,
    exposed_paths: Optional[Sequence[Path]] = None,
    near_dup_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Validate provenance, duplicates, leakage, blind dual annotations, Cohen's Kappa, and support matrix."""
    intake = [json.loads(l) for l in intake_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    authored = [r for r in intake if isinstance(r.get("text"), str) and r["text"].strip()]
    ann_a = {r["slot_id"]: r for r in (json.loads(l) for l in ann_a_path.read_text(encoding="utf-8").splitlines() if l.strip())} if ann_a_path.exists() else {}
    ann_b = {r["slot_id"]: r for r in (json.loads(l) for l in ann_b_path.read_text(encoding="utf-8").splitlines() if l.strip())} if ann_b_path.exists() else {}
    adj_map = {r["slot_id"]: r for r in (json.loads(l) for l in adjudication_path.read_text(encoding="utf-8").splitlines() if l.strip())} if (adjudication_path and adjudication_path.exists()) else {}

    issues: List[str] = []
    exact_dups: List[Tuple[str, str]] = []
    near_dups: List[Dict[str, Any]] = []
    leakage_hits: List[Dict[str, Any]] = []

    if not authored:
        issues.append("No authored scenarios present; independent collection is PENDING.")

    # Check duplicates & near-duplicates within authored cohort
    seen_norm: Dict[str, str] = {}
    authored_norms: List[Tuple[str, str, str]] = []
    for rec in authored:
        sid = rec["slot_id"]
        txt = rec["text"].strip()
        norm = normalize_text(txt)
        if norm in seen_norm:
            exact_dups.append((seen_norm[norm], sid))
        else:
            seen_norm[norm] = sid
        for prev_sid, prev_txt, prev_norm in authored_norms:
            jac = token_jaccard(txt, prev_txt)
            if jac >= near_dup_threshold:
                seq = SequenceMatcher(None, norm, prev_norm).ratio()
                if max(jac, seq) >= near_dup_threshold:
                    near_dups.append({"slot_a": prev_sid, "slot_b": sid, "jaccard": round(jac, 4), "seq_ratio": round(seq, 4)})
        authored_norms.append((sid, txt, norm))

    # Cross-dataset leakage check against V1/V2 splits and 184 dev-exposed human eval
    exposed_corpus = _load_existing_exposed_texts(exposed_paths)
    exposed_norm_map = {normalize_text(t): src for src, t in exposed_corpus}
    for sid, txt, norm in authored_norms:
        if norm in exposed_norm_map:
            leakage_hits.append({"slot_id": sid, "matched_source": exposed_norm_map[norm], "type": "exact_overlap"})

    # Annotator pairing, provenance & blindness validation
    paired_ids = sorted(set(ann_a.keys()) & set(ann_b.keys()))
    valid_langs = set(LANGUAGES + ["mixed_language"])

    cat_a, cat_b = [], []
    prio_a, prio_b = [], []
    lang_a, lang_b = [], []
    hum_a, hum_b = [], []
    disagreements: List[Dict[str, Any]] = []
    adjudicated_records: List[Dict[str, Any]] = []
    provenance_errors: List[str] = []

    for sid in paired_ids:
        ra, rb = ann_a[sid], ann_b[sid]
        aid_a = (ra.get("annotator_id") or "").strip()
        aid_b = (rb.get("annotator_id") or "").strip()
        if not aid_a or not aid_b:
            provenance_errors.append(f"{sid}: missing annotator_id")
            continue
        if aid_a == aid_b:
            provenance_errors.append(f"{sid}: blind violation (annotator_a == annotator_b == {aid_a})")
            continue
        if any(k in aid_a.lower() or k in aid_b.lower() for k in ("qwen", "smollm", "model", "gpt", "llm", "heuristic", "synthetic")):
            provenance_errors.append(f"{sid}: model/synthetic ID prohibited in human annotation provenance ({aid_a}, {aid_b})")
            continue

        # Validate label vocabularies
        for side_name, rec in (("annotator_a", ra), ("annotator_b", rb)):
            if rec.get("category") not in CATEGORIES:
                provenance_errors.append(f"{sid} ({side_name}): invalid category {rec.get('category')}")
            if rec.get("priority") not in PRIORITIES:
                provenance_errors.append(f"{sid} ({side_name}): invalid priority {rec.get('priority')}")
            if rec.get("language") not in valid_langs:
                provenance_errors.append(f"{sid} ({side_name}): invalid language {rec.get('language')}")
            if not isinstance(rec.get("requires_human"), bool):
                provenance_errors.append(f"{sid} ({side_name}): requires_human must be bool")

        if provenance_errors:
            continue

        cat_a.append(ra["category"])
        cat_b.append(rb["category"])
        prio_a.append(ra["priority"])
        prio_b.append(rb["priority"])
        lang_a.append(ra["language"])
        lang_b.append(rb["language"])
        hum_a.append(ra["requires_human"])
        hum_b.append(rb["requires_human"])

        fields_diff = [
            f for f in ("category", "priority", "language", "requires_human")
            if ra[f] != rb[f]
        ]
        if fields_diff:
            adj = adj_map.get(sid)
            resolved = bool(
                adj
                and adj.get("adjudicator_id")
                and adj["adjudicator_id"] not in (aid_a, aid_b)
                and adj.get("category") in CATEGORIES
                and adj.get("priority") in PRIORITIES
                and adj.get("language") in valid_langs
                and isinstance(adj.get("requires_human"), bool)
            )
            disagreements.append(
                {
                    "slot_id": sid,
                    "disagreed_fields": fields_diff,
                    "annotator_a": {f: ra[f] for f in ("category", "priority", "language", "requires_human", "annotator_id")},
                    "annotator_b": {f: rb[f] for f in ("category", "priority", "language", "requires_human", "annotator_id")},
                    "resolved": resolved,
                }
            )
            if resolved:
                adjudicated_records.append(adj)
        else:
            adjudicated_records.append(
                {
                    "slot_id": sid,
                    "partition": ra.get("partition", "sealed_final_eval"),
                    "text": ra.get("text"),
                    "category": ra["category"],
                    "priority": ra["priority"],
                    "language": ra["language"],
                    "requires_human": ra["requires_human"],
                    "adjudication_source": "unanimous_dual_blind",
                }
            )

    kappa_metrics = {
        "category": compute_cohens_kappa(cat_a, cat_b, possible_labels=CATEGORIES),
        "priority_unweighted": compute_cohens_kappa(prio_a, prio_b, possible_labels=PRIORITIES),
        "priority_linear_weighted": compute_cohens_kappa(prio_a, prio_b, possible_labels=PRIORITIES, linear_weights=PRIORITY_ORDINAL),
        "language": compute_cohens_kappa(lang_a, lang_b, possible_labels=sorted(valid_langs)),
        "requires_human": compute_cohens_kappa(hum_a, hum_b, possible_labels=[False, True]),
    }

    sealed_final = [r for r in adjudicated_records if r.get("partition", "sealed_final_eval") == "sealed_final_eval"]
    dev_reserved = [r for r in adjudicated_records if r.get("partition") == "dev_calibration_reserved"]

    cat_support = dict(Counter(r["category"] for r in sealed_final))
    prio_support = dict(Counter(r["priority"] for r in sealed_final))
    lang_support = dict(Counter(r["language"] for r in sealed_final))

    unresolved_count = sum(1 for d in disagreements if not d["resolved"])
    gates_passed = (
        len(sealed_final) >= 300
        and len(exact_dups) == 0
        and len(near_dups) == 0
        and len(leakage_hits) == 0
        and len(provenance_errors) == 0
        and unresolved_count == 0
        and all(cat_support.get(c, 0) > 0 for c in CATEGORIES)
        and all(prio_support.get(p, 0) > 0 for p in PRIORITIES)
    )

    return {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "guideline_version": GUIDELINE_VERSION,
        "status": "COMPLETE_SEALED" if gates_passed else "PENDING",
        "counts": {
            "total_intake_slots": len(intake),
            "authored_scenarios": len(authored),
            "paired_blind_annotations": len(paired_ids),
            "adjudicated_total": len(adjudicated_records),
            "sealed_final_eval_ready": len(sealed_final),
            "dev_calibration_ready": len(dev_reserved),
            "target_minimum_sealed": 300,
        },
        "data_quality": {
            "exact_duplicate_pairs": exact_dups,
            "near_duplicate_pairs": near_dups,
            "cross_dataset_leakage_hits": leakage_hits,
            "provenance_errors": provenance_errors,
            "disagreements_total": len(disagreements),
            "disagreements_unresolved": unresolved_count,
        },
        "inter_annotator_agreement": kappa_metrics,
        "support_counts_sealed_final": {
            "category": cat_support,
            "priority": prio_support,
            "language": lang_support,
        },
        "disagreements_queue": disagreements,
        "blocking_reasons": issues + (
            [f"Sealed ready count {len(sealed_final)} < required 300"] if len(sealed_final) < 300 else []
        ) + (
            [f"{unresolved_count} unresolved disagreements"] if unresolved_count > 0 else []
        ) + provenance_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="NEXUS-SLM Human Eval V2 Governance Manager")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-scaffold", help="Initialize data/human_eval_v2/ intake & governance files")
    p_init.add_argument("--dir", type=Path, default=PROJECT_ROOT / "data/human_eval_v2")

    p_blind = sub.add_parser("prepare-blind-packets", help="Export anonymized blind A/B annotation packets")
    p_blind.add_argument("--intake", type=Path, default=PROJECT_ROOT / "data/human_eval_v2/intake_collection_template.jsonl")
    p_blind.add_argument("--out-a", type=Path, default=PROJECT_ROOT / "data/human_eval_v2/blind_annotator_a_packet.jsonl")
    p_blind.add_argument("--out-b", type=Path, default=PROJECT_ROOT / "data/human_eval_v2/blind_annotator_b_packet.jsonl")

    p_val = sub.add_parser("validate", help="Validate provenance, Kappa, duplicates, leakage & readiness")
    p_val.add_argument("--dir", type=Path, default=PROJECT_ROOT / "data/human_eval_v2")
    p_val.add_argument("--out-report", type=Path, default=PROJECT_ROOT / "experiments/artifacts/human_eval_v2_validation_status.json")

    args = parser.parse_args()
    if args.command == "init-scaffold":
        res = init_scaffold(args.dir)
        print(json.dumps(res, indent=2))
    elif args.command == "prepare-blind-packets":
        res = prepare_blind_packets(args.intake, args.out_a, args.out_b)
        print(json.dumps(res, indent=2))
    elif args.command == "validate":
        res = validate_and_adjudicate(
            intake_path=args.dir / "intake_collection_template.jsonl",
            ann_a_path=args.dir / "blind_annotator_a_packet.jsonl",
            ann_b_path=args.dir / "blind_annotator_b_packet.jsonl",
            adjudication_path=args.dir / "adjudication_log_v2.jsonl",
        )
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "disagreements_queue"}, indent=2))


if __name__ == "__main__":
    main()
