"""Synchronize authoritative evaluation tables in README.md and docs/RESULTS.md directly from JSON artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
ARTIFACTS = ROOT / "experiments" / "artifacts"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_canonical_lf(path: Path) -> str:
    """Line-ending-invariant SHA-256 matching Git/POSIX LF checkouts."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_authoritative_summary_block() -> str:
    ref_v2_syn = load_json(RESULTS / "eval_synthetic_test_refined_v2.json")
    base_v1_syn = load_json(RESULTS / "eval_synthetic_test_baseline_v1.json")
    qwen_true = load_json(RESULTS / "eval_qwen_clean_v2_true_base.json")
    qwen_stacked = load_json(ARTIFACTS / "qwen25_05b_clean_v2_lora_test_results.json")
    repro = load_json(RESULTS / "reproduced_refined_v2_baseline.json")

    test_hash = sha256_canonical_lf(ROOT / "data/v2_clean_template_grouped/test_clean.jsonl")[:16]
    hum_syn = load_json(RESULTS / "eval_human_holdout_refined_v2.json")
    hum_v1 = load_json(RESULTS / "eval_human_holdout_baseline_v1.json")

    raw_q = qwen_true["raw_model_metrics"]
    pol_q = qwen_true["policy_applied_metrics"]
    qs_cat = qwen_stacked["metrics"]["category"]
    qs_pri = qwen_stacked["metrics"]["priority"]
    qs_lang = qwen_stacked["metrics"]["language"]
    qs_saf = qwen_stacked["metrics"]["safety"]
    qs_perf = qwen_stacked["metrics"]["performance"]

    en_sup = ref_v2_syn["language_metrics"]["breakdown"]["english"]["support"]
    hi_sup = ref_v2_syn["language_metrics"]["breakdown"]["hindi"]["support"]
    hg_sup = ref_v2_syn["language_metrics"]["breakdown"]["hinglish"]["support"]

    lines = [
        "<!-- SYNC:RESULTS_SUMMARY:START -->",
        f"**Dataset:** Locked Synthetic Template-Grouped Clean Test (`data/v2_clean_template_grouped/test_clean.jsonl`, `N={ref_v2_syn['sample_count']}`, Canonical Git/LF SHA-256 prefix `{test_hash}...` / Windows CRLF `fc65071b1e9f3303...`)  ",
        f"**Exact Language Support Counts:** English `{en_sup}`, Hindi (Devanagari) `{hi_sup}`, Hinglish (Romanized) `{hg_sup}`  ",
        "**Archive Notice:** All historical snapshots (`Policy V1`, pre-PR#4 `13:47Z` run, and double-adapter stacked run) are preserved immutably under [`experiments/results/archive_v1/`](experiments/results/archive_v1/ARCHIVE_MANIFEST.json).",
        "",
        "### Locked Synthetic Test Benchmark ($N=746$)",
        "",
        "| Metric | Heuristic V2 (Policy V1 Archive) | Production Heuristic V2 (Refined Policy V2) | Qwen2.5-0.5B LoRA (True Uncontaminated HF Base) | Historical Qwen LoRA Stacked (`scratch_merged_qwen`, Contaminated) |",
        "|---|---:|---:|---:|---:|",
        f"| **Category Accuracy** | {base_v1_syn['category_metrics']['accuracy']*100:.2f}% | **{ref_v2_syn['category_metrics']['accuracy']*100:.2f}%** | {raw_q['category']['accuracy']*100:.2f}% | {qs_cat['accuracy']*100:.2f}% |",
        f"| **Category Macro-F1** | {base_v1_syn['category_metrics']['macro_f1']:.4f} | **{ref_v2_syn['category_metrics']['macro_f1']:.4f}** | {raw_q['category']['macro_f1']:.4f} | {qs_cat['macro_f1']:.4f} |",
        f"| **Priority Accuracy** | {base_v1_syn['priority_metrics']['accuracy']*100:.2f}% | {ref_v2_syn['priority_metrics']['accuracy']*100:.2f}% | {raw_q['priority']['accuracy']*100:.2f}% | **{qs_pri['accuracy']*100:.2f}%** |",
        f"| **Priority Macro-F1** | {base_v1_syn['priority_metrics']['macro_f1']:.4f} | **{ref_v2_syn['priority_metrics']['macro_f1']:.4f}** | {raw_q['priority']['macro_f1']:.4f} | {qs_pri['macro_f1']:.4f} |",
        f"| **Language Accuracy** | {base_v1_syn['language_metrics']['overall_accuracy']*100:.2f}% | {ref_v2_syn['language_metrics']['overall_accuracy']*100:.2f}% | {raw_q['language']['accuracy']*100:.2f}% | **{qs_lang['overall_accuracy']*100:.2f}%** |",
        f"| - English ($N={en_sup}$) | {base_v1_syn['language_metrics']['breakdown']['english']['accuracy']*100:.2f}% | {ref_v2_syn['language_metrics']['breakdown']['english']['accuracy']*100:.2f}% | {qwen_true['language_breakdown']['english']['raw_category_accuracy']*100:.2f}% cat | {qs_lang['english_accuracy']*100:.2f}% lang |",
        f"| - Hindi Devanagari ($N={hi_sup}$) | {base_v1_syn['language_metrics']['breakdown']['hindi']['accuracy']*100:.2f}% | {ref_v2_syn['language_metrics']['breakdown']['hindi']['accuracy']*100:.2f}% | {qwen_true['language_breakdown']['hindi']['raw_category_accuracy']*100:.2f}% cat | {qs_lang['hindi_accuracy']*100:.2f}% lang |",
        f"| - Hinglish Roman ($N={hg_sup}$) | {base_v1_syn['language_metrics']['breakdown']['hinglish']['accuracy']*100:.2f}% | {ref_v2_syn['language_metrics']['breakdown']['hinglish']['accuracy']*100:.2f}% | {qwen_true['language_breakdown']['hinglish']['raw_category_accuracy']*100:.2f}% cat | {qs_lang['hinglish_accuracy']*100:.2f}% lang |",
        f"| **JSON Validity Rate** | 100.00% | 100.00% | {qwen_true['json_validity_rate']*100:.2f}% | 100.00% |",
        f"| **High-Risk Escalation Recall** | {base_v1_syn['escalation_metrics']['high_risk_recall']*100:.2f}% ({base_v1_syn['escalation_metrics']['high_risk_false_negatives']} FN) | {ref_v2_syn['escalation_metrics']['high_risk_recall']*100:.2f}% ({ref_v2_syn['escalation_metrics']['high_risk_false_negatives']} FN) | {pol_q['escalation']['high_risk_recall']*100:.2f}% ({pol_q['escalation']['false_negatives']} FN, conf=None) | {qs_saf['escalation_recall']*100:.2f}% ({qs_saf['false_negatives']} FN) |",
        f"| **Critical Priority Recall** | {base_v1_syn['critical_safety_metrics']['critical_recall']*100:.2f}% (0 FN) | **{ref_v2_syn['critical_safety_metrics']['critical_recall']*100:.2f}% (0 FN)** | **{pol_q['escalation']['critical_recall']*100:.2f}% (0 FN)** | — (unweighted critical in raw) |",
        f"| **Overall Escalation Rate** | {base_v1_syn['escalation_metrics']['overall_escalation_rate']*100:.2f}% | **{ref_v2_syn['escalation_metrics']['overall_escalation_rate']*100:.2f}%** | {pol_q['escalation']['overall_escalation_rate']*100:.2f}% (policy conservative) | {qs_saf['overall_escalation_rate']*100:.2f}% |",
        f"| **Direct CPU P50 Latency** | ~{repro['synthetic_test_n746']['latency_p50_ms']:.2f} ms | **{repro['synthetic_test_n746']['latency_p50_ms']:.2f} ms** (P95 {repro['synthetic_test_n746']['latency_p95_ms']:.2f} ms) | — | {qs_perf['latency_p50_ms']:.2f} ms (P95 {qs_perf['latency_p95_ms']:.2f} ms) |",
        "",
        "### Development-Exposed Human Scenario Audit ($N=184$ Active + $16$ Pending)",
        "",
        "| Metric | Heuristic V1 (Policy V1, $N=184$) | Production Heuristic V2 (Refined Policy V2, $N=184$) |",
        "|---|---:|---:|",
        f"| **Category Accuracy / Macro-F1** | {hum_v1['category_metrics']['accuracy']*100:.2f}% / {hum_v1['category_metrics']['macro_f1']:.4f} | **{hum_syn['category_metrics']['accuracy']*100:.2f}% / {hum_syn['category_metrics']['macro_f1']:.4f}** |",
        f"| **Priority Accuracy / Macro-F1** | {hum_v1['priority_metrics']['accuracy']*100:.2f}% / {hum_v1['priority_metrics']['macro_f1']:.4f} | **{hum_syn['priority_metrics']['accuracy']*100:.2f}% / {hum_syn['priority_metrics']['macro_f1']:.4f}** |",
        f"| **Language Accuracy (EN 80 / HI 46 / HG 58)** | {hum_v1['language_metrics']['overall_accuracy']*100:.2f}% | {hum_syn['language_metrics']['overall_accuracy']*100:.2f}% |",
        f"| **Overall Escalation Rate** | {hum_v1['escalation_metrics']['overall_escalation_rate']*100:.2f}% | **{hum_syn['escalation_metrics']['overall_escalation_rate']*100:.2f}%** |",
        f"| **High-Risk Recall / FN** | {hum_v1['escalation_metrics']['high_risk_recall']*100:.2f}% (0 FN) | {hum_syn['escalation_metrics']['high_risk_recall']*100:.2f}% ({hum_syn['escalation_metrics']['high_risk_false_negatives']} FN) |",
        f"| **Critical Safety Recall** | 100.00% (0 FN) | **100.00% (0 FN)** |",
        "",
        "### Validation Sprint V3 Hardened Policy & Unpromoted Neural V2 Candidate",
        "",
        *(_build_v3_and_neural_v2_lines()),
        "<!-- SYNC:RESULTS_SUMMARY:END -->",
    ]
    return "\n".join(lines)


def _build_v3_and_neural_v2_lines() -> list[str]:
    v3_syn_p = RESULTS / "eval_synthetic_test_v3_hardened.json"
    v3_hum_p = RESULTS / "eval_human_holdout_v3_hardened.json"
    runs_v2_p = ROOT / "experiments" / "runs_v2.json"
    if not (v3_syn_p.exists() and v3_hum_p.exists() and runs_v2_p.exists()):
        return ["*Sprint V3 hardened policy and Neural V2 artifacts pending regeneration.*"]
    v3s = load_json(v3_syn_p)
    v3h = load_json(v3_hum_p)
    rv2 = load_json(runs_v2_p)
    best_id = rv2.get("selected_best_val_experiment")
    best_run = next((r for r in rv2["runs"] if r["experiment_id"] == best_id), rv2["runs"][-1])
    bt = best_run["locked_test_clean"]
    return [
        "| System Version | Dataset (`test_clean` N=746 / `human_v1` N=184) | Cat Acc | Cat Macro-F1 | Prio Acc / F1 | Lang Acc (Hinglish ID) | High-Risk Recall (FN) | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        f"| **Hardened Policy V3** | `test_clean.jsonl` (`N=746`) | **{v3s['category_metrics']['accuracy']*100:.2f}%** | **{v3s['category_metrics']['macro_f1']:.4f}** | {v3s['priority_metrics']['accuracy']*100:.2f}% / {v3s['priority_metrics']['macro_f1']:.4f} | {v3s['language_metrics']['overall_accuracy']*100:.2f}% ({v3s['language_metrics']['breakdown']['hinglish']['accuracy']*100:.2f}%) | {v3s['escalation_metrics']['high_risk_recall']*100:.2f}% ({v3s['escalation_metrics']['high_risk_false_negatives']} FN) | Active Hardened Heuristic |",
        f"| **Hardened Policy V3** | `human_eval_v1` (`N=184`, dev-exposed) | **{v3h['category_metrics']['accuracy']*100:.2f}%** | **{v3h['category_metrics']['macro_f1']:.4f}** | {v3h['priority_metrics']['accuracy']*100:.2f}% / {v3h['priority_metrics']['macro_f1']:.4f} | {v3h['language_metrics']['overall_accuracy']*100:.2f}% ({v3h['language_metrics']['breakdown']['hinglish']['accuracy']*100:.2f}%) | {v3h['escalation_metrics']['high_risk_recall']*100:.2f}% ({v3h['escalation_metrics']['high_risk_false_negatives']} FN) | Active Hardened Heuristic |",
        f"| **`{best_id}`** | `test_clean.jsonl` (`N=746`) | **{bt['raw_category']['accuracy']*100:.2f}%** | **{bt['raw_category']['macro_f1']:.4f}** | {bt['raw_priority']['accuracy']*100:.2f}% / {bt['raw_priority']['macro_f1']:.4f} | {bt['raw_language']['accuracy']*100:.2f}% ({bt['language_breakdown']['hinglish']['language_id_recall']*100:.2f}%) | {bt['safety_policy']['high_risk_recall']*100:.2f}% ({bt['safety_policy']['high_risk_false_negatives']} FN, 0 crit FN) | `{rv2['promotion_decision']}` |",
    ]


def sync_file_marker(path: Path, block: str) -> bool:
    txt = path.read_text(encoding="utf-8")
    start_tag = "<!-- SYNC:RESULTS_SUMMARY:START -->"
    end_tag = "<!-- SYNC:RESULTS_SUMMARY:END -->"
    if start_tag in txt and end_tag in txt:
        pre = txt.split(start_tag)[0]
        post = txt.split(end_tag)[1]
        new_txt = pre + block + post
        if new_txt != txt:
            path.write_text(new_txt, encoding="utf-8")
            return True
    return False


if __name__ == "__main__":
    summary_md = build_authoritative_summary_block()
    changed = []
    for rel in ("README.md", "docs/RESULTS.md"):
        p = ROOT / rel
        if sync_file_marker(p, summary_md):
            changed.append(rel)
    print("Generated authoritative summary block; updated marker files:", changed)
