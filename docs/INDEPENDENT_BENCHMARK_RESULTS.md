# NEXUS-SLM Independent Benchmark Results (Human Eval V2)

> **BENCHMARK STATUS: `PENDING`** — Ground-truth scenarios, dual-blind independent human annotations, and third-party adjudications are **not yet executed** because independent external human annotators and legal/DPA-approved live intake sources are unavailable within this automated sprint. No synthetic annotations or simulated reviewers have been fabricated.

---

## 1. Current Validation Snapshot

From [`experiments/artifacts/human_eval_v2_validation_status.json`](file:///c:/Users/avira/Desktop/NEXUS-SLM/experiments/artifacts/human_eval_v2_validation_status.json):

| Metric / Gate | Current Measured Value | Required Gate | Status |
|---|---|---|---|
| **Sealed Independent Scenarios Ready (`sealed_final_eval`)** | `0 / 320` slots authored | `>= 300` | `PENDING` |
| **Development Calibration Partition (`dev_calibration_reserved`)** | `0 / 80` slots authored | Separate from `sealed_final_eval` | `PENDING` |
| **Paired Dual-Blind Annotations** | `0` | 100% paired (`A != B`) | `PENDING` |
| **Cohen's Kappa — Category ($\kappa$)** | `N/A (INSUFFICIENT_DATA)` | Compute over N >= 300 blind pairs | `PENDING` |
| **Cohen's Kappa — Priority (unweighted / weighted $\kappa_w$)** | `N/A / N/A` | Compute over N >= 300 blind pairs | `PENDING` |
| **Cohen's Kappa — Language ($\kappa$)** | `N/A` | Compute over N >= 300 blind pairs | `PENDING` |
| **Cohen's Kappa — Escalation (`requires_human`, $\kappa$)** | `N/A` | Compute over N >= 300 blind pairs | `PENDING` |
| **Unresolved Disagreements** | `0` (queue empty) | `0` after adjudication | `PENDING` |
| **Exact / Near-Duplicate Pairs** | `0 / 0` | `0` | Infrastructure Verified |
| **Overlap with V1 184 Dev-Exposed Scenarios or Clean V2 Splits** | `0` | `0` | Infrastructure Verified |

---

## 2. Historical 184-Scenario Note (Do Not Double-Count)

The existing **184 adjudicated scenarios** in [`data/human_eval/annotation_template.jsonl`](file:///c:/Users/avira/Desktop/NEXUS-SLM/data/human_eval/annotation_template.jsonl) (and 16 pending live-intake slots) were inspected and used during Refined Policy V2 calibration (`eval_human_holdout_refined_v2.json`). Consequently:

- They are classified strictly as **development-exposed** (`N=184`), not an independent V2 sealed benchmark.
- [`scripts/data/human_eval_v2_manager.py`](file:///c:/Users/avira/Desktop/NEXUS-SLM/scripts/data/human_eval_v2_manager.py) enforces a hard rejection if any V1 `human_eval` or synthetic V1/V2 text enters `data/human_eval_v2/`.

---

## 3. Exact Prerequisites to Complete This Benchmark

To transition [`docs/INDEPENDENT_BENCHMARK_RESULTS.md`](file:///c:/Users/avira/Desktop/NEXUS-SLM/docs/INDEPENDENT_BENCHMARK_RESULTS.md) from `PENDING` to `COMPLETE_SEALED`:

1. **Independent Authoring:** Populate `data/human_eval_v2/intake_collection_template.jsonl` (minimum 300 `sealed_final_eval` + 80 `dev_calibration_reserved` texts) via human domain authors or consented PII-scrubbed customer logs meeting [`data/human_eval_v2/consent_and_privacy_checklist.json`](file:///c:/Users/avira/Desktop/NEXUS-SLM/data/human_eval_v2/consent_and_privacy_checklist.json).
2. **Generate Blind Packets:** Execute `scripts/data/human_eval_v2_manager.py prepare-blind-packets` so Annotator A and Annotator B receive stripped records without category/priority hints or model outputs.
3. **Blind Dual Annotation:** Have two distinct human annotators label `blind_annotator_a_packet.jsonl` and `blind_annotator_b_packet.jsonl` per `annotation_guidelines_v2.0.0.json` with valid `annotator_id` and `annotated_at_utc` fields.
4. **Adjudicate Disagreements:** Resolve all items flagged in `disagreements_queue` with a third independent adjudicator (`adjudicator_id`) in `adjudication_log_v2.jsonl`.
5. **Validate & Run Sealed Inference:** Re-run `scripts/data/human_eval_v2_manager.py validate` until `status == "COMPLETE_SEALED"`, record SHA-256 checksums, and run a single frozen evaluation pass across the production heuristic engine and neural candidate without further threshold or rule edits.
