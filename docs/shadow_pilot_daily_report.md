# NEXUS-SLM — Shadow Pilot Daily Operations Log

**Document ID**: `PILOT-DAILY-LOG-2026`  
**Candidate Version**: `nexus-qlora-v1`  
**Base Architecture**: `HuggingFaceTB/SmolLM-135M-Instruct`  
**Git Commit**: `2696df8be85e75b69f8a04345265c7f29d931e02`  
**Governing Protocol**: [experiments/artifacts/pilot_evaluation_protocol.json](file:///c:/Users/avira/OneDrive/Desktop/NEXUS-SLM/experiments/artifacts/pilot_evaluation_protocol.json)  
**Metrics Tracking**: [experiments/artifacts/pilot_metrics.json](file:///c:/Users/avira/OneDrive/Desktop/NEXUS-SLM/experiments/artifacts/pilot_metrics.json)  

---

## Pre-Flight Status Notice (Day 0 — Standing State)

> [!IMPORTANT]
> **Live Shadow Mirror Status: ACTIVATION BLOCKED**  
> Operational activation is blocked pending execution of the Data Protection Agreement and CRM credentials (BLK-01), cloud cluster provisioning (BLK-02), and human reviewer panel rostering (BLK-03).  
> In accordance with strict data integrity and governance rules, **the 14-day observation clock has NOT started**, and no live production customer tickets have been ingested. Pre-pilot synthetic validation numbers are strictly quarantined from prospective pilot counters.

---

## Daily Operational Ledger

### Day 0 — Pre-Activation Infrastructure & Synthetic Baseline (2026-10-11)

| Metric Category | Metric Item | Value / Status | Verification Method |
|:---|:---|:---:|:---|
| **Pilot Lifecycle** | Observation Days Elapsed | **0 / 14 days** | Pre-activation hold |
| | Live Production Tickets Ingested | **0 / 5,000** | Live CRM mirror disabled |
| | Live Shadow Inferences Completed | **0** | No live traffic |
| | Prospective Human Labels Collected | **0** | Awaiting reviewer roster |
| **Synthetic Verification** | Synthetic Smoke Test Inferences | **6 / 6 passed** | `scripts/deployment/run_final_e2e_validation.py` |
| | Synthetic Shadow Path Tests | **1 / 1 passed** | `scripts/deployment/smoke_test_shadow_path.py` |
| | Synthetic Timeouts / Crashes | **0** | Simulated timeout isolated gracefully |
| | Synthetic Backpressure Drops | **0** | Queue overflow drop verified in unit tests |
| | Synthetic Duplicates Dropped | **0** | Deduplication cache verified in unit tests |
| **Model Performance (Canonical)** | Category Accuracy (N=42) | **0.4286** (18/42) | `scripts/evaluation/run_canonical_evaluation.py` |
| | Macro-F1 (14 classes) | **0.4017** | `scripts/evaluation/run_canonical_evaluation.py` |
| | Weighted-F1 (14 classes) | **0.4017** | `scripts/evaluation/run_canonical_evaluation.py` |
| | High-Risk Escalation Recall | **100.0%** (60/60) | `scripts/evaluation/run_escalation_safety_evaluation.py` |
| | Sev-1 Safety Escapes | **0** | Locked test & challenge sets |
| **Inference Latency Profile** | P50 Inference Latency | **1,460.5 ms** | Neural CPU inference (excluding queue wait) |
| | P95 Inference Latency | **1,672.4 ms** | Neural CPU inference (excluding queue wait) |
| | P99 Inference Latency | **2,056.6 ms** | Concurrency benchmark baseline |
| | Roundtrip Overhead (FastAPI) | **~5.5 ms - 8.2 ms** | Pydantic schema validation + ChatML formatting |
| **System Resources (Local)** | Process Base RSS | **205.9 MB** | `psutil.Process().memory_info().rss` |
| | Process Post-Warmup RSS | **981.2 MB** | PyTorch model + weights loaded |
| | Process Peak RSS Under Load | **1,210.7 MB** | Concurrency stress test ($C=4$) |
| | Container Allocation Recommendation | **2 vCPUs / 4,096 MB** | Safety Factor = 3.38x above peak process RSS |
| **Data Drift & Health** | Population Stability Index (PSI) | **Baseline (0.00)** | Pre-pilot distribution |
| | Drift Alert Status | **NORMAL** | No live drift measurable |
| | Rollback Mechanism RTO | **< 30 seconds** | Verified via `/model/rollback` hot-swap |
| **Operational Incidents** | Sev-1 Incidents Recorded | **0** | None |
| | Sev-2 Operational Incidents | **0** | None |
| | Sev-3 Triage Incidents | **0** | None |

---

## Log Template for Active Shadow Days (Days 1–14)

Upon formal resolution of BLK-01, BLK-02, and BLK-03, each observation day will record the following schema:

```markdown
### Day X — [Date]
- **Ingestion**: [Total Mirror Tickets] | **Completed**: [Shadow Inferences] | **Errors**: [E] | **Timeouts**: [T] | **Drops**: [D]
- **Latency**: P50: [X ms] | P95: [Y ms] | P99: [Z ms] (Model Execution Scope)
- **Memory**: Process RSS: [MB] | Container Memory: [MB] | CPU Utilization: [%]
- **Double-Blind Labels**: [N Collected] | Concordance Rate: [X%] (Wilson 95% CI: [L%, U%])
- **Escalation Safety**: High-Risk Evaluated: [N] | Safety Recall: [X%] | Sev-1 Escapes: [0]
- **Drift**: Category PSI: [X] | Language PSI: [Y] | Status: [STABLE / WARNING / DRIFT_ALERT]
- **Incidents & Rollbacks**: [Sev-1 / Sev-2 / Sev-3 records or None]
```
