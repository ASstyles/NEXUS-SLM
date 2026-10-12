# NEXUS-SLM — Unified Activation Blocker Tracker

**Document ID**: `ACTIVATION-TRACKER-2026-V2`  
**Active Production Model**: `nexus-qlora-v1` (Preserved Active — 100% Customer Traffic)  
**Evaluated Pilot Candidate**: `qwen25_05b_lora_seed42_q8_0` (GGUF Weight-Only Q8_0)  
**Candidate Artifact Path**: `models/qwen25_05b_lora_seed42_gguf_q8_0/qwen25_05b_lora_seed42_q8_0.gguf`  
**Candidate Artifact SHA-256**: `b0c891dca80498f95e97d37ace60b574860fbfc3100574aeb76bae524e79f6d8`  
**Candidate Artifact Size**: 531,067,808 bytes (506.47 MB)  
**Git Branch**: `main`  
**Audit Date**: October 12, 2026  
**Lead Auditor**: Lead Safety & Infrastructure Release Auditor  
**Candidate Evaluation Status**: `ALL 6 ACCEPTANCE GATES PASSED (APPROVED FOR STAGING)`  
**Overall Operational Blocker Status**: `BLOCKED BY MISSING EVIDENCE — ACTIVATION REQUIREMENTS UNMET`  

---

## 1. Candidate Technical Validation Summary

The candidate `qwen25_05b_lora_seed42_q8_0` underwent complete independent evaluation against locked datasets and the full test suite:

| Acceptance Gate | Condition | Measured Value | Threshold | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Gate 1 — Validation Macro-F1** | Unweighted harmonic mean ($N=44$) | **0.8974** | $\ge 0.5500$ | **PASS** |
| **Gate 2 — Devanagari Hindi Acc** | Native Hindi subword accuracy | **0.9286 (13/14)** | $\ge 0.5000$ | **PASS** |
| **Gate 3 — Safety Recall** | High-risk challenge set ($N=70$) | **1.0000 (0 FN, 60 TP, 10 TN)** | $= 1.0000$ | **PASS** |
| **Gate 4 — Schema Conformity** | JSON validity via OutputParser | **100.0% (44/44)** | $\ge 95.0\%$ | **PASS** |
| **Gate 5 — System Resources** | CPU P95 Latency & Peak RAM | **P95: 1,374.8 ms, RAM: 1,479.4 MB** | $\le 2500$ ms & $\le 2048$ MB | **PASS** |
| **Gate 6 — Regression Test Suite** | Full repo pytest execution | **125 passed / 125 collected (100%)** | 0 failures | **PASS** |

*Note on Model Derivation:* The candidate weights were mathematically verified against the PEFT LoRA adapter matrices ($\max |\Delta W| = 5.88 \times 10^{-8}$ vs base `Qwen/Qwen2.5-0.5B-Instruct`), certifying that fine-tuned LoRA weights are incorporated.

---

## 2. Unified Blocker Summary Matrix

| Blocker ID | Domain Description | Responsible Owner Role | Required Evidence for Closure | Current Verification Finding | Current Status |
|:---:|:---|:---|:---|:---|:---:|
| **BLK-01** | **Data Access & CRM Authorization** | Data Protection Officer (DPO) & Support CRM Systems Lead | 1. Executed DPA Addendum signed by DPO.<br>2. Read-only CRM webhook signing secret injected into vault (`CRM_WEBHOOK_SECRET` length $\ge 32$).<br>3. Webhook ping handshake. | `os.getenv('CRM_WEBHOOK_SECRET')` is `None`. No executed DPA document found on disk or vault. Webhook listener is offline. | **FAIL / BLOCKED BY MISSING EVIDENCE** |
| **BLK-02** | **Production Cloud Infrastructure** | Site Reliability Engineer (SRE) / Cloud Platform Lead | 1. Containerized service deployed to internal cluster with 2-vCPU / 4-GB limits.<br>2. Remote curl probe returning HTTP 200 from `https://nexus-slm-shadow.internal/health`.<br>3. Prometheus metrics endpoint live. | Socket DNS lookup for `nexus-slm-shadow.internal` fails (`getaddrinfo failed`). Production cluster host unprovisioned. | **FAIL / BLOCKED BY MISSING EVIDENCE** |
| **BLK-03** | **Human Review Workforce Panel** | Support Operations Lead & Quality Assurance Manager | 1. Signed roster of 5+ qualified support specialists with weekly shift schedules.<br>2. Verified labeling capacity ($\ge 50$ tickets/specialist/week).<br>3. Prospective sealed ground-truth cohort manifest `data/pilot/ground_truth_freeze_manifest.json`. | No reviewer roster file on disk. `data/pilot/ground_truth_freeze_manifest.json` does not exist (`exists() == False`). | **FAIL / BLOCKED BY MISSING EVIDENCE** |
| **BLK-04** | **Remote Linux Docker CI** | CI/CD Release Engineer & Platform Architect | Passing GitHub Actions run testing build, health, live inference, and shutdown on Linux container. | GitHub Actions Run ID 37972082549 completed with `success` across all 3 workflow jobs on `ubuntu-latest`. | **PASS / RESOLVED** |

---

## 3. Detailed Blocker Status & Evidence Ledgers

### BLK-01 — Data Protection & CRM Authorization
- **Designated Owners**: Data Protection Officer (DPO) & CRM Systems Lead
- **Governing Package**: `docs/operations/data_access_readiness.md`
- **Current Findings**:
  - `PrivacyRedactor` mechanisms and secret stripping algorithms pass 100% of unit tests.
  - However, no legally executed Data Protection Agreement (DPA) Addendum exists.
  - No CRM webhook credentials exist in the environment (`os.getenv('CRM_WEBHOOK_SECRET')` is `None`).
  - Webhook listener is offline.
- **Required Evidence for Closure**:
  1. DPO executes and deposits the formal DPA Addendum.
  2. CRM Systems Lead provisions the read-only webhook mirror and injects `CRM_WEBHOOK_SECRET` into the production environment.
  3. Passing handshake verification command:
     ```bash
     .\.venv\Scripts\python.exe -c "import os; assert os.getenv('CRM_WEBHOOK_SECRET') and len(os.getenv('CRM_WEBHOOK_SECRET')) >= 32; print('BLK-01 Verified: Secret present.')"
     ```
- **Current Gate Determination**: **FAIL / BLOCKED BY MISSING EVIDENCE**

---

### BLK-02 — Production Cloud Infrastructure
- **Designated Owners**: Site Reliability Engineer (SRE) / Cloud Platform Lead
- **Governing Runbook**: `docs/operations/shadow_infrastructure_runbook.md`
- **Current Findings**:
  - Local engine is healthy, quantized GGUF weights load in 0.66s, and local test suite passes 100%.
  - However, DNS resolution for `nexus-slm-shadow.internal` fails (`socket.gethostbyname` raises `gaierror`).
  - The designated internal VPC cluster container service is unprovisioned.
- **Required Evidence for Closure**:
  1. SRE provisions the container service on the internal VPC cluster with 2-vCPU / 4-GB limits.
  2. SRE wires Prometheus metric scraping for telemetry counters.
  3. Remote probe returns HTTP 200:
     ```bash
     curl -s -f https://nexus-slm-shadow.internal/health
     ```
- **Current Gate Determination**: **FAIL / BLOCKED BY MISSING EVIDENCE**

---

### BLK-03 — Human Review Workforce Panel
- **Designated Owners**: Support Operations Lead & Quality Assurance Manager
- **Governing Package**: `docs/operations/reviewer_onboarding_and_blind_labeling.md`
- **Current Findings**:
  - Double-blind evaluation protocol software is fully implemented in `src/pilot/evaluation_protocol.py` and passes all unit tests.
  - However, no human review panel has been rostered or onboarded.
  - Shift availability of 5+ specialists has not been confirmed.
  - Zero prospective ground-truth labels have been collected or frozen (`data/pilot/ground_truth_freeze_manifest.json` does not exist).
- **Required Evidence for Closure**:
  1. Support Operations Lead rosters 5+ qualified support specialists across shift rotations.
  2. Confirm weekly evaluation capacity ($\ge 50$ tickets/specialist).
  3. Execute double-blind labeling on initial pilot cohort and seal manifest via `freeze_pilot_ground_truth()`:
     ```bash
     .\.venv\Scripts\python.exe -c "from pathlib import Path; assert Path('data/pilot/ground_truth_freeze_manifest.json').exists(), 'Freeze manifest missing'; print('BLK-03 Verified: Human ground truth frozen.')"
     ```
- **Current Gate Determination**: **FAIL / BLOCKED BY MISSING EVIDENCE**

---

### BLK-04 — Remote Linux Docker CI
- **Designated Owners**: CI/CD Release Engineer & Platform Architect
- **Audited Commit SHA**: `2696df8be85e75b69f8a04345265c7f29d931e02`
- **Current Findings**:
  - GitHub Actions API confirmed **Run ID 37972082549** executed on `ubuntu-latest`.
  - Workflow URL: `https://github.com/ASstyles/NEXUS-SLM/actions/runs/37972082549`
  - Created: `2026-10-09T18:16:06Z` | Completed: `2026-10-09T18:28:43Z` | Conclusion: **`success`**.
  - All three jobs completed with `success`:
    1. `Lint, Static Checks, and Automated Test Suite (3.10)`: 94/94 tests passed.
    2. `Container Build & Health Validation`: Runner disk cleanup, Docker buildx build, container execution, healthcheck pass, live inference pass, and graceful shutdown pass.
    3. `Build Next.js Dashboard`: Clean Next.js compilation on Node 20.
- **Current Gate Determination**: **PASS / RESOLVED**

---

## 4. Master Activation Decision

### Current Formal Status: `BLOCKED BY MISSING EVIDENCE — ACTIVATION REQUIREMENTS UNMET`

**Mandatory Governance Invariants**:
1. `nexus-qlora-v1` remains the sole active production model handling 100% of live customer ticket routing.
2. Candidate `qwen25_05b_lora_seed42_q8_0` remains registered as `experimental`.
3. Do not enable live ticket ingestion, passive shadow mirror ingress, or customer traffic routing to the candidate.
4. Do not advance the 14-day shadow observation clock until BLK-01, BLK-02, and BLK-03 are formally marked **PASS** with verifiable documented evidence deposited by their designated owners.
