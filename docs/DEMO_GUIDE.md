# NEXUS-SLM — Interactive Demonstration Guide

> **Environment Notice:**  
> This demonstration executes entirely offline within a local sandbox environment. It utilizes synthetic test fixtures exclusively. **Zero real customer data or production CRM connections are accessed or simulated.** The production deployment status remains `BLOCKED — ACTIVATION REQUIREMENTS UNMET`.

---

## 1. System Overview & Architecture

NEXUS-SLM (*Neural EXecution for Unified Support Intelligence*) is a specialized Small Language Model (SLM) service fine-tuned on customer support ticket triage. It routes tickets into a 14-class taxonomy across three language modalities (English, Hinglish, and Hindi), assigns urgency priority levels, and applies deterministic safety policies to govern human escalation.

### Core Demo Capabilities
1. **Interactive Triage:** Enter support tickets or select from 30 curated multi-lingual synthetic fixtures across 10 categories.
2. **Deterministic Safety Policy:** Automatic policy-driven human escalation for safety-critical tickets, regardless of model confidence.
3. **Native Devanagari Hindi Protection:** Automatic escalation of native-script Hindi tickets due to Byte-Pair Encoding (BPE) subword fragmentation risks.
4. **Uncalibrated Heuristic Observability:** Transparent labeling of raw softmax confidence scores as uncalibrated heuristics.
5. **Zero-Downtime Hot-Swap Rollback:** Instant switching between model versions (`nexus-qlora-v1`, `baseline-smollm-135m`, `nexus-qlora-v2`) via live API.
6. **Session History:** In-memory tracking of all triaged requests with one-click reload capabilities.

---

## 2. Quick Start Launch Instructions

### Prerequisites
- **Python:** 3.10+ (with `.venv` containing project dependencies)
- **Node.js:** 18+ or 20+ (with npm dependencies in `dashboard/`)
- **Port Availability:** `8000` (FastAPI backend) and `3000` (Next.js dashboard)

### Option A — Automated Launchers

#### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/launch_demo.ps1
```

#### Unix / Linux / macOS (Bash)
```bash
chmod +x scripts/launch_demo.sh
./scripts/launch_demo.sh
```

---

### Option B — Manual Step-by-Step Launch

#### Step 1: Start the FastAPI Backend
Open Terminal 1:
```powershell
# Windows
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Unix / Linux / macOS
source .venv/bin/activate
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Verify readiness:
```bash
curl http://127.0.0.1:8000/health
```
Expected output:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "nexus-qlora-v1",
  "inference_mode": "neural"
}
```

#### Step 2: Start the Next.js Dashboard
Open Terminal 2:
```bash
cd dashboard
npm run dev
```
Access the dashboard at [http://localhost:3000](http://localhost:3000).

---

## 3. Demo Walkthrough Scenarios

### Scenario 1: Multi-Lingual Payment Failure (English, Hinglish, Hindi)
Triage a payment deduction failure across three linguistic modalities:

1. **English:**  
   - Select or paste: `"My payment of $49.99 was deducted from my account but the order status is still pending."`
   - **Result:** Category: `payment_failure`, Priority: `high`, Language: `english`.
   - **Escalation:** `REQUIRES HUMAN` (`financial_dispute_keyword_match`).

2. **Hinglish:**  
   - Select or paste: `"Mera payment kat gaya bank se par app me order confirm nahi hua."`
   - **Result:** Category: `payment_failure`, Priority: `high`, Language: `hinglish`.
   - **Escalation:** `REQUIRES HUMAN` (`financial_dispute_keyword_match`).

3. **Devanagari Hindi:**  
   - Select or paste: `"मेरे खाते से पैसे कट गए हैं लेकिन ऑर्डर अभी तक कन्फर्म नहीं हुआ है।"`
   - **Result:** Category: `payment_failure`, Priority: `high`, Language: `hindi`.
   - **Escalation:** `REQUIRES HUMAN` (`native_hindi_devanagari_unreliable_tokenization`).

---

### Scenario 2: Mandatory Devanagari Hindi Safety Policy
SmolLM-135M exhibits severe subword token fragmentation on native Devanagari Hindi characters (averaging 3–5 byte tokens per character), degrading empirical accuracy to ~14.3%. NEXUS-SLM enforces a deterministic safety rule routing **100% of native Devanagari Hindi tickets to human agents**.

1. Select **Hindi Password Reset**:  
   `"मैं अपना पासवर्ड भूल गया हूँ, कृपया पासवर्ड रीसेट करने का लिंक भेजें।"`
2. Click **ANALYZE TICKET**.
3. **Verify:**
   - Even though password resets are normally auto-resolved, the system flags `REQUIRES HUMAN`.
   - Policy trigger: `native_hindi_devanagari_unreliable_tokenization`.
   - The explanatory card displays: *Policy safeguard: Native Devanagari Hindi text detected. Deterministic safety rules require human review.*

---

### Scenario 3: Safety-Critical Fraud Detection
Fraud reports must bypass standard confidence thresholds and route immediately to human risk specialists.

1. Select **EN Fraud Urgent $450**:  
   `"Urgent: Unauthorized charges of $450 detected on my credit card without my OTP or permission!"`
2. Click **ANALYZE TICKET**.
3. **Verify:**
   - Category forced to: `fraud_report`.
   - Priority forced to: `critical`.
   - Human escalation: `REQUIRES HUMAN` (`safety_critical_fraud_detection`).

---

### Scenario 4: Observability of Uncalibrated Heuristic Confidence
1. Observe the **Model Confidence** gauge on any triaged ticket.
2. Note the amber warning banner:
   > **Uncalibrated Heuristic Notice:** Softmax output probability from language head. Does not reflect calibrated Bayesian posterior. Model confidence cannot bypass deterministic human escalation rules.
3. Verify `confidence_status: uncalibrated_heuristic` in the validated JSON payload.

---

### Scenario 5: Zero-Downtime Model Rollback
Demonstrates hot-swapping active models without restarting services:

1. Locate the **Model Version & Zero-Downtime Rollback Demonstration** card at the top of the Analyze tab.
2. Click **baseline-smollm-135m (Base Fallback)**.
3. Observe instant notification:  
   *Model successfully switched to "baseline-smollm-135m" (heuristic_fallback mode). Zero-downtime rollback executed.*
4. Triage a ticket and observe `model_version: baseline-smollm-135m` in the output and telemetry header.
5. Click **nexus-qlora-v1 (Production LoRA)** to restore the active production candidate.
6. Verify live health reflects `nexus-qlora-v1`.

---

### Scenario 6: Session Prediction History
1. Triage multiple tickets using preset buttons.
2. Scroll to the **Synthetic Session Prediction History** table.
3. Inspect logged timestamps, category, priority, language, escalation status, reason, and latency.
4. Click **Load** on any historical row to repopulate the input field and review the output.
5. Click **Clear History** to reset the synthetic session cache.

---

## 4. Ten-Category Synthetic Fixtures Catalog Reference

All 30 documented synthetic demo fixtures are located in [`data/demo_fixtures.json`](file:///c:/Users/avira/OneDrive/Desktop/NEXUS-SLM/data/demo_fixtures.json):

| Category | English Preset | Hinglish Preset | Hindi Devanagari Preset | Escalation Rule |
| :--- | :--- | :--- | :--- | :--- |
| **payment_failure** | `demo_pay_en` | `demo_pay_hinglish` | `demo_pay_hindi` | Financial dispute / Hindi tokenization |
| **refund_request** | `demo_ref_en` | `demo_ref_hinglish` | `demo_ref_hindi` | Automated for EN/Hinglish; Hindi escalated |
| **account_access** | `demo_acc_en` | `demo_acc_hinglish` | `demo_acc_hindi` | Account security lockout |
| **password_reset** | `demo_pwd_en` | `demo_pwd_hinglish` | `demo_pwd_hindi` | Automated for EN/Hinglish; Hindi escalated |
| **subscription** | `demo_sub_en` | `demo_sub_hinglish` | `demo_sub_hindi` | Automated for EN/Hinglish; Hindi escalated |
| **delivery_issue** | `demo_del_en` | `demo_del_hinglish` | `demo_del_hindi` | Tampered package integrity |
| **technical_issue** | `demo_tech_en` | `demo_tech_hinglish` | `demo_tech_hindi` | High-priority checkout blocker |
| **fraud_report** | `demo_fraud_en` | `demo_fraud_hinglish` | `demo_fraud_hindi` | Critical safety zero-tolerance fraud |
| **account_closure** | `demo_close_en` | `demo_close_hinglish` | `demo_close_hindi` | GDPR / Legal right to erasure |
| **general_query** | `demo_gen_en` | `demo_gen_hinglish` | `demo_gen_hindi` | Automated for EN/Hinglish; Hindi escalated |

---

## 5. Automated Verification Test Suite

Run the full integration test suite verifying demo fixtures, safety policies, and rollback:
```powershell
.\.venv\Scripts\pytest.exe tests/test_demo_integration.py -v
```

Run all 99 project tests:
```powershell
.\.venv\Scripts\pytest.exe -q
```
