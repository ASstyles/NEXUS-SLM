# NEXUS-SLM Next.js Dashboard Architecture & Verification Report

## 1. Executive Summary

The NEXUS-SLM operator dashboard is a high-performance Next.js 14 web application designed for real-time customer support operations, live triage demonstration, model registry governance, error diagnosis, and infrastructure monitoring.

This report documents the architectural audit, component verification, and operational capabilities of the dashboard frontend (`dashboard/`).

---

## 2. Dashboard Component Architecture

The frontend is structured into seven primary functional views managed within `dashboard/pages/index.tsx`:

### 2.1 Overview & System Health
- **Live Connection Monitor:** Probes `/health` and displays API connection status, active model version, and inference engine.
- **Hardware Telemetry:** Visualizes real-time CPU core utilization, memory consumption (RSS in MB), and active thread count.
- **Key Metrics Scorecard:** Summarizes category accuracy, P50 latency, human escalation rate, and critical recall.

### 2.2 Live Ticket Triage & Interactive Sandbox (`/analyze`)
- **Interactive Input:** Operators can submit free-form customer inquiries in English, Hinglish, or Devanagari Hindi.
- **Multilingual Support:** One-click pre-populated test prompts representing edge cases across all three linguistic modalities.
- **Real-Time Classification Output:**
  - Support category badge with contextual icon.
  - Urgency priority indicator (`low`, `medium`, `high`, `critical`).
  - Detected language badge with confidence score.
  - Human escalation flag (`requires_human`) and transparent escalation reasoning banner.
  - End-to-end request latency telemetry.
- **Administrative Model Rollback:** Authorized administrators can trigger immediate hot-swap rollbacks between registered models directly from the UI.

### 2.3 Benchmark Intelligence & Comparative Analytics (`/benchmark`)
- **Comparison Matrix:** Side-by-side comparative grid contrasting Heuristic Fallback V2, GGUF Q4_K_M, ONNX INT8, and PyTorch FP32.
- **Quantization Degradation Analysis:** Visualizes JSON validity loss and accuracy trade-offs across quantization profiles.
- **Latency & Resource Telemetry:** Percentile latency graphs (P50, P90, P95, P99) and peak RAM profiles.
- **Cost Simulator:** Dynamic interactive ROI calculator modeling monthly operational expenditure savings across configurable ticket volumes ($10\text{k}$ to $1\text{M}$ tickets/month).

### 2.4 Experiment Tracking & Leaderboard (`/experiments`)
- Displays historical training runs, hyperparameter configurations, learning rate schedules, LoRA adapter configurations ($r=16, \alpha=32$), and convergence loss curves.

### 2.5 Error Analysis & Confusion Breakdown (`/error_analysis`)
- Granular confusion matrices and top failure modes for the experimental neural model.
- Class-level recall disparities highlighting under-represented categories (`product_complaint`, `general_query`).

### 2.6 Model Registry & Governance (`/registry`)
- Displays registered models (`nexus-qlora-v1`, `qwen25_05b_clean_v2_lora`), deployment lifecycle stages (`production`, `experimental_validated`), cryptographic artifact hashes, and promotion gate requirements.

### 2.7 Production Monitoring & Audit Logs (`/monitoring`)
- Real-time stream of incoming triage events, request IDs, rate-limiting headers, and failover notifications.

---

## 3. Build & Operational Verification

| Dimension | Verification Target | Status |
|---|---|:---:|
| **Framework Version** | Next.js 14.2.15, React 18.3.1, TypeScript 5.4.5 | **PASS** |
| **Component Typing** | Strict TypeScript interfaces for all API payloads | **PASS** |
| **Production Build** | `npm run build` succeeds with zero compilation errors | **PASS** |
| **API Proxy Routing** | `/api/backend/*` rewrites to FastAPI backend (`http://127.0.0.1:8000`) | **PASS** |
| **Responsive Layout** | Flexbox/CSS grid responsive layout across desktop and tablet viewports | **PASS** |

### Acceptance Status
The NEXUS-SLM Dashboard is **CERTIFIED FOR PRODUCTION OPERATOR USE**.
