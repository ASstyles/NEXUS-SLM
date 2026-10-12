# NEXUS-SLM Deployment Guide

## 1. Hardware & Resource Requirements

| Resource | Minimum | Recommended Production |
|---|---|---|
| **CPU** | 4 Cores (x86_64 or ARM64) | 8–12 Cores (Intel Core i7/Xeon or AMD Ryzen/EPYC) |
| **RAM** | 4 GB | 8–16 GB |
| **GPU** | None (CPU Native) | Optional: NVIDIA GPU (4GB VRAM) for batch acceleration |
| **Disk Space** | 4 GB free space | 10 GB free space (model weights + logs) |
| **OS** | Linux (Ubuntu 22.04+), Windows 10/11, macOS | Linux Container (Debian/Ubuntu slim) |

---

## 2. Environment Variables Reference

| Variable | Default Value | Description |
|---|---|---|
| `ACTIVE_MODEL_VERSION` | `nexus-qlora-v1` | Model version tag loaded at startup |
| `API_HOST` | `0.0.0.0` | Binding host address |
| `API_PORT` | `8000` | Binding network port |
| `ENV` | `production` | Deployment environment mode |
| `ENABLE_TELEMETRY` | `true` | Enables JSONL telemetry recording |
| `TELEMETRY_STORAGE_PATH` | `monitoring/telemetry.jsonl` | Filepath for request telemetry |
| `LOG_SENSITIVE_TEXT` | `false` | When false, ticket text is **never** logged |
| `MAX_TICKET_CHARS` | `4000` | Enforces request length ceiling |
| `RATE_LIMIT_PER_MINUTE` | `120` | API rate limit per client IP |

---

## 3. Local Deployment (Virtual Environment)

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/organization/nexus-slm.git
cd nexus-slm

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### Step 2: Dependency Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Step 3: Configure Environment
```bash
cp .env.example .env
```

### Step 4: Run the Service
```bash
python scripts/deployment/run_api.py
```
The service will bind to `http://0.0.0.0:8000`.

---

## 4. Docker Container Deployment

### Step 1: Build Docker Image
```bash
docker build -t nexus-slm:1.0.0 -f Dockerfile .
```

### Step 2: Run Container
```bash
docker run -d \
  --name nexus-slm-service \
  -p 8000:8000 \
  -e ACTIVE_MODEL_VERSION=nexus-qlora-v1 \
  -e LOG_SENSITIVE_TEXT=false \
  --restart unless-stopped \
  nexus-slm:1.0.0
```

### Step 3: Verify Health
```bash
curl -f http://localhost:8000/health
```
Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "nexus-qlora-v1",
  "inference_mode": "neural",
  "hardware": {
    "device": "cpu",
    "cpu_count": 12,
    "ram_total_gb": 15.65
  }
}
```

---

## 5. Docker Compose Multi-Service Deployment

Deploy the complete stack (FastAPI inference engine, Next.js operational dashboard, MLflow experiment tracking):

```bash
docker compose up -d
```

### Service Map:
- **NEXUS-SLM Inference API**: `http://localhost:8000`
- **Triage Monitoring Dashboard**: `http://localhost:3000`
- **MLflow Model Registry**: `http://localhost:5000`

---

## 6. Zero-Downtime Rollback Procedure

To roll back to a previously registered model version without restarting or rebuilding the container:

```bash
curl -X POST "http://localhost:8000/model/rollback?target_version=baseline-smollm-135m"
```

To switch back to the candidate:
```bash
curl -X POST "http://localhost:8000/model/rollback?target_version=nexus-qlora-v1"
```

---

## 7. Graceful Shutdown

The Docker container and FastAPI service support standard POSIX termination signals:
```bash
docker stop --time 15 nexus-slm-service
```
This flushes in-flight telemetry logs to disk and terminates cleanly.

---

## 8. Authorised Internal Cloud Deployment Prerequisites

> **Notice:** Cloud deployment is currently **UNPROVISIONED**. NEXUS-SLM remains in Day 0 development and local sandbox mode. Do not claim a running cloud cluster exists.

Before any internal staging or shadow cloud deployment can proceed, the following external infrastructure prerequisites must be satisfied:

1. **Approved Cloud Infrastructure Provisioning (BLK-02):**
   - Formal Terraform/Helm repository for internal cloud Kubernetes or ECS/Cloud Run cluster.
   - Resource quota allocation: minimum 4 vCPU, 8 GB RAM per inference replica.
   - Private VPC peering with egress security groups blocking unauthorized outbound connections.
2. **Secrets & Identity Management:**
   - Secrets must be injected via AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault.
   - Zero hardcoded API keys or plaintext credentials in environment variables or container layers.
3. **Network Isolation & Ingress Security:**
   - mTLS or internal API Gateway authentication on all ingress routes.
   - Reverse proxy rate limiting (120 req/min/IP) and WAF protection against prompt-injection probing.
4. **CRM Mirror Authorization (BLK-01):**
   - Signed Data Protection Agreement (DPA) and formal Data Processing Addendum.
   - Read-only webhook authorization on synthetic/shadow queues only.

---

## 9. Public Demo Deployment Security & Safeguards

If deploying an external demonstration instance for stakeholder evaluation:
1. **Synthetic Data Exclusivity:** The public demo must strictly operate on synthetic test fixtures. Real CRM connectivity is prohibited.
2. **Abuse & Resource Limits:**
   - Restrict max request length to 1,000 characters.
   - Strict client rate limiting (10 requests per minute per IP).
   - In-memory circuit breaker to prevent denial-of-service CPU exhaustion.
3. **No Credential Exposure:** Container images must be audited to ensure zero repository secrets, private certificates, or internal URLs are bundled.

