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
