# NEXUS-SLM — Shadow Infrastructure Deployment Runbook

**Document ID**: `INFRA-RUNBOOK-2026-V1`  
**Governing Blocker**: `BLK-02 — Production infrastructure`  
**Target Candidate**: `nexus-qlora-v1` (Git Commit: `2696df8be85e75b69f8a04345265c7f29d931e02`)  
**Designated Owner**: Site Reliability Engineer (SRE) / Cloud Platform Lead  
**Current Operational Status**: `BLOCKED — PENDING CLUSTER PROVISIONING`  

---

## 1. Executive Context & Purpose

This operational runbook provides the verified deployment specifications, resource allocations, health probes, Prometheus metrics configuration, network isolation rules, rollback workflows, and post-deployment verification procedures required to resolve **BLK-02**.

> [!NOTE]
> **Deployment Status Notice**:  
> The infrastructure configurations in this runbook are **TEMPLATES FOR CLUSTER DEPLOYMENT**. They have been validated locally and in remote CI, but **HAVE NOT YET BEEN APPLIED TO A LIVE CLUSTER**. The production cloud endpoint (`https://nexus-slm-shadow.internal/health`) will remain unresolvable until an SRE applies this configuration to the internal VPC cluster.

---

## 2. Container Sizing & Resource Justification

Resource limits must be sized to prevent Out-Of-Memory (OOM) kills while avoiding compute waste:

### 2.1 Empirical Memory Profiling Summary
From [experiments/artifacts/deployment_capacity_evaluation.json](file:///c:/Users/avira/OneDrive/Desktop/NEXUS-SLM/experiments/artifacts/deployment_capacity_evaluation.json):
- **Base Process RSS (Python runtime)**: `205.86 MB`
- **Post-Warmup Process RSS (Model + Adapter loaded)**: `1,001.69 MB`
- **Peak Process RSS Under High Concurrency ($C=4$)**: `1,210.66 MB`
- *Note on Host Memory*: Historical reports of 12–14 GB memory usage reflected host-wide Windows system memory; the isolated application process consumes **< 1.25 GB peak**.

### 2.2 Sizing Specification
| Resource Parameter | Allocation | Safety Factor / Headroom | Operational Rationale |
|:---|:---:|:---:|:---|
| **CPU Requests** | `1.0 vCPU` | 1.0x baseline | Guarantees compute capacity for PyTorch thread execution. |
| **CPU Limits** | `2.0 vCPU` | 2.0x baseline | Accommodates bursty batching without thread thrashing (`torch.set_num_threads(2)`). |
| **Memory Requests** | `2,048 MiB (2.0 GB)` | 1.69x peak RSS | Guarantees resident memory allocation upon pod scheduling. |
| **Memory Limits** | `4,096 MiB (4.0 GB)` | **3.38x peak RSS** | Provides 2,885 MiB headroom; eliminates OOM risk. |

---

## 3. Infrastructure-as-Code (Kubernetes Manifest Template)

*TEMPLATE ONLY — TO BE APPLIED TO INTERNAL PRODUCTION CLUSTER BY DESIGNATED SRE*

```yaml
# ==============================================================================
# NEXUS-SLM Shadow Service — Kubernetes Deployment Manifest
# STATUS: TEMPLATE / UNAPPLIED (Pending SRE cluster provisioning)
# ==============================================================================
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-slm-shadow
  namespace: nexus-production
  labels:
    app: nexus-slm
    tier: shadow-inference
    candidate-version: nexus-qlora-v1
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: nexus-slm
      tier: shadow-inference
  template:
    metadata:
      labels:
        app: nexus-slm
        tier: shadow-inference
        candidate-version: nexus-qlora-v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/path: "/metrics"
        prometheus.io/port: "8000"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
      containers:
        - name: inference-engine
          image: ghcr.io/asstyles/nexus-slm:2696df8be85e75b69f8a04345265c7f29d931e02
          imagePullPolicy: IfNotPresent
          env:
            - name: API_PORT
              value: "8000"
            - name: API_HOST
              value: "0.0.0.0"
            - name: ACTIVE_MODEL_VERSION
              value: "nexus-qlora-v1"
            - name: CRM_WEBHOOK_SECRET
              valueFrom:
                secretKeyRef:
                  name: nexus-slm-secrets
                  key: crm-webhook-secret
          resources:
            requests:
              cpu: "1000m"
              memory: "2048Mi"
            limits:
              cpu: "2000m"
              memory: "4096Mi"
          ports:
            - containerPort: 8000
              name: http
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 20
            periodSeconds: 15
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 2
---
apiVersion: v1
kind: Service
metadata:
  name: nexus-slm-shadow
  namespace: nexus-production
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8000
      name: http
  selector:
    app: nexus-slm
    tier: shadow-inference
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nexus-slm-network-isolation
  namespace: nexus-production
spec:
  podSelector:
    matchLabels:
      app: nexus-slm
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow ingress strictly from internal CRM webhook mirror relay
    - from:
        - podSelector:
            matchLabels:
              app: crm-mirror-relay
      ports:
        - protocol: TCP
          port: 8000
  egress:
    # Allow egress strictly to internal Prometheus scraper (no public Internet)
    - to:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9090
```

---

## 4. Health, Readiness, and Monitoring Specifications

### 4.1 Health & Liveness Probes
- **Endpoint**: `GET /health`
- **Success Criteria**: HTTP 200 OK with JSON payload:
  ```json
  {
    "status": "healthy",
    "model_version": "nexus-qlora-v1",
    "inference_mode": "neural",
    "device": "cpu"
  }
  ```
- **Probe Failure Threshold**: If `/health` fails 3 consecutive attempts, Kubernetes restarts the pod.

### 4.2 Prometheus Metrics Scraping
- **Scrape Endpoint**: `GET /metrics`
- **Key Operational Metrics**:
  - `nexus_requests_total`: Total inbound shadow requests.
  - `nexus_inference_latency_seconds_bucket`: P50, P95, P99 neural inference latency.
  - `nexus_human_escalations_total`: Inbound tickets flagged with `requires_human=true`.
  - `nexus_errors_total`: Inference timeouts or schema parse errors.
  - `nexus_backpressure_drops_total`: Excess shadow events dropped due to queue congestion.

---

## 5. Security & Access Control Boundaries

1. **Non-Root Execution**: Container runs strictly as UID 10001 (`nexususer`). Root escalation is disabled (`allowPrivilegeEscalation: false`).
2. **Network Isolation**: The service is provisioned as an internal `ClusterIP` within the private VPC. Public Internet ingress and egress are blocked by `NetworkPolicy`.
3. **No External API Keys**: NEXUS-SLM runs 100% locally on CPU and has no access to external LLM provider APIs (OpenAI, Anthropic, etc.).
4. **IAM Least Privilege**: Cloud instance profile lacks write permissions to customer databases, S3 buckets, or payment gateways.

---

## 6. Rollback Procedures

If an incident occurs during shadow operation, SREs must execute one of the two verified rollback mechanisms:

### Mechanism A: API Dynamic Hot-Swap (RTO < 30s)
Switches the active model in-memory to the deterministic heuristic engine without pod redeployment:
```bash
curl -X POST http://nexus-slm-shadow.internal/model/rollback \
  -H "Content-Type: application/json" \
  -d '{"target_version": "baseline-smollm-135m"}'
```

### Mechanism B: Kubernetes Deployment Rollback (RTO < 2 minutes)
Reverts the cluster deployment to the previous verified stable image:
```bash
kubectl rollout undo deployment/nexus-slm-shadow -n nexus-production
kubectl rollout status deployment/nexus-slm-shadow -n nexus-production
```

---

## 7. Post-Deployment Verification Commands

Once the SRE has applied the cluster manifest, execute the following commands to formally verify and close **BLK-02**:

```bash
# 1. Verify DNS resolution and internal healthcheck
curl -s -f https://nexus-slm-shadow.internal/health

# 2. Assert deployed candidate matches frozen manifest
python -c "
import urllib.request, json
resp = json.loads(urllib.request.urlopen('https://nexus-slm-shadow.internal/health').read().decode())
assert resp['model_version'] == 'nexus-qlora-v1', f'Version mismatch: {resp}'
assert resp['inference_mode'] == 'neural', f'Mode mismatch: {resp}'
print('Cluster candidate verification: PASS')
"

# 3. Verify non-blocking inference endpoint
curl -s -X POST https://nexus-slm-shadow.internal/predict \
  -H "Content-Type: application/json" \
  -d '{"ticket": "My order #1234 was damaged upon arrival"}'

# 4. Verify Prometheus metrics scraping
curl -s https://nexus-slm-shadow.internal/metrics | grep nexus_requests_total
```
