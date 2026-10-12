# NEXUS-SLM — Data Authorization & Access Readiness Package

**Document ID**: `DATA-AUTH-READY-2026-V1`  
**Governing Blocker**: `BLK-01 — Data access and CRM authorization`  
**Target Candidate**: `nexus-qlora-v1` (Git Commit: `2696df8be85e75b69f8a04345265c7f29d931e02`)  
**Designated Owners**: Data Protection Officer (DPO) & Support CRM Systems Lead  
**Current Operational Status**: `BLOCKED — PENDING FORMAL EXECUTION & CREDENTIAL PROVISIONING`  

---

## 1. Executive Context & Purpose

This package defines the formal data governance, privacy controls, access boundaries, secret-management requirements, and sign-off procedure necessary to resolve **BLK-01**. 

> [!IMPORTANT]
> **Safety & Governance Barrier**:  
> In accordance with zero-trust release principles, **real customer tickets must NEVER be ingested into NEXUS-SLM without formal, documented authorization**. The live CRM mirror feed remains disabled, and no synthetic event will be represented as real customer data.

---

## 2. Required Formal Approvals & Legal Instruments

To authorize the mirroring of inbound customer support tickets into the NEXUS-SLM shadow evaluation pipeline, the following two organizational instruments must be executed:

1. **Internal Data Protection Impact Assessment (DPIA)**:
   - Evaluates risk of processing unstructured customer communications.
   - Confirms that local, air-gapped CPU model inference within the corporate VPC presents negligible third-party data leakage risk.
   - Status: **PENDING DPO FORMAL REVIEW**

2. **Shadow Mirror Data Processing Authorization (DPA Addendum)**:
   - Authorizes read-only mirroring of ticket text to the internal shadow queue.
   - Prohibits automated action dispatch or writeback to customer records.
   - Sets strict 30-day retention limits on shadow inference logs.
   - Status: **PENDING LEGAL / DPO SIGNATURE**

---

## 3. Data-Flow Architecture & Perimeter Boundaries

```
+---------------------------------------------------------------------------------------+
|                                PRODUCTION TRIAGE PERIMETER                            |
|                                                                                       |
|  [Customer Ticket] ---> [Production CRM Webhook (Salesforce / Zendesk)]               |
|                                     |                                                 |
|                        +------------+------------+                                    |
|                        | (Primary Authoritative) | (Passive Read-Only Mirror)         |
|                        v                         v                                    |
|               [Human Agent Queue]     [CRM Mirror Ingress Listener]                   |
|                        |                         |                                    |
+------------------------|-------------------------|------------------------------------+
                         |                         | (Internal VPC Network Only)
                         |                         v
                         |              +----------------------+
                         |              |   PrivacyRedactor    | <--- Redacts PAN, CVV,
                         |              +----------------------+      Tokens, Passwords
                         |                         |
                         |                         v
                         |              +----------------------+
                         |              | Bounded Async Queue  | <--- Max 500 items,
                         |              +----------------------+      Backpressure Drop
                         |                         |
                         |                         v
                         |              +----------------------+
                         |              | NEXUS-SLM Inference  | <--- SmolLM-135M CPU
                         |              +----------------------+
                         |                         |
                         |                         +---> [shadow_escalations.jsonl]
                         |                         |     (Audit Trail Only)
                         |                         +---> [Telemetry (No Plaintext)]
                         |
                         | (Zero Communication / Read-Only Passive Isolation)
                         x <------------------------------------------------------------
```

### 3.1 Strict Flow Boundaries
1. **Unidirectional Passive Mirroring**: The shadow pipeline operates exclusively as a downstream sink. No network paths, credentials, or API tokens exist that allow NEXUS-SLM to send data back to the primary CRM, customer notification service, or payment gateway.
2. **Air-Gapped Local Inference**: All model inference executes locally on VPC CPU instances. No customer data leaves the corporate VPC perimeter or makes calls to commercial LLM APIs.
3. **Failure Isolation**: An ingestion failure, queue backpressure drop, model timeout, or worker crash has **zero effect** on the primary human agent delivery path.

---

## 4. Permitted CRM Mirror Permissions

The CRM webhook or service account provisioned for the shadow mirror must adhere to the principle of least privilege:

| Permission Scope | Allowed / Forbidden | Justification |
|:---|:---:|:---|
| `tickets:read` | **ALLOWED** | Required to mirror inbound customer ticket subject and body. |
| `tickets:listen_webhook` | **ALLOWED** | Required to receive real-time asynchronous mirror push events. |
| `tickets:write` | **STRICTLY PROHIBITED** | Shadow pipeline must not mutate ticket status, tags, or assignees. |
| `tickets:comment` | **STRICTLY PROHIBITED** | Automated messaging to customers is prohibited. |
| `users:read_pii` | **MINIMIZED** | Customer email, phone, and billing details must not be requested. |
| `payments:*` | **STRICTLY PROHIBITED** | Zero access to payment gateways, refund APIs, or card databases. |

---

## 5. Secret-Management Architecture

Credentials for the CRM mirror webhook must never be stored in repository files, environment scripts, or plain text configuration.

### 5.1 Required Secret Specifications
- **Secret Key Name**: `CRM_WEBHOOK_SECRET`
- **Secret Type**: High-entropy HMAC-SHA256 signing secret (minimum 32 bytes / 256 bits).
- **Approved Secret Managers**:
  - AWS Secrets Manager / Parameter Store (path: `/nexus-slm/production/crm_webhook_secret`)
  - HashiCorp Vault (path: `secret/data/nexus-slm/crm_webhook_secret`)
  - Kubernetes Secret (name: `nexus-slm-secrets`, key: `crm-webhook-secret`)

### 5.2 Secret Injection & Rotation Protocol
1. **Container Injection**: Secrets must be injected at container startup as environment variables from the Kubernetes Secret / Vault agent.
2. **Memory Scrubbing**: Credentials must not be logged or exposed via diagnostic endpoints (`/health`, `/metrics`, or `/model/info`).
3. **Rotation SLA**: Annual rotation or immediate 1-hour revocation upon suspected compromise.

---

## 6. Privacy, Redaction, and Data Retention Controls

### 6.1 Synchronous Ingress Redaction (`PrivacyRedactor`)
Prior to being enqueued in the shadow buffer or presented to model tokenization, all ticket text is sanitized synchronously by [src/pilot/privacy_redactor.py](file:///c:/Users/avira/OneDrive/Desktop/NEXUS-SLM/src/pilot/privacy_redactor.py):
- **Payment Card Numbers (PAN)**: 13–19 digit sequences matching Luhn checksum are replaced with `[CARD_REDACTED]`.
- **Card Security Codes (CVV)**: 3–4 digit codes adjacent to card patterns are replaced with `[CVV_REDACTED]`.
- **Authentication Secrets**: Bearer tokens, private keys, API keys, passwords, and authorization headers are replaced with `[SECRET_REDACTED]`.
- **Customer Contact Information**: Email addresses $\to$ `[EMAIL_REDACTED]`; Phone numbers $\to$ `[PHONE_REDACTED]`.

### 6.2 Telemetry Privacy Policy
- Application logs (`monitoring/telemetry.jsonl`) record only performance metrics, token lengths, status codes, and SHA256 content hashes.
- Full ticket text is **strictly excluded** from persistent telemetry logs.

### 6.3 Data Retention & Purge Policy
- **Shadow Escalation Records** (`experiments/artifacts/shadow_escalations.jsonl`): Retained for **30 calendar days** for safety audit compliance, then securely deleted.
- **Double-Blind Review Pool** (`data/pilot/`): Quarantined under encryption; retained for **60 calendar days** post-pilot for model validation; excluded from model training splits.

---

## 7. Verification Steps for Blocker Resolution

To verify and close **BLK-01**, the following technical and operational verification steps must be executed:

1. **Verification Step 1 — Formal Approval Receipt**:
   - Confirm executed DPIA and DPA Addendum are filed in the compliance repository.
2. **Verification Step 2 — Credential Presence Check**:
   ```bash
   # Must return non-empty without printing the secret value
   .venv\Scripts\python.exe -c "import os; assert os.getenv('CRM_WEBHOOK_SECRET') and len(os.getenv('CRM_WEBHOOK_SECRET')) >= 32, 'CRM_WEBHOOK_SECRET missing or invalid'; print('Credential verified present in environment.')"
   ```
3. **Verification Step 3 — Webhook Handshake Validation**:
   - Send an authenticated test ping with valid HMAC header; confirm HTTP 200 return code.
4. **Verification Step 4 — Ingress Redaction Verification**:
   - Confirm test tickets containing synthetic test PANs and tokens are scrubbed before inference.

---

## 8. Formal Data Authorization Sign-Off

*This section must be completed and signed by authorized personnel before live mirror activation.*

```
========================================================================================
                          NEXUS-SLM DATA AUTHORIZATION SIGN-OFF
========================================================================================

1. Data Protection Officer (DPO) Approval:
   Name:      __________________________________________________
   Title:     __________________________________________________
   Signature: __________________________________________________  Date: _______________
   Status:    [ ] APPROVED FOR SHADOW MIRRORING    [ ] REJECTED

2. CRM Systems Owner Approval:
   Name:      __________________________________________________
   Title:     __________________________________________________
   Signature: __________________________________________________  Date: _______________
   Status:    [ ] READ-ONLY WEBHOOK CONFIGURED     [ ] REJECTED

3. Head of Information Security Approval:
   Name:      __________________________________________________
   Title:     __________________________________________________
   Signature: __________________________________________________  Date: _______________
   Status:    [ ] SECRET MANAGEMENT VERIFIED       [ ] REJECTED
========================================================================================
```
