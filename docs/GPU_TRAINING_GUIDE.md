# NEXUS-SLM — GPU Training Guide: Qwen2.5-0.5B-Instruct LoRA

**Document ID:** `docs/GPU_TRAINING_GUIDE.md`  
**Pipeline Phase:** Phase 5 ML Experimentation  
**Canonical Candidate:** `nexus-qlora-v1` (SmolLM-135M, Preserved Active)  
**Experimental Model:** `Qwen/Qwen2.5-0.5B-Instruct` (494M parameters, 151k vocabulary)  
**Experiment Configuration:** `experiments/configs/qwen25_05b_lora_seed42.json`  

---

## 1. Overview & Scientific Rationale

NEXUS-SLM is an edge-optimized Small Language Model specialized for multilingual customer support ticket triage across English, Hinglish, and native Devanagari Hindi.

### 1.1 The Baseline Bottleneck in SmolLM-135M (`nexus-qlora-v1`)
The active production candidate `nexus-qlora-v1` achieves **0.4286 Category Accuracy** and **0.4017 Macro-F1** on the frozen historical benchmark ($N=42$). However, exhaustive empirical audits revealed two architectural ceiling factors:
1. **Devanagari BPE Byte-Fallback Defect:** SmolLM's 49,152-token BPE tokenizer lacks native Indic subwords, decomposing Hindi text into 6–10 byte replacement tokens (`\ufffd`) per word. On native Devanagari Hindi, Category Accuracy collapses to **0.1429 (14.29%)**, requiring 100% routing to human agents via deterministic safeguards.
2. **135M Parameter Representational Ceiling:** With only 134.5M parameters, the model experiences complete recall starvation ($F_1 = 0.0$) on four overlapping semantic categories (`payment_failure`, `order_tracking`, `technical_issue`, `product_complaint`). Retraining on scaled data in V2 triggered catastrophic negative interference.

### 1.2 The Qwen2.5-0.5B Architectural Advantage
`Qwen/Qwen2.5-0.5B-Instruct` resolves these issues:
- **151,665-Token Multilingual Vocabulary:** Contains native Devanagari subwords, completely eliminating byte-fallback tokens and reducing Hindi sequence length by **17.1%**.
- **494M Parameters (3.67x capacity):** High-capacity Grouped Query Attention (GQA) and SwiGLU activations retain distinct decision boundaries across all 14 categories.
- **Apache 2.0 Permissive License:** Fully un-gated and approved for commercial edge deployment.

---

## 2. Hardware Preflight & Execution Constraints

### 2.1 Why Training is Blocked on Local CPU
The local Windows host runs a PyTorch CPU runtime with approximately 1.0–1.8 GB of unreserved RAM. Backpropagating gradients for 494M parameters using AdamW with gradient accumulation requires $\sim 4.5\text{ GB}$ of active optimizer and tensor memory. Executing full fine-tuning locally would trigger severe OS paging, memory exhaustion (OOM), or system freeze.

The training pipeline enforces an **uncompromising GPU preflight check**:
- If `torch.cuda.is_available()` is false, the script halts cleanly with an informative error rather than silently degrading into CPU training.
- CPU training is only allowed in `--allow-cpu-smoke-test` mode for 2-instance CI verification.

### 2.2 Recommended GPU Compute Environments
| Environment | GPU Type | VRAM | Estimated 3-Epoch Duration | Cost |
| :--- | :--- | :---: | :---: | :---: |
| **Kaggle Notebooks** *(Recommended)* | NVIDIA T4 x2 or P100 | 16 GB | $\sim 8 - 12\text{ minutes}$ | **Free** (30 hrs/wk quota) |
| **Google Colab** | NVIDIA T4 | 15 GB | $\sim 10 - 15\text{ minutes}$ | Free tier |
| **Cloud GPU (RunPod / Lambda)** | NVIDIA A10G / RTX 3090 | 24 GB | $\sim 3 - 5\text{ minutes}$ | $\sim \$0.20$ |
| **Local CUDA Workstation** | NVIDIA RTX 2050 / 3060 / 4060 | $\ge 4\text{ GB}$ (8-bit) / $\ge 8\text{ GB}$ (fp16) | $\sim 15 - 25\text{ minutes}$ | Local hardware |

---

## 3. Option A — Running on Kaggle (Zero-Cost GPU Workflow)

The fastest and most reproducible method is to run `notebooks/nexus_qwen25_05b_lora_training.ipynb` on Kaggle.

### Step 1: Open Kaggle & Create a New Notebook
1. Navigate to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. In the right-hand panel under **Notebook options**:
   - **Accelerator:** Select `GPU T4 x2` or `GPU P100`.
   - **Internet:** Toggle **On**.
   - **Environment:** Select `Always use latest environment`.

### Step 2: Upload the Training Notebook
1. Click **File -> Upload notebook**.
2. Select [`notebooks/nexus_qwen25_05b_lora_training.ipynb`](file:///c:/Users/avira/OneDrive/Desktop/NEXUS-SLM/notebooks/nexus_qwen25_05b_lora_training.ipynb).

### Step 3: Zero-Setup Data Portability (Self-Extracting Datasets)
The notebook is **100% self-contained**. You do not need to manually upload dataset files or connect private git repositories:
- When opened in a cloned repository, it automatically reads the local `data/` splits.
- When opened in a standalone cloud runtime (Kaggle or Google Colab), it automatically **self-extracts the verified canonical dataset files** (`train.jsonl`, `validation.jsonl`, `test.jsonl`).
- It verifies their cryptographic SHA-256 digests against locked standards:
  - `data/train/train.jsonl`: `11df262072d7243f623bfd829b407a8dbd2c5a3087ed697a836a7152c7688967`
  - `data/validation/validation.jsonl`: `2e349ad3b1330aec224c266460d2e4189b94138a1d63146a10bd8f45ea65508e`
  - `data/test/test.jsonl`: `edf3fa5cd949a149ad5d759d81553d9704467491877d0f3c8b7cc5a887a9eda4`

### Step 4: Run the Notebook
Execute all cells sequentially:
1. **Preflight:** Validates CUDA device name, total VRAM, and bfloat16 support.
2. **Dependencies:** Installs pinned `transformers`, `peft`, `accelerate`, and `datasets`.
3. **Data Masking:** Prepares completion-only loss tensors.
4. **Smoke Test:** Executes a 1-step forward/backward pass.
5. **Training:** Trains for 3 epochs with Cosine LR schedule and evaluation logging.
6. **Export:** Exports the LoRA adapter, tokenizer files, and `nexus_qwen_metadata.json`.
7. **Packaging:** Creates `nexus_qwen25_05b_lora_seed42.zip`.

### Step 5: Retrieve Output Artifacts
Download the generated zip from the Kaggle Notebook output tab, or run via Kaggle CLI:
```bash
kaggle kernels output <username>/nexus-qwen25-05b-lora-training -p models/qwen25_05b_lora_seed42/
```
Unpack into the local repository:
```powershell
Expand-Archive -Path models/qwen25_05b_lora_seed42/nexus_qwen25_05b_lora_seed42.zip -DestinationPath models/qwen25_05b_lora_seed42/
```

---

## 4. Option B — Running via CLI on a CUDA Workstation

If executing on a dedicated Linux or Windows machine with CUDA-enabled PyTorch:

```bash
# 1. Activate environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Verify GPU availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 3. Launch reproducible fine-tuning
python scripts/training/train_qwen25_05b_lora.py \
    --config experiments/configs/qwen25_05b_lora_seed42.json \
    --require-gpu
```

---

## 5. Training Protocol & Technical Guarantees

### 5.1 Completion-Only Loss Masking
A critical requirement in instruction fine-tuning is preventing the model from calculating loss on prompt tokens (system instruction, customer ticket text). Training on prompt tokens wastes gradient updates and causes the model to generate repetitive prompts instead of JSON.

The custom `prepare_masked_dataset` and `CompletionOnlyDataCollator` ensure:
- All prompt tokens (system message, ticket text, `<|im_start|>assistant\n`) receive label `-100`.
- Only assistant output tokens (`{"category": "...", ...}<|im_end|>`) have valid label token IDs.
- Dynamic batch padding tokens receive label `-100`.
- In PyTorch, `CrossEntropyLoss(ignore_index=-100)` strictly computes loss on the target JSON.

### 5.2 Vocabulary Size Integrity
The candidate model configuration exhibits a known vocabulary discrepancy:
- `model.config.vocab_size` = **151,936** (embedding layer size, padded to multiple of 64).
- `len(tokenizer)` = **151,665** (actual discrete subword count).
- `pad_token_id` = **151643** (`<|endoftext|>`).
- `eos_token_id` = **151645** (`<|im_end|>`).

The pipeline loads the native Qwen tokenizer and logs both numbers separately, ensuring embeddings are not truncated or misaligned.

### 5.3 LoRA Hyperparameters (`experiments/configs/qwen25_05b_lora_seed42.json`)
```json
{
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "seed": 42,
  "lora": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
      "q_proj", "k_proj", "v_proj", "o_proj",
      "gate_proj", "up_proj", "down_proj"
    ]
  },
  "training": {
    "num_train_epochs": 3,
    "learning_rate": 0.0002,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 8,
    "max_seq_length": 384,
    "weight_decay": 0.01,
    "fp16": true,
    "gradient_checkpointing": true
  }
}
```

---

## 6. Candidate Acceptance Gates & Governance

An experimental candidate cannot replace `nexus-qlora-v1` until it empirically satisfies all six gates:

| Gate | Description | Threshold | Baseline V1 | Verification Method |
| :---: | :--- | :---: | :---: | :--- |
| **Gate 1** | Validation Split Macro-F1 ($N=44$) | $\ge \mathbf{0.5500}$ | 0.4412 | Unweighted macro average across all 14 classes on un-seen validation split. |
| **Gate 2** | Native Devanagari Hindi Accuracy | $\ge \mathbf{0.5000}$ | 0.1429 | Correct category classification on Hindi tickets without transliteration. |
| **Gate 3** | High-Risk Escalation Recall | $=\mathbf{1.0000}$ | 1.0000 | 100% interception of fraud, unauthorized debits, and security vectors ($0\text{ FN}$). |
| **Gate 4** | JSON Schema Validity Rate | $\ge \mathbf{95.0\%}$ | 97.6% | Conformance to strict Pydantic `OutputParser` without code block hallucination. |
| **Gate 5** | CPU Inference Latency & RAM | $\text{P95} \le \mathbf{2,500\text{ ms}}$<br>$\text{RAM} \le \mathbf{2.0\text{ GB}}$ | 1,306 ms<br>517 MB | Single-thread CPU benchmark matching edge deployment hardware. |
| **Gate 6** | System & Contract Invariants | **PASS** | PASS | Full compatibility with FastAPI `/api/v1/triage`, dashboard, and fallback policies. |

### Preserving Model Governance
- The candidate remains cataloged as `qwen25_05b_lora_seed42` (Experimental Candidate).
- `models/registry.json`, active API aliases, and dashboard baselines must **NOT** be modified until trained weights pass all gates.
