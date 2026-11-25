# NEXUS-SLM — End-to-End Reproducibility Guide

Every quantitative figure, dataset checksum, evaluation snapshot, security audit, and benchmark reported in this sprint can be reproduced cleanly on a fresh checkout or local Python 3.10+ environment.

---

## 1. Environment Setup

```powershell
# Clone repository (on validation sprint branch or clean checkout)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ci.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Large binary weights (`*.safetensors`, `*.gguf`, `*.onnx`, `*.pt`) are excluded via `.gitignore`. A clean clone runs deterministic **Heuristic Fallback V2/V3** immediately with zero binary downloads, and reproduces the neural research pipeline once Hugging Face `Qwen/Qwen2.5-0.5B-Instruct` is cached locally.

---

## 2. Exact Verification & Reproduction Command Sequence

```powershell
# 1. Verify dataset and model SHA-256 provenance manifests
.\.venv\Scripts\python.exe scripts\deployment\generate_artifact_manifests.py --verify

# 2. Verify independent Human Eval V2 governance scaffold (PENDING status until N>=300 reviewed)
.\.venv\Scripts\python.exe scripts\data\human_eval_v2_manager.py validate

# 3. Reproduce Refined Policy V2 baseline & verify documentation consistency
.\.venv\Scripts\python.exe scripts\evaluation\reproduce_refined_v2_baseline.py
.\.venv\Scripts\python.exe scripts\evaluation\check_documentation_consistency.py

# 4. Evaluate Hardened Policy V3 & Multilingual/8-Domain Safety Audit
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_hardened_v3_policy.py
.\.venv\Scripts\python.exe scripts\evaluation\run_multilingual_and_safety_audit.py

# 5. Reproduce V3 diverse dataset (verified zero leakage) and V2 Multitask Neural experiments + calibration
.\.venv\Scripts\python.exe scripts\data\build_v3_diverse_dataset.py
.\.venv\Scripts\python.exe scripts\experiments_v2\run_v2_architecture_experiments.py

# 6. Benchmark CPU runtimes (Heuristic, GGUF Q4_K_M/Q8_0, ONNX INT8, PyTorch Hybrid) & Live HTTP Audit
.\.venv\Scripts\python.exe scripts\evaluation\benchmark_all_runtimes_v2.py
.\.venv\Scripts\python.exe scripts\deployment\run_live_http_load_and_failover_audit.py

# 7. Run Security Audit & Full Pytest Suite + Next.js Dashboard Build
.\.venv\Scripts\python.exe scripts\deployment\run_security_audit.py
.\.venv\Scripts\python.exe -m pytest -v tests\
cd dashboard; npm install; npm run build
```

---

## 3. Canonical Dataset & Weight Checksums (`data/dataset_manifest.json`, `models/artifact_manifest.json`)

- `data/v2_clean_template_grouped/train_clean.jsonl` (`N=4,797`): `f67b50c7724e8cd53d029058c688e712b99b9a3f1be3df0a92c756975220fe1c`
- `data/v2_clean_template_grouped/val_clean.jsonl` (`N=449`): `faa98cf83cbe209736d8890a902f65fa382cb2ff31b96e55608d98d9e68fa29a`
- `data/v2_clean_template_grouped/test_clean.jsonl` (`N=746`): `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b`
- `data/human_eval/annotation_template.jsonl` (`N=200` total, `184` dev-exposed active): `e26d4eec99ddd247d3037652466df0024100b0cc85b9ca48e0fa32ed300923d3`
- `data/v3_diverse/train_v3.jsonl` (`N=680`): `d72cba8975c1bb0160704643c567b420b34b7fd1a60897fdecc5e43465d10c04`
- `data/v3_diverse/cal_v3.jsonl` (`N=84`): `af638a6e21c671f32d666f49a817a9c5074dd9868d5be501c31e26b58beb1b94`
- `data/v3_diverse/dev_audit_v3.jsonl` (`N=84`): `0a28299713712c99689fd1f7bfb825d3174b16ac23d175c1fb66d9abf71dfab4`
- `experiments/artifacts/v2_models/nexus_v2_hybrid_multitask_head.pt` (gitignored): `29156d3904293c0fa60819144cd4fa87b8bed4093135fd24cea989772ba680b6`
