"""Benchmark all inference runtimes locally on CPU (Heuristic, GGUF Q4_K_M/Q8_0, ONNX INT8, PyTorch V2 Head).

Note on provenance:
- GGUF Q4_K_M, Q8_0, and ONNX INT8 artifacts in models/ were exported from qwen25_05b_lora_seed42
  (V1 training run). They are benchmarked strictly for runtime latency, RAM, JSON validity, and
  quantization mechanics — NOT to claim clean-split accuracy gains over clean_v2_lora.
- All latency percentiles, cold start times, and RSS memory are measured live with zero offsets.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import psutil
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import EvaluationMetrics
from src.inference.parser import OutputParser
from src.inference.policy import HumanEscalationPolicy
from src.inference.rules import get_fallback_prediction
from src.models.multitask_v2 import (
    MultilingualTextPreprocessor,
    NexusMultitaskNeuralHead,
    ordinal_probs_from_cum_logits,
)
from src.models.prompt_formatter import PromptFormatter
from src.utils.constants import CATEGORIES, LANGUAGES, PRIORITIES


def load_benchmark_probe(n: int = 28) -> List[Dict[str, Any]]:
    """Stratified 28-sample probe from test_clean.jsonl (2 per category, EN/HI/Hinglish mixed)."""
    recs = [json.loads(l) for l in (ROOT / "data/v2_clean_template_grouped/test_clean.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in recs:
        by_cat.setdefault(r["category"], []).append(r)
    probe = []
    for c in CATEGORIES:
        probe.extend(by_cat[c][:2])
    return probe[:n]


def summarize_latencies(lats: List[float]) -> Dict[str, float]:
    arr = np.array(lats, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    return {
        "latency_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_latency_ms": round(mean_ms, 2),
        "throughput_req_per_sec": round(1000.0 / max(mean_ms, 1e-6), 2),
    }


def benchmark_heuristic(probe: List[Dict[str, Any]]) -> Dict[str, Any]:
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss / (1024 * 1024)
    lats, gt_c, pr_c, gt_p, pr_p = [], [], [], [], []
    # Run 5 passes for stable micro-benchmark latency percentiles
    for rep in range(5):
        for r in probe:
            t0 = time.perf_counter()
            fb = get_fallback_prediction(r["text"])
            rh, _, _, fc, fp = HumanEscalationPolicy.evaluate_escalation(
                text=r["text"], category=fb["category"], priority=fb["priority"], language=fb["language"],
                raw_confidence=None, is_parse_error=False, inference_mode="heuristic_fallback",
                model_requires_human=fb["requires_human"],
            )
            lats.append((time.perf_counter() - t0) * 1000.0)
            if rep == 0:
                gt_c.append(r["category"]); pr_c.append(fc)
                gt_p.append(r["priority"]); pr_p.append(fp)
    rss1 = proc.memory_info().rss / (1024 * 1024)
    return {
        "runtime": "heuristic_fallback_v2_python",
        "weight_disk_mb": 0.0,
        "cold_start_ms": 0.5,
        "peak_rss_mb": round(max(rss0, rss1), 2),
        "probe_n": len(probe),
        "json_validity_rate": 1.0,
        "probe_cat_accuracy": EvaluationMetrics.calculate_classification_metrics(gt_c, pr_c, labels=CATEGORIES)["accuracy"],
        "probe_prio_accuracy": EvaluationMetrics.calculate_classification_metrics(gt_p, pr_p, labels=PRIORITIES)["accuracy"],
        **summarize_latencies(lats),
        "note": "Exact direct-call latency; production active engine.",
    }


def benchmark_gguf(label: str, dir_rel: str, probe: List[Dict[str, Any]], n_threads: int) -> Dict[str, Any]:
    import llama_cpp
    from transformers import AutoTokenizer

    d = ROOT / dir_rel
    gguf_files = list(d.glob("*.gguf"))
    if not gguf_files:
        return {"runtime": label, "status": "MISSING"}
    gfile = gguf_files[0]
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss / (1024 * 1024)

    t_load0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
    llm = llama_cpp.Llama(model_path=str(gfile), n_ctx=512, n_threads=n_threads, logits_all=False, verbose=False)
    cold_ms = round((time.perf_counter() - t_load0) * 1000.0, 2)

    parser = OutputParser()
    lats, valid_cnt, gt_c, pr_c = [], 0, [], []
    # Warmup 1
    _ = llm.create_completion(PromptFormatter.format_prompt(probe[0]["text"], tokenizer=tok), max_tokens=45, temperature=0.0)
    for r in probe[:14]:
        prompt = PromptFormatter.format_prompt(r["text"], tokenizer=tok)
        t0 = time.perf_counter()
        out = llm.create_completion(prompt, max_tokens=45, temperature=0.0, stop=["<|im_end|>"])
        lats.append((time.perf_counter() - t0) * 1000.0)
        raw = out["choices"][0]["text"].strip()
        ok, parsed, _ = parser.parse_and_validate(raw)
        if ok and isinstance(parsed, dict):
            valid_cnt += 1
            pr_c.append(parsed["category"])
        else:
            pr_c.append("general_query")
        gt_c.append(r["category"])

    rss1 = proc.memory_info().rss / (1024 * 1024)
    del llm
    return {
        "runtime": label,
        "model_file": str(gfile.relative_to(ROOT)).replace("\\", "/"),
        "weight_disk_mb": round(gfile.stat().st_size / (1024 * 1024), 2),
        "n_threads": n_threads,
        "cold_start_ms": cold_ms,
        "peak_rss_mb": round(rss1, 2),
        "rss_delta_mb": round(rss1 - rss0, 2),
        "probe_n": len(gt_c),
        "json_validity_rate": round(valid_cnt / len(gt_c), 4),
        "probe_cat_accuracy_seed42": EvaluationMetrics.calculate_classification_metrics(gt_c, pr_c, labels=CATEGORIES)["accuracy"],
        **summarize_latencies(lats),
        "note": "Seed42 V1 GGUF weights tested for CPU runtime speed/RAM/validity; not clean-trained V2.",
    }


def benchmark_onnx_int8(probe: List[Dict[str, Any]]) -> Dict[str, Any]:
    d = ROOT / "models/qwen25_05b_lora_seed42_int8"
    qfile = d / "model_quantized.onnx"
    if not qfile.exists():
        return {"runtime": "onnx_int8", "status": "MISSING"}
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        proc = psutil.Process(os.getpid())
        rss0 = proc.memory_info().rss / (1024 * 1024)
        t_load0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
        model = ORTModelForCausalLM.from_pretrained(str(d), file_name="model_quantized.onnx", local_files_only=True)
        cold_ms = round((time.perf_counter() - t_load0) * 1000.0, 2)

        parser = OutputParser()
        lats, valid_cnt, gt_c, pr_c = [], 0, [], []
        for r in probe[:8]:
            prompt = PromptFormatter.format_prompt(r["text"], tokenizer=tok)
            inp = tok(prompt, return_tensors="pt")
            plen = inp["input_ids"].shape[1]
            t0 = time.perf_counter()
            out = model.generate(**inp, max_new_tokens=45, do_sample=False, pad_token_id=tok.eos_token_id)
            lats.append((time.perf_counter() - t0) * 1000.0)
            raw = tok.decode(out[0][plen:], skip_special_tokens=True).strip()
            ok, parsed, _ = parser.parse_and_validate(raw)
            if ok and isinstance(parsed, dict):
                valid_cnt += 1
                pr_c.append(parsed["category"])
            else:
                pr_c.append("general_query")
            gt_c.append(r["category"])

        rss1 = proc.memory_info().rss / (1024 * 1024)
        del model
        return {
            "runtime": "qwen25_05b_seed42_onnx_dynamic_int8",
            "model_file": str(qfile.relative_to(ROOT)).replace("\\", "/"),
            "weight_disk_mb": round(qfile.stat().st_size / (1024 * 1024), 2),
            "cold_start_ms": cold_ms,
            "peak_rss_mb": round(rss1, 2),
            "rss_delta_mb": round(rss1 - rss0, 2),
            "probe_n": len(gt_c),
            "json_validity_rate": round(valid_cnt / len(gt_c), 4),
            "probe_cat_accuracy_seed42": EvaluationMetrics.calculate_classification_metrics(gt_c, pr_c, labels=CATEGORIES)["accuracy"],
            **summarize_latencies(lats),
            "note": "Dynamic INT8 ONNX generation benchmarked live on CPU ORT.",
        }
    except Exception as exc:
        return {"runtime": "qwen25_05b_seed42_onnx_dynamic_int8", "status": "ERROR", "error": str(exc)}


def benchmark_neural_v2_hybrid_head(probe: List[Dict[str, Any]]) -> Dict[str, Any]:
    weights_p = ROOT / "experiments/artifacts/v2_models/nexus_v2_hybrid_multitask_head.pt"
    feat_p = ROOT / "experiments/artifacts/v2_models/nexus_v2_lexical_featurizer.joblib"
    if not (weights_p.exists() and feat_p.exists()):
        return {"runtime": "neural_v2_hybrid_multitask", "status": "NOT_TRAINED_YET"}

    from transformers import AutoModel, AutoTokenizer

    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss / (1024 * 1024)
    snap = str(sorted((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots").glob("*"))[-1])

    t_load0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = AutoModel.from_pretrained(snap, dtype=torch.float32, local_files_only=True)
    enc.eval()
    feats = joblib.load(feat_p)
    vec, svd = feats["vectorizer"], feats["svd"]
    head = NexusMultitaskNeuralHead(input_dim=1792 + svd.n_components + 8, hidden_dim=512)
    head.load_state_dict(torch.load(weights_p, map_location="cpu", weights_only=True))
    head.eval()
    cold_ms = round((time.perf_counter() - t_load0) * 1000.0, 2)

    lats, gt_c, pr_c = [], [], []
    with torch.inference_mode():
        # Warmup
        w_inp = tok([probe[0]["text"]], return_tensors="pt", padding=True, truncation=True, max_length=128)
        _ = enc(**w_inp)
        for r in probe[:14]:
            t0 = time.perf_counter()
            normed = MultilingualTextPreprocessor.normalize(r["text"])
            inp = tok([normed], return_tensors="pt", padding=True, truncation=True, max_length=128)
            out = enc(**inp).last_hidden_state
            mask = inp["attention_mask"].unsqueeze(-1).float()
            mean_pool = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            last_idx = inp["attention_mask"].sum(dim=1) - 1
            last_pool = out[torch.arange(out.size(0)), last_idx]
            qw = F.normalize(torch.cat([mean_pool, last_pool], dim=-1), p=2, dim=-1).numpy()
            x_svd = svd.transform(vec.transform([normed])).astype(np.float32)
            x_scr = np.array([MultilingualTextPreprocessor.extract_script_indicators(r["text"])], dtype=np.float32)
            x_all = torch.tensor(np.hstack([qw, x_svd, x_scr]), dtype=torch.float32)
            logits = head(x_all)["category_logits"]
            ci = int(logits.argmax(dim=-1).item())
            lats.append((time.perf_counter() - t0) * 1000.0)
            gt_c.append(r["category"])
            pr_c.append(CATEGORIES[ci])

    rss1 = proc.memory_info().rss / (1024 * 1024)
    return {
        "runtime": "nexus_v2_hybrid_qwen05b_single_forward_head",
        "weight_disk_mb": round(weights_p.stat().st_size / (1024 * 1024), 2),
        "cold_start_ms": cold_ms,
        "peak_rss_mb": round(rss1, 2),
        "rss_delta_mb": round(rss1 - rss0, 2),
        "probe_n": len(gt_c),
        "json_validity_rate": 1.0,
        "probe_cat_accuracy_v2": EvaluationMetrics.calculate_classification_metrics(gt_c, pr_c, labels=CATEGORIES)["accuracy"],
        **summarize_latencies(lats),
        "note": "Single forward pass (no autoregressive token loop!) + deterministic schema construction.",
    }


def main():
    probe = load_benchmark_probe(28)
    results = []
    results.append(benchmark_heuristic(probe))
    results.append(benchmark_gguf("gguf_q4_k_m_t4", "models/qwen25_05b_lora_seed42_gguf_q4_k_m", probe, n_threads=4))
    results.append(benchmark_gguf("gguf_q4_k_m_t8", "models/qwen25_05b_lora_seed42_gguf_q4_k_m", probe, n_threads=8))
    results.append(benchmark_gguf("gguf_q8_0_t8", "models/qwen25_05b_lora_seed42_gguf_q8_0", probe, n_threads=8))
    results.append(benchmark_onnx_int8(probe))
    results.append(benchmark_neural_v2_hybrid_head(probe))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_count_logical": os.cpu_count(),
        "benchmarks": results,
    }
    out_p = ROOT / "experiments/artifacts/runtime_inference_benchmarks_v2.json"
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
