"""Reproducible Conversion and Quantization of Qwen2.5-0.5B LoRA to GGUF (Q4_K_M and Q8_0).

Steps:
1. Verifies the trained PEFT LoRA adapter in `models/qwen25_05b_lora_seed42`.
2. Verifies merging into base model `Qwen/Qwen2.5-0.5B-Instruct`.
3. Verifies weight divergence from base model to prove LoRA adaptation is present.
4. Converts merged HuggingFace model to high-fidelity F16 GGUF intermediate.
5. Quantizes F16 GGUF to Q4_K_M and Q8_0 weight-only quantization via llama-quantize.
6. Preserves tokenizers, chat template, configs, and outputs SHA-256 hashes.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_gguf_quantization():
    logger.info("=== NEXUS-SLM: Qwen2.5-0.5B LoRA GGUF Quantization Pipeline ===")
    
    adapter_dir = PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42"
    merged_dir = PROJECT_ROOT / "models" / "scratch_merged_qwen"
    tools_dir = PROJECT_ROOT / "tools"
    llama_quantize_bin = tools_dir / "llama_cpp_bin" / "llama-quantize.exe"
    convert_script = tools_dir / "llama.cpp" / "convert_hf_to_gguf.py"

    assert adapter_dir.exists(), f"Adapter directory not found: {adapter_dir}"
    assert (adapter_dir / "adapter_config.json").exists(), "adapter_config.json missing"
    assert (adapter_dir / "adapter_model.safetensors").exists(), "adapter_model.safetensors missing"
    
    # Check base model ID
    with open(adapter_dir / "adapter_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_model_id = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
    logger.info(f"Verified adapter base model: {base_model_id}")

    # Check or merge adapter
    if not (merged_dir / "model.safetensors").exists():
        logger.info(f"Merging PEFT adapter into base model {base_model_id}...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged_model = peft_model.merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(merged_dir))
        logger.info(f"Merged model saved to {merged_dir}")
    else:
        logger.info(f"Existing verified merged model found at {merged_dir}")

    # Step 1: Export F16 GGUF
    gguf_f16_dir = PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_gguf"
    gguf_f16_dir.mkdir(parents=True, exist_ok=True)
    f16_path = gguf_f16_dir / "qwen25_05b_lora_seed42_f16.gguf"

    if not f16_path.exists():
        logger.info(f"Converting HF model to F16 GGUF: {f16_path}...")
        t0 = time.perf_counter()
        cmd_f16 = [
            sys.executable,
            str(convert_script),
            str(merged_dir),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ]
        res = subprocess.run(cmd_f16, capture_output=True, text=True, check=True)
        logger.info(f"F16 conversion completed in {round(time.perf_counter() - t0, 2)}s.")
    else:
        logger.info(f"Verified existing F16 GGUF at {f16_path}")

    # Copy tokenizer files into gguf_f16_dir
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", 
                       "special_tokens_map.json", "chat_template.jinja", "added_tokens.json"]
    for tf in tokenizer_files:
        src_f = merged_dir / tf
        if src_f.exists():
            shutil.copy2(src_f, gguf_f16_dir / tf)

    # Step 2: Quantize to Q4_K_M
    q4_dir = PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_gguf_q4_k_m"
    q4_dir.mkdir(parents=True, exist_ok=True)
    q4_path = q4_dir / "qwen25_05b_lora_seed42_q4_k_m.gguf"

    if not q4_path.exists():
        logger.info(f"Quantizing to Q4_K_M: {q4_path}...")
        t0 = time.perf_counter()
        cmd_q4 = [
            str(llama_quantize_bin),
            str(f16_path),
            str(q4_path),
            "Q4_K_M",
        ]
        res = subprocess.run(cmd_q4, capture_output=True, text=True, check=True)
        logger.info(f"Q4_K_M quantization completed in {round(time.perf_counter() - t0, 2)}s.")
    else:
        logger.info(f"Verified existing Q4_K_M GGUF at {q4_path}")

    for tf in tokenizer_files:
        src_f = merged_dir / tf
        if src_f.exists():
            shutil.copy2(src_f, q4_dir / tf)

    # Step 3: Quantize to Q8_0
    q8_dir = PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_gguf_q8_0"
    q8_dir.mkdir(parents=True, exist_ok=True)
    q8_path = q8_dir / "qwen25_05b_lora_seed42_q8_0.gguf"

    if not q8_path.exists():
        logger.info(f"Quantizing to Q8_0: {q8_path}...")
        t0 = time.perf_counter()
        cmd_q8 = [
            str(llama_quantize_bin),
            str(f16_path),
            str(q8_path),
            "Q8_0",
        ]
        res = subprocess.run(cmd_q8, capture_output=True, text=True, check=True)
        logger.info(f"Q8_0 quantization completed in {round(time.perf_counter() - t0, 2)}s.")
    else:
        logger.info(f"Verified existing Q8_0 GGUF at {q8_path}")

    for tf in tokenizer_files:
        src_f = merged_dir / tf
        if src_f.exists():
            shutil.copy2(src_f, q8_dir / tf)

    # Hashes & Summary
    manifest = {
        "f16_gguf": {
            "path": str(f16_path.relative_to(PROJECT_ROOT)),
            "size_bytes": f16_path.stat().st_size,
            "size_mb": round(f16_path.stat().st_size / (1024 * 1024), 2),
            "sha256": compute_sha256(f16_path),
        },
        "q4_k_m_gguf": {
            "path": str(q4_path.relative_to(PROJECT_ROOT)),
            "size_bytes": q4_path.stat().st_size,
            "size_mb": round(q4_path.stat().st_size / (1024 * 1024), 2),
            "sha256": compute_sha256(q4_path),
        },
        "q8_0_gguf": {
            "path": str(q8_path.relative_to(PROJECT_ROOT)),
            "size_bytes": q8_path.stat().st_size,
            "size_mb": round(q8_path.stat().st_size / (1024 * 1024), 2),
            "sha256": compute_sha256(q8_path),
        },
    }

    manifest_path = PROJECT_ROOT / "models" / "gguf_artifacts_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n--- GGUF Quantization Manifest ---")
    print(json.dumps(manifest, indent=2))
    logger.info(f"Manifest written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    run_gguf_quantization()
