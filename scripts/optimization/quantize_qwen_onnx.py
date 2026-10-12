"""Reproducible ONNX Export & Dynamic INT8 Quantization for Qwen2.5-0.5B LoRA Candidate.

Pipeline:
1. Loads base model (Qwen/Qwen2.5-0.5B-Instruct) and trained LoRA adapter (models/qwen25_05b_lora_seed42).
2. Merges LoRA adapter into base weights without modifying original adapter directory.
3. Exports merged causal LM to ONNX with dynamic KV-caching.
4. Quantizes ONNX weights to dynamic INT8 (avx512_vnni / avx2 compatible).
5. Exports complete standalone candidate bundle to models/qwen25_05b_lora_seed42_int8/.
6. Computes and records SHA-256 hashes for all output artifacts.
7. Packages candidate archive as nexus_qwen25_05b_lora_seed42_int8.zip.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_quantization(
    adapter_dir: Path,
    output_dir: Path,
    base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> Dict[str, any]:
    print("=" * 75)
    print("NEXUS-SLM: QWEN2.5-0.5B ONNX INT8 QUANTIZATION PIPELINE")
    print("=" * 75)
    
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Adapter weights not found in {adapter_dir}")
        
    scratch_dir = PROJECT_ROOT / "scratch" / "quant_build"
    shutil.rmtree(scratch_dir, ignore_errors=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    merged_dir = scratch_dir / "merged"
    onnx_fp32_dir = scratch_dir / "onnx_fp32"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    t_start = time.perf_counter()
    
    # ---------------------------------------------------------
    # Stage 1: Load and Merge LoRA Adapter
    # ---------------------------------------------------------
    print(f"\n[Stage 1/5] Merging trained LoRA adapter from {adapter_dir}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()
    
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"  Merged model saved in {time.perf_counter() - t0:.2f}s")
    
    # ---------------------------------------------------------
    # Stage 2: Export Merged Model to ONNX
    # ---------------------------------------------------------
    print(f"\n[Stage 2/5] Exporting merged causal LM to ONNX with KV-cache...")
    t1 = time.perf_counter()
    ort_fp32 = ORTModelForCausalLM.from_pretrained(str(merged_dir), export=True)
    onnx_fp32_dir.mkdir(parents=True, exist_ok=True)
    ort_fp32.save_pretrained(str(onnx_fp32_dir))
    tokenizer.save_pretrained(str(onnx_fp32_dir))
    print(f"  ONNX FP32 export completed in {time.perf_counter() - t1:.2f}s")
    
    # ---------------------------------------------------------
    # Stage 3: Dynamic INT8 Quantization
    # ---------------------------------------------------------
    print(f"\n[Stage 3/5] Applying dynamic INT8 quantization via ONNX Runtime...")
    t2 = time.perf_counter()
    quantizer = ORTQuantizer.from_pretrained(str(onnx_fp32_dir))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    
    quant_temp_dir = scratch_dir / "quant_temp"
    quant_temp_dir.mkdir(parents=True, exist_ok=True)
    quantizer.quantize(save_dir=str(quant_temp_dir), quantization_config=qconfig)
    print(f"  INT8 quantization completed in {time.perf_counter() - t2:.2f}s")
    
    # ---------------------------------------------------------
    # Stage 4: Assemble Final Standalone Candidate Directory
    # ---------------------------------------------------------
    print(f"\n[Stage 4/5] Assembling final candidate files into {output_dir}...")
    
    # Copy quantized onnx model and configs
    for item in quant_temp_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, output_dir / item.name)
            
    # Ensure tokenizer files and configs are present
    tokenizer.save_pretrained(str(output_dir))
    
    # Record file sizes and SHA-256 hashes
    artifact_hashes = {}
    total_bytes = 0
    for f in sorted(output_dir.iterdir()):
        if f.is_file() and not f.name.endswith(".zip"):
            h = compute_sha256(f)
            artifact_hashes[f.name] = h
            sz = f.stat().st_size
            total_bytes += sz
            print(f"  {f.name:<30}: {h} ({sz:,} bytes)")
            
    quantized_size_mb = round(total_bytes / (1024**2), 2)
    print(f"  Total Quantized Artifact Size: {quantized_size_mb} MB")
    
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "candidate_version": "qwen25_05b_lora_seed42_int8",
        "base_model": base_model_id,
        "source_adapter_dir": str(adapter_dir.relative_to(PROJECT_ROOT)),
        "quantization_format": "onnx_dynamic_int8",
        "runtime_engine": "ONNXRuntime",
        "target_hardware": "x86_64 CPU (AVX2 / AVX-512 VNNI)",
        "quantized_size_mb": quantized_size_mb,
        "conversion_duration_sec": round(time.perf_counter() - t_start, 2),
        "artifact_hashes": artifact_hashes,
    }
    
    meta_path = output_dir / "nexus_qwen_int8_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    artifact_hashes["nexus_qwen_int8_metadata.json"] = compute_sha256(meta_path)
    
    # ---------------------------------------------------------
    # Stage 5: Package Archive
    # ---------------------------------------------------------
    print(f"\n[Stage 5/5] Packaging candidate archive...")
    temp_zip_stem = scratch_dir / "nexus_qwen25_05b_lora_seed42_int8"
    archive_path = shutil.make_archive(str(temp_zip_stem), "zip", root_dir=str(output_dir))
    final_zip_path = output_dir / "nexus_qwen25_05b_lora_seed42_int8.zip"
    shutil.move(archive_path, final_zip_path)
    zip_hash = compute_sha256(final_zip_path)
    print(f"  Archive created: {final_zip_path}")
    print(f"  Archive SHA-256: {zip_hash} ({final_zip_path.stat().st_size:,} bytes)")
    
    # Cleanup scratch
    print(f"\nCleaning temporary build directories...")
    shutil.rmtree(scratch_dir, ignore_errors=True)
    shutil.rmtree(PROJECT_ROOT / "models" / "scratch_merged_qwen", ignore_errors=True)
    shutil.rmtree(PROJECT_ROOT / "models" / "scratch_onnx_qwen", ignore_errors=True)
    
    total_time = round(time.perf_counter() - t_start, 2)
    print(f"\nQuantization pipeline successfully completed in {total_time}s!")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX Dynamic INT8 Quantization for Qwen2.5-0.5B LoRA")
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="models/qwen25_05b_lora_seed42",
        help="Path to trained source adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/qwen25_05b_lora_seed42_int8",
        help="Path to destination quantized candidate directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model ID",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    adapter_p = PROJECT_ROOT / args.adapter_dir
    output_p = PROJECT_ROOT / args.output_dir
    run_quantization(adapter_dir=adapter_p, output_dir=output_p, base_model_id=args.base_model)
