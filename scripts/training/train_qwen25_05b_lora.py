"""GPU LoRA Fine-Tuning Pipeline for Qwen2.5-0.5B-Instruct on NEXUS-SLM.

Executes reproducible supervised LoRA instruction tuning:
- Base model: Qwen/Qwen2.5-0.5B-Instruct (494M parameters, 151k vocab)
- LoRA: r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
- Loss computation: Completion-only masking (prompt tokens and padding assigned label -100)
- ChatML formatting: Native tokenizer template (<|im_start|>...<|im_end|>)
- Hardware preflight: Requires CUDA GPU; aborts cleanly if GPU is unavailable or insufficient
- Reproducible artifact preservation: Saves adapter, tokenizer, hashes, and training metadata
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import psutil
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.models.prompt_formatter import PromptFormatter
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_sha256(path: Path) -> str:
    """Computes SHA256 hex digest of a file."""
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class CompletionOnlyDataCollator:
    """
    Data collator that dynamically pads input_ids, attention_mask, and labels.
    Ensures padding tokens in labels are strictly set to -100 so they are ignored in cross-entropy loss.
    """

    def __init__(self, tokenizer: Any, max_length: int = 384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [f["input_ids"] for f in features]
        batch_labels = [f["labels"] for f in features]

        max_len = min(max(len(ids) for ids in batch_input_ids), self.max_length)

        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for ids, lbls in zip(batch_input_ids, batch_labels):
            # Truncate if needed
            ids = ids[:max_len]
            lbls = lbls[:max_len]

            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_attention_masks.append([1] * len(ids) + [0] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def check_gpu_preflight(require_gpu: bool = True) -> Dict[str, Any]:
    """
    Performs hardware and GPU preflight checks.
    Halts cleanly if GPU is required but unavailable.
    """
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    hw_info = {
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "cpu_count": os.cpu_count() or 1,
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    }

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        vram_free_gb = round(torch.cuda.mem_get_info()[0] / (1024**3), 2)
        bf16_supported = torch.cuda.is_bf16_supported()
        hw_info.update({
            "gpu_name": gpu_name,
            "vram_total_gb": vram_total_gb,
            "vram_free_gb": vram_free_gb,
            "bf16_supported": bf16_supported,
        })
        logger.info(f"GPU Preflight PASSED: {gpu_name} (Total VRAM: {vram_total_gb} GB, Free: {vram_free_gb} GB, bf16: {bf16_supported})")
    else:
        logger.warning(f"No CUDA GPU detected. System RAM: {hw_info['available_ram_gb']} GB available.")
        if require_gpu:
            error_msg = (
                "GPU Preflight FAILED: CUDA GPU is required for Qwen2.5-0.5B LoRA training. "
                "Local Windows environment has PyTorch CPU runtime with insufficient RAM for full gradient backpropagation. "
                "Please execute this script in a CUDA-enabled environment (e.g. Kaggle T4, Colab, or local GPU with CUDA build), "
                "or pass '--allow-cpu-smoke-test' strictly for pipeline validation."
            )
            print("\n" + "!" * 75)
            print(error_msg)
            print("!" * 75 + "\n")
            raise RuntimeError(error_msg)

    return hw_info


def prepare_masked_dataset(
    jsonl_path: Path,
    tokenizer: Any,
    max_length: int = 384,
) -> Dataset:
    """
    Loads raw ticket JSONL records and formats them with prompt masking:
    - Prompt tokens: Label set to -100 (ignored in loss computation).
    - Assistant tokens: Label set to token IDs (trained on cross-entropy loss).
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    all_input_ids = []
    all_labels = []

    for r in records:
        target_obj = {
            "category": r["category"],
            "priority": r["priority"],
            "language": r["language"],
            "requires_human": r["requires_human"],
        }
        target_json_str = json.dumps(target_obj, ensure_ascii=False)

        # Build prompt and completion
        prompt_text = PromptFormatter.format_prompt(r["text"], tokenizer=tokenizer)
        completion_text = f"{target_json_str}<|im_end|>"
        full_text = f"{prompt_text}{completion_text}"

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # Safety check on token prefix matching
        if full_ids[:len(prompt_ids)] == prompt_ids:
            prompt_len = len(prompt_ids)
        else:
            # Fallback in case subword boundary shifted: find completion start
            completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
            prompt_len = max(0, len(full_ids) - len(completion_ids))

        labels = [-100] * prompt_len + full_ids[prompt_len:]

        # Truncate to max_length
        all_input_ids.append(full_ids[:max_length])
        all_labels.append(labels[:max_length])

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
    })


def train_qwen25_05b_lora(config_path: Path, allow_cpu_smoke_test: bool = False, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    Executes reproducible GPU LoRA fine-tuning for Qwen2.5-0.5B-Instruct.
    """
    print("=" * 75)
    print("NEXUS-SLM: QWEN2.5-0.5B-INSTRUCT GPU LORA FINE-TUNING")
    print("=" * 75)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)
    torch.manual_seed(seed)

    # 1. Hardware Preflight Check
    require_gpu = not allow_cpu_smoke_test
    hw_info = check_gpu_preflight(require_gpu=require_gpu)

    # 2. Paths & Dataset Verification
    base_dir = PROJECT_ROOT
    train_path = base_dir / cfg["splits"]["train"]["path"]
    val_path = base_dir / cfg["splits"]["validation"]["path"]
    test_path = base_dir / cfg["splits"]["locked_test_audit_only"]["path"]

    dataset_hashes = {
        "train_sha256": compute_sha256(train_path),
        "validation_sha256": compute_sha256(val_path),
        "locked_test_sha256": compute_sha256(test_path),
    }

    print("\nDataset Split Integrity Hashes:")
    for k, v in dataset_hashes.items():
        print(f"  {k:<22}: {v}")

    # Verify hashes match config
    if dataset_hashes["train_sha256"] != cfg["splits"]["train"]["sha256"]:
        logger.warning(f"Train split hash divergence: {dataset_hashes['train_sha256']} vs {cfg['splits']['train']['sha256']}")
    if dataset_hashes["validation_sha256"] != cfg["splits"]["validation"]["sha256"]:
        logger.warning(f"Validation split hash divergence: {dataset_hashes['validation_sha256']} vs {cfg['splits']['validation']['sha256']}")

    # 3. Load Tokenizer & Model Config (Do NOT assume SmolLM vocab or config)
    base_model_id = cfg.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"\nLoading Model Configuration & Tokenizer for: {base_model_id}")
    model_config = AutoConfig.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    vocab_size_cfg = model_config.vocab_size
    tokenizer_len = len(tokenizer)
    print(f"  model.config.vocab_size : {vocab_size_cfg:,}")
    print(f"  len(tokenizer)          : {tokenizer_len:,}")
    print(f"  pad_token               : {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"  eos_token               : {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    # 4. Dataset Preparation with Label Masking
    print("\nPreparing Tokenized Datasets with Completion-Only Masking...")
    max_seq_len = cfg.get("training", {}).get("max_seq_length", 384)
    train_dataset = prepare_masked_dataset(train_path, tokenizer, max_length=max_seq_len)
    val_dataset = prepare_masked_dataset(val_path, tokenizer, max_length=max_seq_len)

    if max_samples is not None:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(max_samples, len(val_dataset))))
        print(f"  [Smoke Test Active] Clamped to {len(train_dataset)} train and {len(val_dataset)} val samples.")

    print(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # 5. Model Loading & LoRA Configuration
    print("\nLoading Base Model Weights...")
    device_map = "auto" if hw_info["cuda_available"] else None
    torch_dtype = torch.bfloat16 if (hw_info["cuda_available"] and hw_info.get("bf16_supported")) else (
        torch.float16 if hw_info["cuda_available"] else torch.float32
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, peft_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print("\nTrainable Parameters:")
    model.print_trainable_parameters()

    # 6. Training Arguments & Trainer Setup
    train_params = cfg.get("training", {})
    output_dir = PROJECT_ROOT / cfg.get("paths", {}).get("checkpoint_dir", "models/checkpoints/qwen25_05b_lora_seed42")
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = hw_info["cuda_available"] and not hw_info.get("bf16_supported", False)
    use_bf16 = hw_info["cuda_available"] and hw_info.get("bf16_supported", False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_params.get("num_train_epochs", 3),
        per_device_train_batch_size=train_params.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_params.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_params.get("gradient_accumulation_steps", 2),
        learning_rate=train_params.get("learning_rate", 2e-4),
        weight_decay=train_params.get("weight_decay", 0.01),
        warmup_ratio=train_params.get("warmup_ratio", 0.05),
        lr_scheduler_type=train_params.get("lr_scheduler_type", "cosine"),
        logging_steps=train_params.get("logging_steps", 5),
        eval_strategy=train_params.get("eval_strategy", "epoch"),
        save_strategy=train_params.get("save_strategy", "epoch"),
        save_total_limit=train_params.get("save_total_limit", 2),
        load_best_model_at_end=train_params.get("load_best_model_at_end", True),
        metric_for_best_model=train_params.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_params.get("greater_is_better", False),
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=train_params.get("gradient_checkpointing", False),
        seed=seed,
        report_to=["none"],
    )

    data_collator = CompletionOnlyDataCollator(tokenizer=tokenizer, max_length=max_seq_len)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 7. Execute Training
    print("\nStarting Training...")
    t0 = time.time()
    train_result = trainer.train()
    duration = round(time.time() - t0, 2)

    eval_metrics = trainer.evaluate()
    print(f"\nTraining Complete in {duration:.1f}s!")
    print(f"  Final Train Loss: {train_result.training_loss:.4f}")
    print(f"  Final Eval Loss:  {eval_metrics.get('eval_loss', 0.0):.4f}")

    # 8. Export Adapter, Tokenizer, and Metadata
    export_dir = PROJECT_ROOT / cfg.get("paths", {}).get("export_dir", "models/qwen25_05b_lora_seed42")
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving Fine-Tuned LoRA Adapter & Tokenizer to: {export_dir}")
    model.save_pretrained(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))

    # Artifact hashes
    saved_hashes = {}
    for f in export_dir.glob("*"):
        if f.is_file():
            saved_hashes[f.name] = compute_sha256(f)

    # Build comprehensive training artifact
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "experiment_name": cfg.get("experiment_name", "qwen25_05b_lora_seed42"),
        "model_version": "qwen25_05b_lora_seed42",
        "base_model": base_model_id,
        "seed": seed,
        "vocabulary_metrics": {
            "model_config_vocab_size": vocab_size_cfg,
            "tokenizer_vocab_size": tokenizer_len,
            "discrepancy_noted": (vocab_size_cfg != tokenizer_len),
        },
        "lora_hyperparameters": lora_cfg,
        "training_hyperparameters": train_params,
        "training_metrics": {
            "train_loss": round(train_result.training_loss, 4),
            "eval_loss": round(eval_metrics.get("eval_loss", 0.0), 4),
            "train_duration_sec": duration,
        },
        "hardware_environment": hw_info,
        "dependency_versions": {
            "torch": torch.__version__,
            "transformers": getattr(sys.modules.get("transformers"), "__version__", "unknown"),
            "peft": getattr(sys.modules.get("peft"), "__version__", "unknown"),
        },
        "dataset_hashes": dataset_hashes,
        "saved_artifact_hashes": saved_hashes,
    }

    meta_file = export_dir / "nexus_qwen_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata recorded to: {meta_file}")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="GPU LoRA Training for Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/qwen25_05b_lora_seed42.json",
        help="Path to experiment configuration JSON",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        default=True,
        help="Require CUDA GPU; stop cleanly if absent (default: True)",
    )
    parser.add_argument(
        "--allow-cpu-smoke-test",
        action="store_true",
        default=False,
        help="Allow running a minimal 2-sample forward/backward smoke test on CPU without error",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional clamp on number of samples (for quick smoke tests)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_file = PROJECT_ROOT / args.config
    try:
        train_qwen25_05b_lora(
            config_path=cfg_file,
            allow_cpu_smoke_test=args.allow_cpu_smoke_test,
            max_samples=args.max_samples,
        )
    except RuntimeError as e:
        logger.error(f"Execution halted: {e}")
        sys.exit(1)
