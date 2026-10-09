"""Executes Step 5: Genuine LoRA fine-tuning for NEXUS-SLM on dataset_v1."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import psutil
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.prompt_formatter import PromptFormatter
from src.utils.config import load_yaml_config

CONFIG_PATH = "configs/training/qlora_step5.yaml"
OUT_DIR = Path("experiments/exp_002_qlora")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== STEP 5: GENUINE LoRA FINE-TUNING EXECUTION ===")
cfg = load_yaml_config(CONFIG_PATH)

model_id = cfg["model"]["pretrained_model_name_or_path"]
lora_cfg = cfg["lora"]
train_cfg = cfg["training"]
export_dir = Path(cfg["export"]["export_dir"])
export_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Tokenizer
print(f"Loading tokenizer: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare Datasets
def prepare_jsonl(jsonl_path: str):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    formatted = []
    for r in records:
        target_obj = {
            "category": r["category"],
            "priority": r["priority"],
            "language": r["language"],
            "requires_human": r["requires_human"],
        }
        target_str = json.dumps(target_obj, ensure_ascii=False)
        full_text = PromptFormatter.format_training_example(
            text=r["text"],
            target_json_str=target_str,
            tokenizer=tokenizer,
        )
        formatted.append(full_text)
    return Dataset.from_dict({"text": formatted})

print("Preparing train and validation datasets...")
train_dataset = prepare_jsonl("data/train/train.jsonl")
val_dataset = prepare_jsonl("data/validation/validation.jsonl")

max_len = train_cfg.get("max_seq_length", 128)
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=max_len)

tok_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tok_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"Dataset tokenized: Train={len(tok_train)}, Val={len(tok_val)}")

# 3. Load Model & Attach LoRA
print("Loading base model in Float32 on CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# Attach PEFT LoRA
peft_config = LoraConfig(
    r=lora_cfg.get("r", 16),
    lora_alpha=lora_cfg.get("lora_alpha", 32),
    target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
    lora_dropout=lora_cfg.get("lora_dropout", 0.05),
    bias=lora_cfg.get("bias", "none"),
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, peft_config)

# Verify Frozen vs Trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
trainable_percent = (trainable_params / total_params) * 100

print("\n" + "="*60)
print("PEFT LoRA PARAMETER VERIFICATION:")
print(f"  Total Parameters:     {total_params:,}")
print(f"  Trainable Parameters: {trainable_params:,} ({trainable_percent:.3f}%)")
print(f"  Frozen Parameters:    {frozen_params:,} ({100 - trainable_percent:.3f}%)")
print("="*60)

assert frozen_params > 0, "Base parameters MUST be frozen!"
assert trainable_params > 0, "LoRA adapter MUST be trainable!"

# 4. Training Arguments
checkpoint_dir = Path(train_cfg.get("output_dir", "./models/checkpoints/exp_002_qlora"))
checkpoint_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(checkpoint_dir),
    num_train_epochs=train_cfg.get("num_train_epochs", 2),
    per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
    per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
    gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 2),
    learning_rate=train_cfg.get("learning_rate", 3e-4),
    weight_decay=train_cfg.get("weight_decay", 0.01),
    warmup_steps=10,
    lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
    logging_steps=train_cfg.get("logging_steps", 10),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    fp16=False,
    bf16=False,
    gradient_checkpointing=False,
    report_to=["none"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# 5. Execute Genuine Fine-Tuning
start_time = time.time()
print(f"\nInitiating training for {train_cfg.get('num_train_epochs', 2)} epochs on CPU...")
train_result = trainer.train()
duration_sec = round(time.time() - start_time, 2)

# 6. Evaluation on Validation Set
val_metrics = trainer.evaluate()
print(f"Validation loss: {val_metrics.get('eval_loss', 0.0):.4f}")

# 7. Save Model Adapter
print(f"Saving fine-tuned adapter to {export_dir}...")
model.save_pretrained(str(export_dir))
tokenizer.save_pretrained(str(export_dir))

# 8. Log Training Results
training_log = {
    "experiment_id": "exp_002_qlora",
    "timestamp": datetime.utcnow().isoformat(),
    "model_id": model_id,
    "lora_config": {
        "r": lora_cfg.get("r", 16),
        "lora_alpha": lora_cfg.get("lora_alpha", 32),
        "target_modules": lora_cfg.get("target_modules"),
        "dropout": lora_cfg.get("lora_dropout", 0.05),
    },
    "parameter_counts": {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": round(trainable_percent, 4),
    },
    "training_hyperparameters": {
        "epochs": train_cfg.get("num_train_epochs", 2),
        "batch_size": train_cfg.get("per_device_train_batch_size", 4),
        "grad_accum_steps": train_cfg.get("gradient_accumulation_steps", 2),
        "effective_batch_size": train_cfg.get("per_device_train_batch_size", 4) * train_cfg.get("gradient_accumulation_steps", 2),
        "learning_rate": train_cfg.get("learning_rate", 3e-4),
        "lr_scheduler": train_cfg.get("lr_scheduler_type", "cosine"),
        "max_seq_length": max_len,
    },
    "training_metrics": {
        "train_loss": round(float(train_result.training_loss), 4),
        "eval_loss": round(float(val_metrics.get("eval_loss", 0.0)), 4),
        "global_steps": train_result.global_step,
        "duration_seconds": duration_sec,
    },
    "status": "COMPLETED",
}

with open(OUT_DIR / "training_log.json", "w", encoding="utf-8") as f:
    json.dump(training_log, f, indent=2)

with open(OUT_DIR / "config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)

print("\n" + "="*60)
print("LoRA TRAINING COMPLETED SUCCESSFULLY!")
print(f"  Duration: {duration_sec}s")
print(f"  Final Train Loss: {training_log['training_metrics']['train_loss']}")
print(f"  Validation Loss: {training_log['training_metrics']['eval_loss']}")
print(f"  Exported to: {export_dir}")
print("="*60)
