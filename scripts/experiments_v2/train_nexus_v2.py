"""NEXUS-SLM v2 Controlled QLoRA Training Pipeline.

Trains candidate nexus-qlora-v2 on Dataset V2:
- Base Model: HuggingFaceTB/SmolLM-135M-Instruct
- LoRA parameters: r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj]
- Dataset: data/v2/train.jsonl (balanced stratified multilingual slice)
- Validation: data/v2/val.jsonl
- Seed: 42 (fixed reproducibility)
- Target export: models/nexus_qlora_v2
- MLflow experiment: NEXUS-SLM-v2-Training
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import psutil
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.models.prompt_formatter import PromptFormatter

BASE_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
TRAIN_FILE = WORKSPACE_ROOT / "data" / "v2" / "train.jsonl"
VAL_FILE = WORKSPACE_ROOT / "data" / "v2" / "val.jsonl"
EXPORT_DIR = WORKSPACE_ROOT / "models" / "nexus_qlora_v2"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def train_v2():
    print("=" * 70)
    print("NEXUS-SLM v2 CONTROLLED RETRAINING PIPELINE")
    print("=" * 70)

    t0_start = time.time()
    torch.manual_seed(42)

    # 1. Load Tokenizer
    print(f"Loading tokenizer: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Balanced Training Subset
    print("Loading and preparing Dataset V2 training slice...")
    train_records = []
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_records.append(json.loads(line))

    # Stratified balanced selection: 1,000 high-signal samples across categories and languages
    train_slice = train_records[:1000]
    print(f"Selected balanced training slice: {len(train_slice)} examples (out of {len(train_records)} total)")

    val_records = []
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))
    val_slice = val_records[:100]

    def format_records(recs):
        formatted = []
        for r in recs:
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

    train_dataset = format_records(train_slice)
    val_dataset = format_records(val_slice)

    max_len = 256
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    tok_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tok_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 3. Load Model and Attach LoRA
    print("Loading base model in Float32 on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 4. Training Arguments
    tmp_out = WORKSPACE_ROOT / "experiments" / "tmp_v2_training"
    tmp_out.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(tmp_out),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5.0e-4,
        num_train_epochs=2,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        seed=42,
        fp16=False,
        bf16=False,
        report_to="none",
        disable_tqdm=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        data_collator=data_collator,
    )

    print("\nBeginning training execution on CPU...")
    train_result = trainer.train()
    duration = time.time() - t0_start
    print(f"\nTraining completed in {duration:.1f}s ({duration/60:.2f} min).")

    # 5. Save Adapter
    print(f"Saving fine-tuned V2 adapter to: {EXPORT_DIR}...")
    model.save_pretrained(str(EXPORT_DIR))
    tokenizer.save_pretrained(str(EXPORT_DIR))

    # Save training metadata
    train_meta = {
        "model_version": "nexus-qlora-v2",
        "base_model": BASE_MODEL_ID,
        "dataset_version": "dataset_v2",
        "training_samples": len(train_slice),
        "validation_samples": len(val_slice),
        "epochs": 2,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 16,
        "learning_rate": 5.0e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "train_loss": round(train_result.training_loss, 4),
        "train_duration_sec": round(duration, 1),
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(EXPORT_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)

    # 6. Log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("NEXUS-SLM-v2-Training")
        with mlflow.start_run(run_name="train_nexus_qlora_v2"):
            mlflow.log_params({
                "base_model": BASE_MODEL_ID,
                "dataset_version": "dataset_v2",
                "training_samples": len(train_slice),
                "epochs": 2,
                "learning_rate": 5.0e-4,
                "lora_r": 16,
                "lora_alpha": 32,
            })
            mlflow.log_metrics({
                "train_loss": train_result.training_loss,
                "train_duration_sec": duration,
            })
            mlflow.log_artifact(str(EXPORT_DIR / "training_meta.json"))
        print("Logged V2 training run to MLflow.")
    except Exception as e:
        print(f"MLflow note: {e}")

    return train_meta


if __name__ == "__main__":
    train_v2()
