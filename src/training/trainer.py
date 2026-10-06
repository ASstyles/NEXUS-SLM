"""Supervised QLoRA / LoRA fine-tuning pipeline for NEXUS-SLM."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from src.models.prompt_formatter import PromptFormatter
from src.training.tracker import ExperimentTracker
from src.utils.config import load_yaml_config
from src.utils.hardware import get_hardware_info
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NexusTrainer:
    """Orchestrates QLoRA fine-tuning of 1B-3B SLMs for structured support triage."""

    def __init__(self, config_path: str = "configs/training/qlora_default.yaml"):
        self.config_path = config_path
        self.config = load_yaml_config(config_path)
        self.tracker = ExperimentTracker()

    def _prepare_dataset(self, jsonl_path: str, tokenizer: Any) -> Any:
        """Loads and formats jsonl dataset records into tokenized inputs."""
        from datasets import Dataset

        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        formatted_texts = []
        for r in records:
            target_obj = {
                "category": r["category"],
                "priority": r["priority"],
                "language": r["language"],
                "requires_human": r["requires_human"],
            }
            target_json = json.dumps(target_obj, ensure_ascii=False)
            full_prompt = PromptFormatter.format_training_example(
                text=r["text"],
                target_json_str=target_json,
                tokenizer=tokenizer,
            )
            formatted_texts.append(full_prompt)

        return Dataset.from_dict({"text": formatted_texts})

    def train(self) -> Dict[str, Any]:
        """Executes QLoRA training workflow."""
        start_time = time.time()
        hw_info = get_hardware_info()
        logger.info(f"Initiating training. Hardware: {hw_info['os']}, CUDA={hw_info['cuda_available']}")

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model_cfg = self.config.get("model", {})
        quant_cfg = self.config.get("quantization", {})
        lora_cfg = self.config.get("lora", {})
        train_cfg = self.config.get("training", {})
        export_cfg = self.config.get("export", {})

        model_id = model_cfg.get("pretrained_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")

        # 1. Start Experiment Tracking
        run_name = self.config.get("experiment_name", "nexus-qlora")
        self.tracker.start_run(
            run_name=run_name,
            tags={"model": model_id, "mode": "qlora"},
        )
        self.tracker.log_params({
            "model_id": model_id,
            "lora_r": lora_cfg.get("r", 16),
            "lora_alpha": lora_cfg.get("lora_alpha", 32),
            "learning_rate": train_cfg.get("learning_rate", 2e-4),
            "epochs": train_cfg.get("num_train_epochs", 3),
            "batch_size": train_cfg.get("per_device_train_batch_size", 2),
            "quantization": "4bit" if quant_cfg.get("load_in_4bit") else "fp16",
        })

        # 2. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 3. Model Loading & Quantization
        bnb_config = None
        if hw_info["cuda_available"] and hw_info["bitsandbytes_available"] and quant_cfg.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=torch.float16,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto" if hw_info["cuda_available"] else None,
            torch_dtype=torch.float16 if hw_info["cuda_available"] else torch.float32,
            trust_remote_code=True,
        )

        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)

        # 4. LoRA Adapter Configuration
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # 5. Datasets
        train_path = "data/train/train.jsonl"
        val_path = "data/validation/validation.jsonl"
        train_dataset = self._prepare_dataset(train_path, tokenizer)
        val_dataset = self._prepare_dataset(val_path, tokenizer)

        def tokenize_fn(batch):
            return tokenizer(batch["text"], truncation=True, max_length=train_cfg.get("max_seq_length", 512))

        tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        # 6. Training Arguments
        output_dir = train_cfg.get("output_dir", "./models/checkpoints/qlora-default")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            logging_steps=train_cfg.get("logging_steps", 10),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            fp16=(hw_info["cuda_available"] and not torch.cuda.is_bf16_supported()),
            bf16=(hw_info["cuda_available"] and torch.cuda.is_bf16_supported()),
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
            report_to=["none"],  # MLflow handled via custom tracker
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        logger.info("Starting model training execution...")
        train_result = trainer.train()

        duration = round(time.time() - start_time, 2)
        metrics = {
            "train_loss": train_result.training_loss,
            "train_duration_sec": duration,
        }

        # 7. Model Export
        export_dir = Path(export_cfg.get("export_dir", "./models/nexus-v1.0"))
        export_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(export_dir))
        tokenizer.save_pretrained(str(export_dir))

        # Save metadata
        metadata = {
            "model_version": "nexus-v1.0",
            "base_model": model_id,
            "lora_r": lora_cfg.get("r", 16),
            "lora_alpha": lora_cfg.get("lora_alpha", 32),
            "train_loss": train_result.training_loss,
            "duration_sec": duration,
            "hardware": hw_info,
        }
        with open(export_dir / "nexus_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.tracker.log_metrics(metrics)
        self.tracker.end_run(
            run_name=run_name,
            params=self.config,
            metrics=metrics,
            status="FINISHED",
        )

        logger.info(f"Training completed successfully in {duration}s! Exported to {export_dir}")
        return metrics


def main():
    trainer = NexusTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
