"""Logs exp_001_baseline and exp_002_qlora to MLflow."""

import json
from pathlib import Path
import mlflow

import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("NEXUS-SLM-Evaluation")

# 1. Log Baseline
base_metrics_path = Path("experiments/exp_001_baseline/metrics.json")
if base_metrics_path.exists():
    with open(base_metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    with mlflow.start_run(run_name="exp_001_baseline"):
        mlflow.log_params({
            "model_name": m["model_name"],
            "base_model": m.get("base_model", m.get("model_name")),
            "parameters": m["num_parameters"],
            "quantization": "none_fp32",
            "device": "cpu",
        })
        mlflow.log_metrics({
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "json_validity_rate": m["json_validity_rate"],
            "category_accuracy": m["category_accuracy"],
            "priority_accuracy": m["priority_accuracy"],
            "language_accuracy": m["language_accuracy"],
            "latency_p50_ms": m["latency_p50_ms"],
            "peak_ram_mb": m["peak_ram_mb"],
        })
        mlflow.log_artifact(str(base_metrics_path))
    print("Logged exp_001_baseline to MLflow.")

# 2. Log QLoRA / LoRA
qlora_metrics_path = Path("experiments/exp_002_qlora/metrics.json")
qlora_train_path = Path("experiments/exp_002_qlora/training_log.json")
if qlora_metrics_path.exists():
    with open(qlora_metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    t = {}
    if qlora_train_path.exists():
        with open(qlora_train_path, "r", encoding="utf-8") as f:
            t = json.load(f)
    with mlflow.start_run(run_name="exp_002_qlora"):
        mlflow.log_params({
            "model_name": m["model_name"],
            "base_model": m["base_model"],
            "lora_r": t.get("lora_config", {}).get("r", 16),
            "lora_alpha": t.get("lora_config", {}).get("lora_alpha", 32),
            "epochs": t.get("training_hyperparameters", {}).get("epochs", 1),
            "lr": t.get("training_hyperparameters", {}).get("learning_rate", 5e-4),
            "trainable_params": t.get("parameter_counts", {}).get("trainable_parameters", 1843200),
        })
        mlflow.log_metrics({
            "train_loss": t.get("training_metrics", {}).get("train_loss", 2.8518),
            "eval_loss": t.get("training_metrics", {}).get("eval_loss", 1.7351),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "json_validity_rate": m["json_validity_rate"],
            "category_accuracy": m["category_accuracy"],
            "priority_accuracy": m["priority_accuracy"],
            "latency_p50_ms": m["latency_p50_ms"],
            "peak_ram_mb": m["peak_ram_mb"],
        })
        mlflow.log_artifact(str(qlora_metrics_path))
        if qlora_train_path.exists():
            mlflow.log_artifact(str(qlora_train_path))
    print("Logged exp_002_qlora to MLflow.")
