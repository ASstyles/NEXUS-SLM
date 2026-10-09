"""Evaluates confidence calibration on the locked test set for NEXUS-SLM-135M-LoRA."""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluation.calibration import ConfidenceCalibrator
from src.inference.engine import InferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    test_path = Path("data/test/test.jsonl")
    adapter_path = Path("models/nexus_qlora_v1")

    logger.info("Loading locked test dataset...")
    test_samples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))

    logger.info(f"Loaded {len(test_samples)} test samples.")

    logger.info("Loading candidate model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM-135M-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    engine = InferenceEngine(model=model, tokenizer=tokenizer, model_version="nexus-qlora-v1")

    confidences = []
    category_correct = []
    exact_match_correct = []
    detailed_records = []

    logger.info("Running inference across test samples to collect confidence scores...")
    for idx, sample in enumerate(test_samples):
        text = sample["text"]
        true_cat = sample["category"]
        true_prio = sample["priority"]
        true_lang = sample["language"]

        pred = engine.predict(text)
        conf = float(pred.confidence)
        confidences.append(conf)

        is_cat_correct = (pred.category == true_cat)
        is_exact = (
            pred.category == true_cat
            and pred.priority == true_prio
            and pred.language == true_lang
        )

        category_correct.append(is_cat_correct)
        exact_match_correct.append(is_exact)

        detailed_records.append({
            "id": sample.get("id", idx),
            "text": text[:60] + "...",
            "true_category": true_cat,
            "pred_category": pred.category,
            "category_correct": is_cat_correct,
            "exact_match_correct": is_exact,
            "confidence": conf,
            "requires_human": pred.requires_human,
        })

    calibrator = ConfidenceCalibrator(num_bins=5)
    cal_metrics = calibrator.compute_calibration_metrics(
        confidences=confidences,
        correctness=category_correct,
    )

    plot_path = "experiments/artifacts/confidence_calibration.png"
    calibrator.plot_reliability_diagram(cal_metrics, output_path=plot_path)

    out_json = Path("experiments/artifacts/confidence_calibration.json")
    full_output = {
        "candidate_model": "nexus-qlora-v1",
        "dataset": "data/test/test.jsonl",
        "calibration_metrics": cal_metrics,
        "exact_match_accuracy": round(sum(exact_match_correct) / len(exact_match_correct), 4),
        "detailed_sample_records": detailed_records,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    logger.info(f"Calibration evaluation saved to {out_json}")
    logger.info(f"Expected Calibration Error (ECE): {cal_metrics['expected_calibration_error']}")
    logger.info(f"Mean Confidence: {cal_metrics['mean_confidence']} vs Mean Accuracy: {cal_metrics['mean_accuracy']}")
    logger.info(f"Overconfidence Gap: {cal_metrics['overconfidence_gap']}")
    logger.info(f"Policy recommendation: {cal_metrics['recommendation']}")


if __name__ == "__main__":
    main()
