"""Executes Step 9: Robustness evaluation on fine-tuned NEXUS-SLM."""

import json
import sys
import time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.robustness import RobustnessBenchmark, ROBUSTNESS_TEST_CASES
from src.inference.parser import OutputParser
from src.models.prompt_formatter import PromptFormatter
from src.utils.constants import CATEGORIES

BASE_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"
ADAPTER_PATH = Path("models/nexus_qlora_v1")
OUT_DIR = Path("experiments/exp_002_qlora")

print("=== STEP 9: ROBUSTNESS TEST SUITE EXECUTION ===")

# 1. Load fine-tuned model
print(f"Loading model + adapter from {ADAPTER_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
model.eval()

parser = OutputParser()

def predictor(text: str):
    prompt = PromptFormatter.format_prompt(text, tokenizer=tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    is_valid, data, _ = parser.parse_and_validate(raw)

    class PredObj:
        def __init__(self, c, v, r):
            self.category = c
            self.confidence = 1.0 if v else 0.0
            self.raw_output = r
    cat = data["category"] if (is_valid and data) else "unknown"
    return PredObj(cat, is_valid, raw)

# 2. Run Robustness Benchmark
rob = RobustnessBenchmark(output_dir="experiments/exp_002_qlora")
report = rob.run_benchmark(predictor)

# Group and calculate per-subset Macro-F1
subsets_table = []
subsets = sorted(list({tc["subset"] for tc in ROBUSTNESS_TEST_CASES}))

for sub in subsets:
    cases = [tc for tc in ROBUSTNESS_TEST_CASES if tc["subset"] == sub]
    y_true = [tc["category"] for tc in cases]
    y_pred = []
    for tc in cases:
        p = predictor(tc["text"])
        y_pred.append(p.category)

    # Classification metrics for this subset
    acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
    sub_metrics = EvaluationMetrics.calculate_classification_metrics(y_true, y_pred, labels=CATEGORIES)
    macro_f1 = sub_metrics["macro_f1"]

    subsets_table.append({
        "input_type": sub.replace("_", " ").title(),
        "subset_key": sub,
        "samples": len(cases),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
    })

report["robustness_breakdown_table"] = subsets_table

with open(OUT_DIR / "robustness_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

# Copy to artifacts
with open(Path("experiments/artifacts/robustness_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "="*60)
print("ROBUSTNESS BENCHMARK RESULTS:")
print(f"| {'Input Type':<30} | {'Samples':<7} | {'Accuracy':<10} | {'Macro-F1':<10} |")
print("|" + "-"*32 + "|" + "-"*9 + "|" + "-"*12 + "|" + "-"*12 + "|")
for r in subsets_table:
    print(f"| {r['input_type']:<30} | {r['samples']:<7} | {r['accuracy']*100:<9.1f}% | {r['macro_f1']:<10.4f} |")
print("="*60)
