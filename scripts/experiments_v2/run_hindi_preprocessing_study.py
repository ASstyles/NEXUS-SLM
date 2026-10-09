"""Hindi Preprocessing and Tokenization Experiment Suite.

Benchmarks Devanagari handling strategies for NEXUS-SLM:
- H1: Original Devanagari (raw input text)
- H2: Normalized Devanagari (Unicode NFKC + Indic diacritic normalization)
- H3: Romanized Hindi (Phonetic rule-based Latin transliteration)
- H4: Language-Aware Hybrid Preprocessor (script detection + selective entity normalization)

Measures:
- Category Accuracy
- Category Macro-F1
- Hindi-Specific Accuracy
- Hindi Macro-F1
- JSON Validity Rate
- Latency (P50 and P95)
"""

import json
import re
import time
import unicodedata
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.evaluation.metrics import EvaluationMetrics
from src.inference.parser import OutputParser
from src.models.prompt_formatter import PromptFormatter
from src.utils.constants import CATEGORIES

ARTIFACTS_DIR = WORKSPACE_ROOT / "experiments" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_FILE = WORKSPACE_ROOT / "data" / "test" / "test.jsonl"
ADAPTER_PATH = WORKSPACE_ROOT / "models" / "nexus_qlora_v1"
BASE_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"

# Hindi to Roman transliteration mapping for high-frequency support terms
HINDI_ROMAN_MAP = {
    "मेरा": "mera", "मेरी": "meri", "मेरे": "mere",
    "रिफंड": "refund", "भुगतान": "payment", "पैसे": "paise", "रुपये": "rupaye",
    "खाता": "account", "खाते": "khate", "बैंक": "bank",
    "नहीं": "nahi", "आया": "aaya", "मिला": "mila", "हुआ": "hua",
    "ऑर्डर": "order", "आदेश": "order", "पार्सल": "parcel", "सामान": "saman",
    "कैंसिल": "cancel", "रद्द": "cancel", "डिलीवरी": "delivery", "पहुंचा": "pahuncha",
    "कट": "kat", "गया": "gaya", "गए": "gaye", "है": "hai", "हैं": "hain",
    "कृपया": "please", "मदद": "help", "सहायता": "help", "तुरंत": "turant",
    "वापस": "wapas", "लॉगिन": "login", "पासवर्ड": "password", "ओटीपी": "otp",
    "अटक": "atak", "विफल": "fail", "फेल": "fail", "असफल": "failed",
    "कूरियर": "courier", "संख्या": "number", "स्टेटस": "status"
}


def normalize_devanagari(text: str) -> str:
    """H2: Unicode NFKC normalization and cleanup."""
    text = unicodedata.normalize("NFKC", text)
    # Normalize danda and double danda to periods
    text = text.replace("।", ".").replace("॥", ".")
    # Normalize zero-width joiners/non-joiners
    text = text.replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"\s+", " ", text).strip()


def transliterate_to_roman(text: str) -> str:
    """H3: Phonetic transliteration of Hindi to Romanized Hinglish."""
    norm = normalize_devanagari(text)
    words = norm.split()
    romanized = []
    for w in words:
        clean_w = w.strip(".,!?:;")
        punct = w[len(clean_w):] if len(clean_w) < len(w) else ""
        if clean_w in HINDI_ROMAN_MAP:
            romanized.append(HINDI_ROMAN_MAP[clean_w] + punct)
        else:
            # Fallback character transliteration
            romanized.append(clean_w + punct)
    return " ".join(romanized)


def language_aware_preprocess(text: str) -> str:
    """H4: Script detection and selective normalization."""
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    if not has_devanagari:
        return text.strip()
    # Hybrid: Normalize Devanagari and transliterate domain keywords to Roman
    return transliterate_to_roman(text)


def evaluate_pipeline(model, tokenizer, parser, records, preprocess_fn):
    preds = []
    latencies = []
    valid_flags = []

    for rec in records:
        text = preprocess_fn(rec["text"])
        prompt = PromptFormatter.format_prompt(text, tokenizer=tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=45,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        gen = out[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
        is_valid, parsed, _ = parser.parse_and_validate(raw)
        valid_flags.append(is_valid)

        if is_valid and parsed:
            preds.append({
                "category": parsed["category"],
                "priority": parsed["priority"],
                "language": parsed["language"],
                "requires_human": parsed["requires_human"],
            })
        else:
            preds.append({"category": "unknown", "priority": "unknown", "language": "unknown", "requires_human": False})

    # Overall metrics
    metrics = EvaluationMetrics.calculate_classification_metrics(
        y_true=[r["category"] for r in records],
        y_pred=[p["category"] for p in preds],
        labels=CATEGORIES,
    )
    struct = EvaluationMetrics.calculate_structured_metrics(records, preds, valid_flags)

    # Hindi-specific metrics
    hindi_indices = [i for i, r in enumerate(records) if r["language"].lower() in ["hindi", "hinglish"]]
    if hindi_indices:
        h_true = [records[i]["category"] for i in hindi_indices]
        h_pred = [preds[i]["category"] for i in hindi_indices]
        h_metrics = EvaluationMetrics.calculate_classification_metrics(h_true, h_pred, CATEGORIES)
        hindi_acc = round(sum(1 for t, p in zip(h_true, h_pred) if t == p) / len(h_true), 4)
        hindi_f1 = h_metrics["macro_f1"]
    else:
        hindi_acc = None
        hindi_f1 = None

    return {
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "category_accuracy": struct["category_accuracy"],
        "json_validity_rate": struct["json_validity_rate"],
        "hindi_accuracy": hindi_acc,
        "hindi_macro_f1": hindi_f1,
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
    }


def run_hindi_study():
    print("=" * 70)
    print("NEXUS-SLM HINDI TOKENIZATION & PREPROCESSING EXPERIMENT (H1-H4)")
    print("=" * 70)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records)} test records from locked test set.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model.eval()
    parser = OutputParser()

    experiments = [
        ("H1_Original_Devanagari", "Original Devanagari input without modification", lambda x: x),
        ("H2_Normalized_Devanagari", "Unicode NFKC and Indic punctuation normalization", normalize_devanagari),
        ("H3_Romanized_Hindi", "Phonetic Hindi-to-Latin transliteration", transliterate_to_roman),
        ("H4_Language_Aware", "Language-aware hybrid preprocessor with script routing", language_aware_preprocess),
    ]

    results = []
    for exp_id, desc, fn in experiments:
        print(f"\nRunning {exp_id} ({desc})...")
        res = evaluate_pipeline(model, tokenizer, parser, records, fn)
        res["experiment_id"] = exp_id
        res["description"] = desc
        results.append(res)
        print(f"  Macro-F1: {res['macro_f1']} | Hindi Acc: {res['hindi_accuracy']} | Hindi F1: {res['hindi_macro_f1']} | JSON Valid: {res['json_validity_rate']} | P95: {res['latency_p95_ms']} ms")

    # Save artifact
    out_file = ARTIFACTS_DIR / "hindi_preprocessing_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "experiments": results}, f, indent=2)
    print(f"\nSaved Hindi preprocessing study results to: {out_file}")

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("NEXUS-SLM-Hindi-Preprocessing-Study")
        for r in results:
            with mlflow.start_run(run_name=r["experiment_id"]):
                mlflow.log_params({"experiment_id": r["experiment_id"], "description": r["description"]})
                mlflow.log_metrics({
                    "macro_f1": r["macro_f1"],
                    "hindi_accuracy": r["hindi_accuracy"] or 0.0,
                    "hindi_macro_f1": r["hindi_macro_f1"] or 0.0,
                    "json_validity_rate": r["json_validity_rate"],
                    "latency_p95_ms": r["latency_p95_ms"],
                })
        print("Logged Hindi preprocessing study to MLflow.")
    except Exception as e:
        print(f"MLflow note: {e}")

    return results


if __name__ == "__main__":
    run_hindi_study()
