"""Robustness benchmarking suite for NEXUS-SLM across linguistic variations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from src.evaluation.metrics import EvaluationMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Standardized Robustness Test Suite
ROBUSTNESS_TEST_CASES = [
    # English Canonical
    {"subset": "english_canonical", "text": "My payment was deducted from my bank account, but my order status still shows unpaid.", "category": "payment_failure", "priority": "high", "language": "english", "requires_human": True},
    {"subset": "english_canonical", "text": "I returned the item three days ago. When will I receive my refund?", "category": "refund_request", "priority": "medium", "language": "english", "requires_human": False},
    {"subset": "english_canonical", "text": "I am unable to log in to my account. It says account suspended or blocked.", "category": "account_access", "priority": "high", "language": "english", "requires_human": True},

    # Hindi Devanagari
    {"subset": "hindi_devanagari", "text": "मेरे बैंक खाते से पैसे कट गए हैं, लेकिन ऑर्डर कन्फर्म नहीं हुआ।", "category": "payment_failure", "priority": "high", "language": "hindi", "requires_human": True},
    {"subset": "hindi_devanagari", "text": "मैंने सामान वापस कर दिया था, मेरा रिफंड कब तक खाते में आएगा?", "category": "refund_request", "priority": "medium", "language": "hindi", "requires_human": False},
    {"subset": "hindi_devanagari", "text": "डिलीवरी एजेंट ने 'डिलीवर' मार्क कर दिया लेकिन मुझे कोई सामान नहीं मिला!", "category": "delivery_issue", "priority": "high", "language": "hindi", "requires_human": True},

    # Hinglish Roman
    {"subset": "hinglish_roman", "text": "Mera payment kat gaya but order confirm nahi hua.", "category": "payment_failure", "priority": "high", "language": "hinglish", "requires_human": True},
    {"subset": "hinglish_roman", "text": "Maine item return kar diya tha 3 din pehle, mera refund kab aayega?", "category": "refund_request", "priority": "medium", "language": "hinglish", "requires_human": False},
    {"subset": "hinglish_roman", "text": "Mera parcel abhi tak nahi aaya, courier tracking update nahi ho rahi.", "category": "order_tracking", "priority": "medium", "language": "hinglish", "requires_human": False},

    # Typos & Misspellings
    {"subset": "typos_and_misspellings", "text": "my paymnt faild and mony deductd pls hlp", "category": "payment_failure", "priority": "high", "language": "english", "requires_human": True},
    {"subset": "typos_and_misspellings", "text": "refnd kab credit hoga accnt me", "category": "refund_request", "priority": "medium", "language": "hinglish", "requires_human": False},
    {"subset": "typos_and_misspellings", "text": "delvry boy ne galat parcel delivrd kiya", "category": "delivery_issue", "priority": "high", "language": "hinglish", "requires_human": True},

    # Informal Slang & Emojis
    {"subset": "informal_slang_emojis", "text": "bro money got deducted 😭 help me out yaar", "category": "payment_failure", "priority": "high", "language": "hinglish", "requires_human": True},
    {"subset": "informal_slang_emojis", "text": "yaar refund do na itna late kyu karte ho 😡", "category": "refund_request", "priority": "medium", "language": "hinglish", "requires_human": False},
    {"subset": "informal_slang_emojis", "text": "bhai password reset link bhej do urgently 🙏", "category": "password_reset", "priority": "medium", "language": "hinglish", "requires_human": False},

    # Ultra Short Queries
    {"subset": "ultra_short_queries", "text": "refund?", "category": "refund_request", "priority": "medium", "language": "english", "requires_human": False},
    {"subset": "ultra_short_queries", "text": "cancel order", "category": "order_cancel", "priority": "high", "language": "english", "requires_human": False},
    {"subset": "ultra_short_queries", "text": "paise wapas do", "category": "refund_request", "priority": "medium", "language": "hinglish", "requires_human": False},

    # Noisy Punctuation & Caps
    {"subset": "noisy_punctuation_caps", "text": "PAYMENT!!!! FAILED???? WHERE IS MY ORDER????", "category": "payment_failure", "priority": "high", "language": "english", "requires_human": True},
    {"subset": "noisy_punctuation_caps", "text": "MERA ACCOUNT BLOCK KYUN KIYA???!! UNBLOCK NOW!!!!!", "category": "account_access", "priority": "high", "language": "hinglish", "requires_human": True},
    {"subset": "noisy_punctuation_caps", "text": "FRAUD CHARGE ON MY CARD!!! $500 STOLEN????", "category": "fraud_report", "priority": "critical", "language": "english", "requires_human": True},

    # Complex Multi-Sentence Long
    {"subset": "complex_multisentence_long", "text": "Good morning. I placed an order three weeks ago for a birthday present. The package was delayed twice, and when the courier finally arrived yesterday, the outer shipping carton was smashed and the contents were completely missing. I have tried reaching the seller with no reply.", "category": "delivery_issue", "priority": "high", "language": "english", "requires_human": True},
    {"subset": "complex_multisentence_long", "text": "Hello support team, I noticed that my credit card statement has three separate debit deductions of ₹1499 each on 4th October. However, on my orders dashboard, only a single order ID is listed. Please verify transaction logs and immediately refund the duplicate debits.", "category": "payment_failure", "priority": "high", "language": "hinglish", "requires_human": True},

    # Mixed-Script Code-Switched
    {"subset": "mixed_script_code_switched", "text": "Mera payment कट गया लेकिन order confirm नहीं हुआ।", "category": "payment_failure", "priority": "high", "language": "hinglish", "requires_human": True},
    {"subset": "mixed_script_code_switched", "text": "Order deliver दिखा रहा है but mujhe parcel नहीं मिला।", "category": "delivery_issue", "priority": "high", "language": "hinglish", "requires_human": True},
]


class RobustnessBenchmark:
    """Evaluates model degradation across noisy, informal, and multilingual variations."""

    def __init__(self, output_dir: str = "experiments/artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(self, predictor_fn: Callable[[str], Any]) -> Dict[str, Any]:
        """Runs the robustness test suite and returns performance breakdown by subset."""
        logger.info(f"Running Robustness Benchmark across {len(ROBUSTNESS_TEST_CASES)} cases...")

        results_by_subset: Dict[str, Dict[str, Any]] = {}
        subsets = sorted(list({tc["subset"] for tc in ROBUSTNESS_TEST_CASES}))

        total_correct = 0
        total_valid_json = 0
        total_cases = len(ROBUSTNESS_TEST_CASES)

        for sub in subsets:
            cases = [tc for tc in ROBUSTNESS_TEST_CASES if tc["subset"] == sub]
            sub_correct = 0
            sub_valid = 0

            for tc in cases:
                pred = predictor_fn(tc["text"])
                p_cat = getattr(pred, "category", "")
                conf = getattr(pred, "confidence", 0.0)

                valid_json = (conf > 0.0)
                if valid_json:
                    sub_valid += 1
                    total_valid_json += 1

                if str(p_cat).lower() == tc["category"]:
                    sub_correct += 1
                    total_correct += 1

            n = len(cases)
            results_by_subset[sub] = {
                "test_count": n,
                "category_accuracy": round(sub_correct / n, 4) if n else 0.0,
                "json_validity_rate": round(sub_valid / n, 4) if n else 0.0,
            }

        overall_acc = round(total_correct / total_cases, 4) if total_cases else 0.0
        overall_valid = round(total_valid_json / total_cases, 4) if total_cases else 0.0

        canonical_acc = results_by_subset.get("english_canonical", {}).get("category_accuracy", 1.0)
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_accuracy": overall_acc,
            "overall_json_validity": overall_valid,
            "canonical_accuracy": canonical_acc,
            "subsets": results_by_subset,
        }

        with open(self.output_dir / "robustness_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Robustness Benchmark complete: Overall Acc={overall_acc}, JSON Valid={overall_valid}")
        return report
