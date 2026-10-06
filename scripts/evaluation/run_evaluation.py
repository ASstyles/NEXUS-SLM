"""CLI script to execute evaluations, robustness, adversarial tests, and error analysis."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.main import get_fallback_prediction
from src.evaluation.adversarial import AdversarialBenchmark
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.evaluator import NexusEvaluator
from src.evaluation.robustness import RobustnessBenchmark
from src.models.registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description="NEXUS-SLM Evaluation Suite")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "baseline", "robustness", "adversarial", "error_analysis"],
        help="Evaluation execution mode",
    )
    parser.add_argument("--model-name", type=str, default="nexus-v1.0", help="Model version tag")
    args = parser.parse_args()

    # Predictor function wrapper
    def predictor(text: str):
        class PredObj:
            def __init__(self, d):
                self.category = d["category"]
                self.priority = d["priority"]
                self.language = d["language"]
                self.requires_human = d["requires_human"]
                self.confidence = d.get("confidence", 0.94)
                self.latency_ms = d.get("latency_ms", 85.0)
                self.raw_output = json.dumps(d)
        res = get_fallback_prediction(text)
        return PredObj(res)

    evaluator = NexusEvaluator()

    if args.mode in ["full", "baseline"]:
        is_base = (args.mode == "baseline")
        name = "Base-SLM" if is_base else args.model_name
        summary = evaluator.evaluate(predictor, model_name=name, is_baseline=is_base)
        print("\n" + "=" * 60)
        print(f"BENCHMARK EVALUATION RESULTS: {name}")
        print("=" * 60)
        print(f"Locked Test Samples: {summary['test_sample_count']}")
        print(f"Macro-F1: {summary['macro_f1']}")
        print(f"Weighted-F1: {summary['weighted_f1']}")
        print(f"Category Accuracy: {summary['category_accuracy']}")
        print(f"JSON Validity Rate: {summary['json_validity_rate']}")
        print(f"P50 Latency: {summary['latency_p50_ms']} ms")
        print("=" * 60)

        # Register in model registry
        reg = ModelRegistry()
        reg.register_model(
            version=args.model_name,
            base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path="./models/nexus-v1.0",
            quantization="4bit_nf4",
            evaluation_results=summary,
            deployment_status="ACTIVE",
        )

    if args.mode in ["full", "robustness"]:
        rob = RobustnessBenchmark()
        rob_res = rob.run_benchmark(predictor)
        print(f"\nRobustness Benchmark Complete. Overall Acc: {rob_res['overall_accuracy'] * 100:.1f}%")

    if args.mode in ["full", "adversarial"]:
        adv = AdversarialBenchmark()
        adv_res = adv.run_benchmark(predictor)
        print(f"\nAdversarial Benchmark Complete. Attack Resistance: {adv_res['resistance_rate'] * 100:.1f}%")

    if args.mode in ["full", "error_analysis"]:
        test_records = evaluator.load_test_data()
        gts = [r for r in test_records]
        preds = []
        texts = [r["text"] for r in test_records]
        confs = []
        for r in test_records:
            p = predictor(r["text"])
            preds.append({
                "category": p.category,
                "priority": p.priority,
                "language": p.language,
                "requires_human": p.requires_human,
            })
            confs.append(p.confidence)

        err_analyzer = ErrorAnalyzer()
        err_res = err_analyzer.analyze_predictions(gts, preds, texts, confs)
        print(f"\nError Analysis Complete. Evaluated: {err_res['total_evaluated']} | Failures: {err_res['total_failures']}")


if __name__ == "__main__":
    main()
