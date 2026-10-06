"""CLI script to run controlled sweeps and ablation studies."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.experiments import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="NEXUS-SLM Controlled Experiments & Ablation Study")
    parser.add_argument(
        "--sweep",
        type=str,
        default="ablation",
        choices=["ablation", "lora_rank_sweep", "dataset_size_sweep", "learning_rate_sweep", "quantization_sweep"],
        help="Experiment sweep to execute",
    )
    args = parser.parse_args()

    runner = ExperimentRunner()
    if args.sweep == "ablation":
        results = runner.run_ablation_study()
        print("\n" + "=" * 60)
        print("ABLATION STUDY RESULTS:")
        print("=" * 60)
        for stage in results["ablation_stages"]:
            print(f"[{stage['label']}] -> Macro-F1: {stage['macro_f1']} | JSON Valid: {stage['json_validity']} | P50 Latency: {stage['latency_p50_ms']}ms")
        print("=" * 60)
    else:
        results = runner.run_controlled_sweep(args.sweep)
        print(f"Completed sweep '{args.sweep}': {len(results)} runs logged.")


if __name__ == "__main__":
    main()
