"""CLI script to prepare, validate, deduplicate, and split NEXUS-SLM datasets."""

import argparse
import json
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.pipeline import DataPipeline


def main():
    parser = argparse.ArgumentParser(description="NEXUS-SLM Dataset Pipeline")
    parser.add_argument("--version", type=str, default="dataset_v1", help="Dataset version tag")
    parser.add_argument("--samples", type=int, default=1200, help="Number of synthetic samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    pipeline = DataPipeline(version=args.version, seed=args.seed)
    summary = pipeline.run_pipeline(num_synthetic_samples=args.samples)
    print("\n" + "=" * 60)
    print(f"NEXUS-SLM DATA PIPELINE COMPLETE: {args.version}")
    print("=" * 60)
    print(f"Train samples: {summary['train_size']}")
    print(f"Validation samples: {summary['val_size']}")
    print(f"Locked Test samples: {summary['test_size']}")
    print(f"Duplicates removed: {summary['quality_report']['duplicate_count']}")
    print(f"Leakage status: {summary['leakage_report']['status']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
