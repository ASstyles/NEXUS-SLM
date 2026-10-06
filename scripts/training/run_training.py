"""CLI script to run QLoRA fine-tuning for NEXUS-SLM."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.trainer import NexusTrainer
from src.utils.hardware import print_hardware_summary


def main():
    parser = argparse.ArgumentParser(description="NEXUS-SLM QLoRA Training Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/qlora_default.yaml",
        help="Path to training configuration YAML",
    )
    args = parser.parse_args()

    print_hardware_summary()
    trainer = NexusTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
