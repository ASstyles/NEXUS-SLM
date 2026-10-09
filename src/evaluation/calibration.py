"""Confidence calibration evaluation and reliability diagram analysis for NEXUS-SLM."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceCalibrator:
    """Evaluates whether model confidence scores correlate with empirical correctness."""

    def __init__(self, num_bins: int = 5):
        self.num_bins = num_bins

    def compute_calibration_metrics(
        self,
        confidences: List[float],
        correctness: List[bool],
    ) -> Dict[str, Any]:
        """
        Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
        Brier Score, and reliability bucket statistics.
        """
        confs = np.array(confidences, dtype=np.float64)
        accs = np.array(correctness, dtype=np.float64)
        n = len(confs)

        if n == 0:
            return {
                "sample_count": 0,
                "expected_calibration_error": 0.0,
                "max_calibration_error": 0.0,
                "brier_score": 0.0,
                "mean_confidence": 0.0,
                "mean_accuracy": 0.0,
                "buckets": [],
                "correlation_pearson": 0.0,
            }

        # Calculate Brier score: mean squared difference
        brier_score = float(np.mean((confs - accs) ** 2))

        # Binning
        bins = np.linspace(0.0, 1.0, self.num_bins + 1)
        bin_indices = np.digitize(confs, bins[1:-1])

        bucket_data = []
        ece = 0.0
        mce = 0.0

        for b_idx in range(self.num_bins):
            mask = (bin_indices == b_idx)
            count = int(np.sum(mask))
            low_edge = float(bins[b_idx])
            high_edge = float(bins[b_idx + 1])

            if count > 0:
                bin_conf = float(np.mean(confs[mask]))
                bin_acc = float(np.mean(accs[mask]))
                abs_diff = abs(bin_acc - bin_conf)
                ece += (count / n) * abs_diff
                mce = max(mce, abs_diff)
            else:
                bin_conf = (low_edge + high_edge) / 2.0
                bin_acc = 0.0
                abs_diff = 0.0

            bucket_data.append({
                "bucket_index": b_idx,
                "range": [round(low_edge, 2), round(high_edge, 2)],
                "count": count,
                "mean_confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
                "calibration_gap": round(abs_diff, 4),
            })

        # Pearson correlation
        if np.std(confs) > 1e-6 and np.std(accs) > 1e-6:
            corr = float(np.corrcoef(confs, accs)[0, 1])
        else:
            corr = 0.0

        mean_conf = float(np.mean(confs))
        mean_acc = float(np.mean(accs))

        # Overconfidence gap
        overconfidence_gap = mean_conf - mean_acc

        recommendation = "calibrated_confidence"
        if overconfidence_gap > 0.25 or ece > 0.20:
            recommendation = (
                "raw_model_confidence_is_overconfident: "
                "Expose calibrated confidence or apply conservative human-escalation gating "
                "when confidence < 0.70. Do not treat raw confidence as true Bayesian posterior."
            )
        else:
            recommendation = "calibrated_confidence_acceptable"

        return {
            "sample_count": n,
            "mean_confidence": round(mean_conf, 4),
            "mean_accuracy": round(mean_acc, 4),
            "overconfidence_gap": round(overconfidence_gap, 4),
            "expected_calibration_error": round(float(ece), 4),
            "max_calibration_error": round(float(mce), 4),
            "brier_score": round(brier_score, 4),
            "correlation_pearson": round(corr, 4),
            "recommendation": recommendation,
            "buckets": bucket_data,
        }

    def plot_reliability_diagram(
        self,
        calibration_results: Dict[str, Any],
        output_path: str = "experiments/artifacts/confidence_calibration.png",
    ):
        """Generates reliability diagram and confidence distribution plots."""
        buckets = calibration_results["buckets"]
        ranges = [f"{b['range'][0]:.1f}-{b['range'][1]:.1f}" for b in buckets]
        confs = [b["mean_confidence"] for b in buckets]
        accs = [b["accuracy"] for b in buckets]
        counts = [b["count"] for b in buckets]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # 1. Reliability curve
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
        ax1.bar(
            confs,
            accs,
            width=0.12,
            alpha=0.6,
            color="#2563eb",
            edgecolor="#1d4ed8",
            label="Empirical Accuracy",
        )
        ax1.plot(confs, accs, "ro-", linewidth=2, label="Observed Calibration")
        ax1.set_xlabel("Mean Model Confidence")
        ax1.set_ylabel("Empirical Accuracy")
        ax1.set_title(
            f"Reliability Diagram (ECE = {calibration_results['expected_calibration_error']:.3f})"
        )
        ax1.set_xlim(0, 1.05)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper left")

        # 2. Confidence & Sample Count Distribution
        ax2.bar(ranges, counts, color="#0d9488", edgecolor="#0f766e", alpha=0.8)
        ax2.set_xlabel("Confidence Bucket Range")
        ax2.set_ylabel("Number of Samples")
        ax2.set_title(
            f"Sample Distribution (N={calibration_results['sample_count']}, Gap={calibration_results['overconfidence_gap']:.3f})"
        )
        ax2.grid(True, linestyle="--", alpha=0.5)

        for i, count in enumerate(counts):
            ax2.text(i, count + 0.3, str(count), ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        logger.info(f"Saved confidence calibration plot to {output_path}")
