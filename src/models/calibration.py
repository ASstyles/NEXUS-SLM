"""Confidence calibration, ECE computation, and selective prediction for NEXUS-SLM."""

import math
from typing import Any, Dict, List, Optional, Tuple


class TemperatureScalingCalibrator:
    """Post-hoc probability calibration using temperature scaling."""

    def __init__(self, temperature: float = 1.15):
        if temperature <= 0:
            raise ValueError("Temperature parameter must be positive (> 0.0)")
        self.temperature = temperature

    def calibrate_probability(self, raw_prob: float) -> float:
        """Applies temperature scaling to a scalar confidence estimate [0, 1]."""
        if raw_prob is None:
            return None

        # Clamp to avoid log(0)
        eps = 1e-6
        p = min(max(raw_prob, eps), 1.0 - eps)

        # Logit transformation
        logit = math.log(p / (1.0 - p))
        scaled_logit = logit / self.temperature

        # Sigmoid inversion
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
        return round(calibrated, 4)


class CalibrationEvaluator:
    """Computes empirical calibration metrics: Expected Calibration Error (ECE) and Brier Score."""

    @staticmethod
    def compute_ece(
        confidences: List[float],
        accuracies: List[bool],
        num_bins: int = 10,
    ) -> Dict[str, Any]:
        """Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
        if not confidences or len(confidences) != len(accuracies):
            return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "num_samples": 0}

        bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
        bins = [[] for _ in range(num_bins)]

        for conf, acc in zip(confidences, accuracies):
            if conf is None:
                continue
            # Find bin index
            bin_idx = min(int(conf * num_bins), num_bins - 1)
            bins[bin_idx].append((conf, 1.0 if acc else 0.0))

        total_samples = len(confidences)
        ece = 0.0
        mce = 0.0
        bin_details = []

        for idx, b in enumerate(bins):
            if not b:
                bin_details.append({
                    "bin_range": f"{bin_boundaries[idx]:.1f}-{bin_boundaries[idx+1]:.1f}",
                    "count": 0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "gap": 0.0,
                })
                continue

            bin_size = len(b)
            bin_acc = sum(acc for _, acc in b) / bin_size
            bin_conf = sum(conf for conf, _ in b) / bin_size
            gap = abs(bin_acc - bin_conf)

            ece += (bin_size / total_samples) * gap
            mce = max(mce, gap)

            bin_details.append({
                "bin_range": f"{bin_boundaries[idx]:.1f}-{bin_boundaries[idx+1]:.1f}",
                "count": bin_size,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "gap": round(gap, 4),
            })

        # Brier score
        brier = sum((conf - (1.0 if acc else 0.0)) ** 2 for conf, acc in zip(confidences, accuracies) if conf is not None) / max(total_samples, 1)

        return {
            "ece": round(ece, 4),
            "mce": round(mce, 4),
            "brier_score": round(brier, 4),
            "num_samples": total_samples,
            "bins": bin_details,
        }


class SelectivePredictor:
    """Selective prediction and risk-coverage routing based on calibrated confidence thresholds."""

    def __init__(self, confidence_threshold: float = 0.85):
        self.threshold = confidence_threshold

    def evaluate_routing(self, confidence: Optional[float], deterministic_policy_escalation: bool) -> Tuple[bool, str]:
        """Decides whether to automate ticket or route to human review.

        Returns (requires_human, reason).
        """
        # Mandatory policy escalation always dominates
        if deterministic_policy_escalation:
            return True, "mandatory_policy_escalation"

        # Missing confidence must be escalated safely
        if confidence is None:
            return True, "confidence_unavailable_fallback"

        # Confidence below automation threshold
        if confidence < self.threshold:
            return True, f"confidence_below_threshold ({confidence:.2f} < {self.threshold:.2f})"

        # Straight-through automation approved
        return False, "automated_high_confidence"

