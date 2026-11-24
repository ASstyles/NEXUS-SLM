"""Multiclass Temperature Scaling calibration and selective prediction for NEXUS-SLM."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MulticlassTemperatureScaler:
    """
    Implements standard post-hoc multiclass temperature scaling over class logits (Guo et al., 2017).
    Optimizes a single scalar temperature T > 0 on a dedicated validation split to minimize
    Negative Log-Likelihood (NLL) without modifying classification argmax decisions.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-4, float(temperature))
        self.is_fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Fits temperature parameter T > 0 using L-BFGS-B on calibration logits.
        Args:
            logits: (N, K) array of unnormalized class logits.
            labels: (N,) array of integer ground truth class indices in [0, K-1].
        """
        assert logits.ndim == 2, "logits must be (N, K)"
        assert len(logits) == len(labels), "logits and labels count mismatch"
        n, k = logits.shape

        # One-hot encode labels
        one_hot = np.zeros((n, k), dtype=np.float64)
        for i, y in enumerate(labels):
            one_hot[i, y] = 1.0

        def nll_loss(t_val: np.ndarray) -> float:
            t = float(t_val[0])
            scaled_logits = logits / t
            # Stable log-sum-exp
            max_l = np.max(scaled_logits, axis=1, keepdims=True)
            log_sum_exp = max_l + np.log(np.sum(np.exp(scaled_logits - max_l), axis=1, keepdims=True))
            log_probs = scaled_logits - log_sum_exp
            loss = -np.sum(one_hot * log_probs) / n
            return float(loss)

        res = minimize(
            nll_loss,
            x0=np.array([1.0]),
            bounds=[(0.05, 10.0)],
            method="L-BFGS-B",
        )

        if res.success:
            self.temperature = float(res.x[0])
            self.is_fitted = True
            logger.info(f"Fitted Multiclass Temperature Scaler: T = {self.temperature:.4f}")
        else:
            self.temperature = 1.0
            self.is_fitted = False
            logger.warning(f"Temperature scaling optimization did not converge: {res.message}. Keeping T=1.0.")

        return self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Applies fitted temperature and returns calibrated softmax probabilities."""
        scaled = logits / self.temperature
        max_l = np.max(scaled, axis=1, keepdims=True)
        exp_logits = np.exp(scaled - max_l)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    @staticmethod
    def compute_multiclass_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
        """Computes standard multiclass Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2."""
        n, k = probs.shape
        one_hot = np.zeros((n, k), dtype=np.float64)
        for i, y in enumerate(labels):
            one_hot[i, y] = 1.0
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    @staticmethod
    def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> Dict[str, Any]:
        """
        Computes multiclass Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        Evaluates the top-predicted class confidence against empirical accuracy across bins.
        """
        confs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        correct = (preds == labels).astype(np.float64)
        n = len(confs)

        bins = np.linspace(0.0, 1.0, num_bins + 1)
        bin_indices = np.digitize(confs, bins[1:-1])

        ece = 0.0
        mce = 0.0
        bucket_records = []

        for b_idx in range(num_bins):
            mask = (bin_indices == b_idx)
            count = int(np.sum(mask))
            low_edge = float(bins[b_idx])
            high_edge = float(bins[b_idx + 1])

            if count > 0:
                bin_conf = float(np.mean(confs[mask]))
                bin_acc = float(np.mean(correct[mask]))
                gap = abs(bin_acc - bin_conf)
                ece += (count / n) * gap
                mce = max(mce, gap)
            else:
                bin_conf = (low_edge + high_edge) / 2.0
                bin_acc = 0.0
                gap = 0.0

            bucket_records.append({
                "bucket": b_idx,
                "range": [round(low_edge, 2), round(high_edge, 2)],
                "count": count,
                "confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
                "gap": round(gap, 4),
            })

        return {
            "expected_calibration_error": round(float(ece), 4),
            "max_calibration_error": round(float(mce), 4),
            "mean_confidence": round(float(np.mean(confs)), 4),
            "mean_accuracy": round(float(np.mean(correct)), 4),
            "buckets": bucket_records,
        }

    @staticmethod
    def compute_selective_prediction_curve(
        probs: np.ndarray, labels: np.ndarray, thresholds: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Computes Risk vs. Coverage trade-off for selective prediction.
        At each threshold tau:
          - Coverage: Proportion of tickets where max confidence >= tau
          - Selective Risk: Error rate (1 - Accuracy) on the covered subset
        """
        if thresholds is None:
            thresholds = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        confs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        correct = (preds == labels)
        n = len(confs)

        curve = []
        for tau in thresholds:
            mask = (confs >= tau)
            covered_count = int(np.sum(mask))
            coverage = round(covered_count / n, 4)

            if covered_count > 0:
                acc_on_covered = float(np.mean(correct[mask]))
                selective_risk = round(1.0 - acc_on_covered, 4)
            else:
                selective_risk = 0.0

            curve.append({
                "confidence_threshold": tau,
                "coverage_rate": coverage,
                "covered_samples": covered_count,
                "selective_accuracy": round(acc_on_covered, 4) if covered_count > 0 else 0.0,
                "selective_risk": selective_risk,
                "automation_decision": "PERMITTED" if tau >= 0.85 and coverage > 0.4 else "REQUIRES_HUMAN",
            })

        return curve
