"""Multilingual preprocessing, dedicated multitask neural heads, ordinal priority classification,
and true multiclass temperature/vector calibration for NEXUS-SLM V2 research.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES


class MultilingualTextPreprocessor:
    """Unicode-aware preprocessing preserving Devanagari vowel marks and masking IDs."""

    URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    ORDER_ID_RE = re.compile(r"\b(?:ORD|INV|AWB|TRK|TXN)[-_]?\d{4,}\b", re.IGNORECASE)
    LONG_NUM_RE = re.compile(r"\b\d{5,}\b")

    @classmethod
    def normalize(cls, text: str) -> str:
        """Apply NFC Unicode normalization and normalize IDs/whitespace cleanly."""
        if not text:
            return ""
        # NFC keeps Devanagari matras + nukta combined consistently; strip zero-width chars
        cleaned = unicodedata.normalize("NFC", text)
        cleaned = cleaned.replace("\u200b", "").replace("\u200c", "").replace("\ufeff", "")
        cleaned = cls.URL_RE.sub(" <URL> ", cleaned)
        cleaned = cls.EMAIL_RE.sub(" <EMAIL> ", cleaned)
        cleaned = cls.ORDER_ID_RE.sub(" <ORDER_ID> ", cleaned)
        cleaned = cls.LONG_NUM_RE.sub(" <NUM> ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def extract_script_indicators(text: str) -> List[float]:
        """Compute explicit script/lexical domain signal vector (length 8)."""
        if not text:
            return [0.0] * 8
        n_chars = max(len(text), 1)
        dev_chars = sum(1 for c in text if "\u0900" <= c <= "\u097f")
        ascii_alpha = sum(1 for c in text if ("a" <= c <= "z") or ("A" <= c <= "Z"))
        digits = sum(1 for c in text if c.isdigit())
        excl = text.count("!")
        ques = text.count("?")
        upper_ratio = sum(1 for c in text if c.isupper()) / max(ascii_alpha, 1)
        words = len(text.split())
        return [
            min(1.0, dev_chars / n_chars),
            min(1.0, ascii_alpha / n_chars),
            min(1.0, digits / n_chars),
            min(1.0, excl / 5.0),
            min(1.0, ques / 5.0),
            min(1.0, upper_ratio),
            min(1.0, words / 60.0),
            1.0 if dev_chars > 0 and ascii_alpha > 0 else 0.0,
        ]


def ordinal_targets_from_priority(prio_indices: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Convert priority index (0..3) into K-1 binary cumulative thresholds (y >= k)."""
    # Shape: [batch, num_classes - 1]
    thresholds = torch.arange(1, num_classes, device=prio_indices.device).unsqueeze(0)
    return (prio_indices.unsqueeze(1) >= thresholds).float()


def ordinal_probs_from_cum_logits(cum_logits: torch.Tensor) -> torch.Tensor:
    """Convert K-1 monotonic/independent cumulative logits P(y >= k) into 4-class marginals."""
    # Ensure monotonicity via cumprod on survival probabilities or clamping sequential probs
    p_ge = torch.sigmoid(cum_logits)  # [B, 3]: p>=1, p>=2, p>=3
    p_ge1 = p_ge[:, 0]
    p_ge2 = p_ge[:, 1] * p_ge1
    p_ge3 = p_ge[:, 2] * p_ge2

    p0 = 1.0 - p_ge1
    p1 = (p_ge1 - p_ge2).clamp(min=0.0)
    p2 = (p_ge2 - p_ge3).clamp(min=0.0)
    p3 = p_ge3.clamp(min=0.0)
    probs = torch.stack([p0, p1, p2, p3], dim=-1)
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)


class NexusMultitaskNeuralHead(nn.Module):
    """Dedicated PyTorch multitask classifier with shared residual trunk + 4 output heads.

    Jointly predicts:
    - category_logits (14 classes)
    - priority_cum_logits (3 cumulative ordinal thresholds for 4 ordered levels)
    - language_logits (3 classes: english, hindi, hinglish)
    - human_risk_logit (binary scalar probability logit)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_categories: int = len(CATEGORIES),
        num_languages: int = len(LANGUAGES),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.category_head = nn.Linear(hidden_dim, num_categories)
        self.priority_ordinal_head = nn.Linear(hidden_dim, len(PRIORITIES) - 1)
        self.language_head = nn.Linear(hidden_dim, num_languages)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.proj(self.input_norm(x))
        h = h + 0.5 * self.res_block(h)
        return {
            "category_logits": self.category_head(h),
            "priority_cum_logits": self.priority_ordinal_head(h),
            "language_logits": self.language_head(h),
            "risk_logit": self.risk_head(h).squeeze(-1),
        }


class MulticlassTemperatureScaler:
    """Fits scalar Temperature T > 0 or diagonal Vector Scaling strictly on holdout logits via NLL."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)
        self.bias: Optional[np.ndarray] = None

    def fit_temperature(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Optimize scalar T > 0 minimizing Negative Log-Likelihood on calibration holdout."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        targets_t = torch.tensor(targets, dtype=torch.long)
        log_temp = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=100)

        def closure():
            optimizer.zero_grad()
            temp = torch.exp(log_temp).clamp(min=0.05, max=20.0)
            loss = F.cross_entropy(logits_t / temp, targets_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.temperature = float(torch.exp(log_temp).clamp(min=0.05, max=20.0).item())
        return round(self.temperature, 4)

    def fit_vector_scaling(self, logits: np.ndarray, targets: np.ndarray, l2: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
        """Fit class-wise diagonal temperature + bias vector (regularized against overfitting)."""
        n_cls = logits.shape[1]
        logits_t = torch.tensor(logits, dtype=torch.float32)
        targets_t = torch.tensor(targets, dtype=torch.long)
        scales = nn.Parameter(torch.ones(n_cls))
        bias = nn.Parameter(torch.zeros(n_cls))
        opt = torch.optim.AdamW([scales, bias], lr=0.05, weight_decay=0.0)
        for _ in range(250):
            opt.zero_grad()
            scaled = logits_t * scales.unsqueeze(0) + bias.unsqueeze(0)
            loss = F.cross_entropy(scaled, targets_t) + l2 * ((scales - 1.0) ** 2).mean() + l2 * (bias ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            self.vector_scales = scales.detach().numpy()
            self.bias = bias.detach().numpy()
        return self.vector_scales, self.bias

    def predict_probs_temperature(self, logits: np.ndarray) -> np.ndarray:
        z = logits / max(self.temperature, 1e-4)
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=-1, keepdims=True)

    def predict_probs_vector(self, logits: np.ndarray) -> np.ndarray:
        if not hasattr(self, "vector_scales") or self.bias is None:
            return self.predict_probs_temperature(logits)
        z = logits * self.vector_scales[None, :] + self.bias[None, :]
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=-1, keepdims=True)


def compute_multiclass_calibration_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, Any]:
    """Compute ECE, MCE, Multi-class Brier Score, NLL, and reliability diagram bins on true posteriors."""
    n, k = probs.shape
    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "nll": 0.0, "num_samples": 0, "bins": []}

    pred_cls = np.argmax(probs, axis=-1)
    confidences = np.max(probs, axis=-1)
    correctness = (pred_cls == targets).astype(np.float64)

    # Multi-class Brier score: mean over samples of sum_c (p_c - y_c)^2
    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(n), targets] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))

    # Negative Log-Likelihood (cross-entropy)
    clipped = np.clip(probs[np.arange(n), targets], 1e-7, 1.0)
    nll = float(-np.mean(np.log(clipped)))

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bins_out = []
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            bins_out.append({"bin_range": f"{lo:.1f}-{hi:.1f}", "count": 0, "accuracy": 0.0, "confidence": 0.0, "gap": 0.0})
            continue
        acc = float(np.mean(correctness[mask]))
        conf = float(np.mean(confidences[mask]))
        gap = abs(acc - conf)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        bins_out.append({
            "bin_range": f"{lo:.1f}-{hi:.1f}",
            "count": cnt,
            "accuracy": round(acc, 4),
            "confidence": round(conf, 4),
            "gap": round(gap, 4),
        })

    return {
        "num_samples": n,
        "accuracy": round(float(np.mean(correctness)), 4),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "overconfidence_gap": round(float(np.mean(confidences) - np.mean(correctness)), 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "nll": round(nll, 4),
        "bins": bins_out,
    }
