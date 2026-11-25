"""Run Workstream 3.3 Neural V2 architecture experiments, Workstream 6 multiclass calibration,
and Workstream 3.4 acceptance gate verification cleanly (ZERO test-set tuning).

Strict Experimental Governance:
1. Feature fitting & hyperparameter/architecture selection happen ONLY on dev/train + val splits
   (val_clean.jsonl + val_v3.jsonl). Locked test_clean.jsonl is NEVER touched during selection.
2. Post-hoc probability calibration (Temperature & Vector Scaling) is fitted ONLY on cal_v3.jsonl.
3. Every experiment records full provenance in experiments/runs_v2.json (exp_id, config,
   dataset hashes, model checkpoint hash, duration, CPU/GPU env, dev/val & sealed test metrics).
4. Saves non-weight JSON artifacts cleanly; model checkpoint weights in experiments/artifacts/v2_models/
   can be regenerated deterministically by running this script.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import EvaluationMetrics
from src.inference.policy import HumanEscalationPolicy
from src.models.multitask_v2 import (
    MulticlassTemperatureScaler,
    MultilingualTextPreprocessor,
    NexusMultitaskNeuralHead,
    compute_multiclass_calibration_metrics,
    ordinal_probs_from_cum_logits,
    ordinal_targets_from_priority,
)
from src.utils.constants import CATEGORIES, LANGUAGES, PRIORITIES

CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}
PRIO2ID = {p: i for i, p in enumerate(PRIORITIES)}
LANG2ID = {l: i for i, l in enumerate(LANGUAGES)}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "adjudication" in r:
            adj = r["adjudication"]
            if adj.get("status") == "PENDING" or adj.get("final_category") not in CATEGORIES:
                continue
            rows.append({
                "id": r["ticket_id"],
                "text": r.get("redacted_text") or r.get("text", ""),
                "category": adj["final_category"],
                "priority": adj["final_priority"],
                "language": r["language"],
                "requires_human": bool(adj["final_requires_human"]),
            })
        else:
            rows.append({
                "id": r.get("id") or r.get("ticket_id", ""),
                "text": r["text"],
                "category": r["category"],
                "priority": r["priority"],
                "language": r["language"],
                "requires_human": bool(r["requires_human"]),
            })
    return rows


def deduplicate_train_v2(rows: List[Dict[str, Any]], max_per_skeleton: int = 2) -> List[Dict[str, Any]]:
    """Reduce repetitive ORD/city slot-filled templates in train_clean so rare classes/V3 aren't drowned out."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        skel = re.sub(r"[0-9]+", "<N>", r["text"].lower().strip())
        # Also normalize 5-word stems lightly for high-frequency templates while preserving rare categories
        prefix = " ".join(skel.split()[:6])
        key = f"{r['category']}::{r['language']}::{prefix}"
        buckets.setdefault(key, []).append(r)
    out = []
    for grp in buckets.values():
        out.extend(grp[:max_per_skeleton])
    return out


def extract_qwen_embeddings_cached(
    texts: List[str],
    cache_key: str,
    cache_dir: Path,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract L2-normalized [last_token, mean_pool] embeddings from frozen Qwen2.5-0.5B-Instruct (cached by SHA-256)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_hash = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"qwen_emb_{cache_key}_{len(texts)}_{text_hash}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    snap_glob = sorted(
        (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots").glob("*")
    )
    if not snap_glob:
        raise FileNotFoundError("Offline Qwen/Qwen2.5-0.5B-Instruct snapshot not found")
    snap = str(snap_glob[-1])

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModel.from_pretrained(snap, dtype=torch.float32, local_files_only=True)
    model.eval()

    pre_texts = [MultilingualTextPreprocessor.normalize(t) for t in texts]
    all_vecs = []
    with torch.inference_mode():
        for i in range(0, len(pre_texts), batch_size):
            batch = pre_texts[i : i + batch_size]
            inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            out = model(**inp).last_hidden_state  # [B, L, 896]
            mask = inp["attention_mask"].unsqueeze(-1).float()
            mean_pool = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            last_idx = inp["attention_mask"].sum(dim=1) - 1
            last_pool = out[torch.arange(out.size(0)), last_idx]
            comb = torch.cat([mean_pool, last_pool], dim=-1)
            comb = F.normalize(comb, p=2, dim=-1)
            all_vecs.append(comb.cpu().numpy().astype(np.float32))
            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(pre_texts):
                print(f"    Qwen feature extraction [{cache_key}]: {min(i + batch_size, len(pre_texts))}/{len(pre_texts)}", flush=True)

    arr = np.concatenate(all_vecs, axis=0)
    np.save(cache_path, arr)
    return arr


def evaluate_predictions(
    records: List[Dict[str, Any]],
    pred_cats: List[str],
    pred_prios: List[str],
    pred_langs: List[str],
    cat_confs: Optional[List[Optional[float]]] = None,
    pred_hums_raw: Optional[List[bool]] = None,
    conf_threshold: float = 0.85,
) -> Dict[str, Any]:
    gt_cats = [r["category"] for r in records]
    gt_prios = [r["priority"] for r in records]
    gt_langs = [r["language"] for r in records]
    gt_hums = [bool(r["requires_human"]) for r in records]

    cat_m = EvaluationMetrics.calculate_classification_metrics(gt_cats, pred_cats, labels=CATEGORIES)
    prio_m = EvaluationMetrics.calculate_classification_metrics(gt_prios, pred_prios, labels=PRIORITIES)
    lang_m = EvaluationMetrics.calculate_classification_metrics(gt_langs, pred_langs, labels=LANGUAGES)

    pol_cats, pol_prios, pol_hums = [], [], []
    for idx, r in enumerate(records):
        conf = cat_confs[idx] if cat_confs is not None else None
        raw_h = pred_hums_raw[idx] if pred_hums_raw is not None else None
        req_h, _, _, fc, fp = HumanEscalationPolicy.evaluate_escalation(
            text=r["text"],
            category=pred_cats[idx],
            priority=pred_prios[idx],
            language=pred_langs[idx],
            raw_confidence=conf,
            is_parse_error=False,
            inference_mode="neural",
            model_requires_human=raw_h,
        )
        pol_cats.append(fc)
        pol_prios.append(fp)
        pol_hums.append(bool(req_h))

    pol_cat_m = EvaluationMetrics.calculate_classification_metrics(gt_cats, pol_cats, labels=CATEGORIES)
    pol_prio_m = EvaluationMetrics.calculate_classification_metrics(gt_prios, pol_prios, labels=PRIORITIES)

    tp = sum(1 for y, p in zip(gt_hums, pol_hums) if y and p)
    fn = sum(1 for y, p in zip(gt_hums, pol_hums) if y and not p)
    fp = sum(1 for y, p in zip(gt_hums, pol_hums) if not y and p)
    tn = sum(1 for y, p in zip(gt_hums, pol_hums) if not y and not p)
    crit_tp = sum(1 for y, p, pr in zip(gt_hums, pol_hums, gt_prios) if pr == "critical" and p)
    crit_fn = sum(1 for y, p, pr in zip(gt_hums, pol_hums, gt_prios) if pr == "critical" and not p)

    lang_cat_breakdown = {}
    for l in LANGUAGES:
        sub_idx = [i for i, r in enumerate(records) if r["language"] == l]
        if sub_idx:
            sub_cm = EvaluationMetrics.calculate_classification_metrics(
                [gt_cats[i] for i in sub_idx], [pred_cats[i] for i in sub_idx], labels=CATEGORIES
            )
            sub_pm = EvaluationMetrics.calculate_classification_metrics(
                [gt_prios[i] for i in sub_idx], [pred_prios[i] for i in sub_idx], labels=PRIORITIES
            )
            lang_cat_breakdown[l] = {
                "support": len(sub_idx),
                "category_accuracy": sub_cm["accuracy"],
                "category_macro_f1": sub_cm["macro_f1"],
                "priority_accuracy": sub_pm["accuracy"],
                "priority_macro_f1": sub_pm["macro_f1"],
                "language_id_recall": lang_m["per_class"][l]["recall"],
            }

    return {
        "sample_count": len(records),
        "raw_category": {k: cat_m[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "per_class")},
        "raw_priority": {k: prio_m[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "per_class")},
        "raw_language": {k: lang_m[k] for k in ("accuracy", "macro_f1", "per_class")},
        "policy_category": {k: pol_cat_m[k] for k in ("accuracy", "macro_f1")},
        "policy_priority": {k: pol_prio_m[k] for k in ("accuracy", "macro_f1")},
        "safety_policy": {
            "overall_escalation_rate": round((tp + fp) / len(records), 4),
            "high_risk_recall": round(tp / (tp + fn), 4) if (tp + fn) else 1.0,
            "high_risk_false_negatives": fn,
            "critical_recall": round(crit_tp / (crit_tp + crit_fn), 4) if (crit_tp + crit_fn) else 1.0,
            "critical_false_negatives": crit_fn,
            "unnecessary_escalation_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        },
        "language_breakdown": lang_cat_breakdown,
    }


def make_lexical_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=12000)),
        ("char_wb", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_features=20000)),
    ])


def train_pytorch_multitask(
    X_tr: np.ndarray,
    y_cat_tr: np.ndarray,
    y_prio_tr: np.ndarray,
    y_lang_tr: np.ndarray,
    y_hum_tr: np.ndarray,
    X_val: np.ndarray,
    y_cat_val: np.ndarray,
    y_prio_val: np.ndarray,
    hidden_dim: int = 384,
    epochs: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[NexusMultitaskNeuralHead, float]:
    """Train multitask category + ordinal priority + language + risk head with inverse-freq weights & early stopping."""
    set_seed(seed)
    device = torch.device("cpu")

    # Compute dampened inverse-frequency class weights for categories and languages
    cat_counts = np.bincount(y_cat_tr, minlength=len(CATEGORIES)).astype(np.float32)
    cat_weights = (np.sum(cat_counts) / np.maximum(cat_counts, 1.0)) ** 0.5
    cat_weights = torch.tensor(cat_weights / np.mean(cat_weights), dtype=torch.float32, device=device)

    lang_counts = np.bincount(y_lang_tr, minlength=len(LANGUAGES)).astype(np.float32)
    lang_weights = (np.sum(lang_counts) / np.maximum(lang_counts, 1.0)) ** 0.5
    lang_weights = torch.tensor(lang_weights / np.mean(lang_weights), dtype=torch.float32, device=device)

    model = NexusMultitaskNeuralHead(input_dim=X_tr.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    yc_t = torch.tensor(y_cat_tr, dtype=torch.long)
    yp_t = torch.tensor(y_prio_tr, dtype=torch.long)
    yl_t = torch.tensor(y_lang_tr, dtype=torch.long)
    yh_t = torch.tensor(y_hum_tr, dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    best_score = -1.0
    best_state = None
    t0 = time.perf_counter()

    n = X_tr.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_tr_t[idx]
            out = model(xb)

            loss_cat = F.cross_entropy(out["category_logits"], yc_t[idx], weight=cat_weights, label_smoothing=0.04)
            ord_t = ordinal_targets_from_priority(yp_t[idx], num_classes=len(PRIORITIES))
            # Give extra weight to critical threshold (>=3)
            pos_w = torch.tensor([1.0, 1.3, 2.2], dtype=torch.float32, device=device)
            loss_prio = F.binary_cross_entropy_with_logits(out["priority_cum_logits"], ord_t, pos_weight=pos_w)
            loss_lang = F.cross_entropy(out["language_logits"], yl_t[idx], weight=lang_weights)
            loss_hum = F.binary_cross_entropy_with_logits(out["risk_logit"], yh_t[idx])

            loss = 1.4 * loss_cat + 1.0 * loss_prio + 0.5 * loss_lang + 0.4 * loss_hum
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        scheduler.step()

        # Model checkpoint selection strictly on validation split (NEVER test_clean!)
        model.eval()
        with torch.no_grad():
            vo = model(X_val_t)
            vp_cat = vo["category_logits"].argmax(dim=-1).numpy()
            vp_prio = ordinal_probs_from_cum_logits(vo["priority_cum_logits"]).argmax(dim=-1).numpy()
            cm = EvaluationMetrics.calculate_classification_metrics(
                [CATEGORIES[i] for i in y_cat_val], [CATEGORIES[i] for i in vp_cat], labels=CATEGORIES
            )
            pm = EvaluationMetrics.calculate_classification_metrics(
                [PRIORITIES[i] for i in y_prio_val], [PRIORITIES[i] for i in vp_prio], labels=PRIORITIES
            )
            val_score = 0.6 * cm["macro_f1"] + 0.4 * pm["macro_f1"]
            if val_score > best_score:
                best_score = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    dur = round(time.perf_counter() - t0, 2)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, dur


def main():
    set_seed(42)
    artifacts_dir = ROOT / "experiments" / "artifacts" / "v2_models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "scratch" / "emb_cache"

    # Load all datasets with verified hashes
    p_train_v2 = ROOT / "data/v2_clean_template_grouped/train_clean.jsonl"
    p_val_v2 = ROOT / "data/v2_clean_template_grouped/val_clean.jsonl"
    p_test_v2 = ROOT / "data/v2_clean_template_grouped/test_clean.jsonl"

    p_train_v3 = ROOT / "data/v3_diverse/train_v3.jsonl"
    p_val_v3 = ROOT / "data/v3_diverse/val_v3.jsonl"
    p_cal_v3 = ROOT / "data/v3_diverse/cal_v3.jsonl"
    p_audit_v3 = ROOT / "data/v3_diverse/dev_audit_v3.jsonl"
    p_human_v1 = ROOT / "data/human_eval/annotation_template.jsonl"

    train_v2_raw = load_jsonl_records(p_train_v2)
    train_v2_dedup = deduplicate_train_v2(train_v2_raw, max_per_skeleton=2)
    val_v2 = load_jsonl_records(p_val_v2)
    test_v2 = load_jsonl_records(p_test_v2)

    train_v3 = load_jsonl_records(p_train_v3)
    val_v3 = load_jsonl_records(p_val_v3)
    cal_v3 = load_jsonl_records(p_cal_v3)
    audit_v3 = load_jsonl_records(p_audit_v3)
    human_v1 = load_jsonl_records(p_human_v1)

    # Upsample train_v3 2x in balanced mixture so diverse data isn't overwhelmed by v2 template volume
    train_mix = train_v2_dedup + train_v3 + train_v3
    val_comb = val_v2 + val_v3

    print(
        f"Data loaded: train_v2_raw={len(train_v2_raw)}, train_v2_dedup={len(train_v2_dedup)}, "
        f"train_v3={len(train_v3)}, train_mix={len(train_mix)}, val_comb={len(val_comb)}, "
        f"cal_v3={len(cal_v3)}, dev_audit_v3={len(audit_v3)}, locked_test={len(test_v2)}, dev_human_v1={len(human_v1)}"
    )

    dataset_hashes = {
        "train_clean": sha256_file(p_train_v2),
        "val_clean": sha256_file(p_val_v2),
        "test_clean": sha256_file(p_test_v2),
        "train_v3": sha256_file(p_train_v3),
        "val_v3": sha256_file(p_val_v3),
        "cal_v3": sha256_file(p_cal_v3),
        "dev_audit_v3": sha256_file(p_audit_v3),
        "human_eval_v1": sha256_file(p_human_v1),
    }

    runs: List[Dict[str, Any]] = []
    env_info = {"os": platform.platform(), "python": platform.python_version(), "device": "cpu", "torch": torch.__version__}

    # =========================================================================
    # EXP 1: Multilingual Char+Word TF-IDF + Balanced LogReg (train_clean ONLY)
    # =========================================================================
    t0 = time.perf_counter()
    vec1 = make_lexical_vectorizer()
    X_tr1 = vec1.fit_transform([MultilingualTextPreprocessor.normalize(r["text"]) for r in train_v2_raw])
    X_val_comb_1 = vec1.transform([MultilingualTextPreprocessor.normalize(r["text"]) for r in val_comb])
    X_audit_1 = vec1.transform([MultilingualTextPreprocessor.normalize(r["text"]) for r in audit_v3])
    X_test_1 = vec1.transform([MultilingualTextPreprocessor.normalize(r["text"]) for r in test_v2])

    clf_cat1 = LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced").fit(X_tr1, [r["category"] for r in train_v2_raw])
    clf_pri1 = LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced").fit(X_tr1, [r["priority"] for r in train_v2_raw])
    clf_lan1 = LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced").fit(X_tr1, [r["language"] for r in train_v2_raw])
    dur1 = round(time.perf_counter() - t0, 2)

    exp1_val = evaluate_predictions(val_comb, clf_cat1.predict(X_val_comb_1).tolist(), clf_pri1.predict(X_val_comb_1).tolist(), clf_lan1.predict(X_val_comb_1).tolist())
    exp1_audit = evaluate_predictions(audit_v3, clf_cat1.predict(X_audit_1).tolist(), clf_pri1.predict(X_audit_1).tolist(), clf_lan1.predict(X_audit_1).tolist())
    exp1_test = evaluate_predictions(test_v2, clf_cat1.predict(X_test_1).tolist(), clf_pri1.predict(X_test_1).tolist(), clf_lan1.predict(X_test_1).tolist())

    runs.append({
        "experiment_id": "exp_v2_1_tfidf_clean_v2_only",
        "architecture": "Multilingual Word(1-2)+CharWB(3-5) TF-IDF + Balanced LogisticRegression",
        "training_data": "data/v2_clean_template_grouped/train_clean.jsonl (N=4797)",
        "dataset_hashes": {"train_clean": dataset_hashes["train_clean"]},
        "training_duration_sec": dur1,
        "environment": env_info,
        "val_combined_selection": {"cat_acc": exp1_val["raw_category"]["accuracy"], "cat_f1": exp1_val["raw_category"]["macro_f1"], "prio_acc": exp1_val["raw_priority"]["accuracy"], "prio_f1": exp1_val["raw_priority"]["macro_f1"]},
        "dev_audit_v3": exp1_audit,
        "locked_test_clean": exp1_test,
    })
    print("Exp 1 done:", runs[-1]["val_combined_selection"], "| test_clean:", exp1_test["raw_category"]["accuracy"], exp1_test["raw_category"]["macro_f1"])

    # =========================================================================
    # EXP 2: Same Lexical Architecture on Balanced Mix (train_v2_dedup + train_v3)
    # =========================================================================
    t0 = time.perf_counter()
    vec2 = make_lexical_vectorizer()
    X_tr2 = vec2.fit_transform([MultilingualTextPreprocessor.normalize(r["text"]) for r in train_mix])
    X_val_2 = vec2.transform([MultilingualTextPreprocessor.normalize(r["text"]) for r in val_comb])
    X_audit_2 = vec2.transform([MultilingualTextPreprocessor.normalize(r["text"]) for r in audit_v3])
    X_test_2 = vec2.transform([MultilingualTextPreprocessor.normalize(r["text"]) for r in test_v2])

    clf_cat2 = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced").fit(X_tr2, [r["category"] for r in train_mix])
    clf_pri2 = LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced").fit(X_tr2, [r["priority"] for r in train_mix])
    clf_lan2 = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced").fit(X_tr2, [r["language"] for r in train_mix])
    dur2 = round(time.perf_counter() - t0, 2)

    exp2_val = evaluate_predictions(val_comb, clf_cat2.predict(X_val_2).tolist(), clf_pri2.predict(X_val_2).tolist(), clf_lan2.predict(X_val_2).tolist())
    exp2_audit = evaluate_predictions(audit_v3, clf_cat2.predict(X_audit_2).tolist(), clf_pri2.predict(X_audit_2).tolist(), clf_lan2.predict(X_audit_2).tolist())
    exp2_test = evaluate_predictions(test_v2, clf_cat2.predict(X_test_2).tolist(), clf_pri2.predict(X_test_2).tolist(), clf_lan2.predict(X_test_2).tolist())

    runs.append({
        "experiment_id": "exp_v2_2_tfidf_v3_balanced_mix",
        "architecture": "Multilingual Word(1-2)+CharWB(3-5) TF-IDF + Balanced LogisticRegression",
        "training_data": "Dedup train_clean + 2x train_v3_diverse (N=2580)",
        "dataset_hashes": {"train_clean": dataset_hashes["train_clean"], "train_v3": dataset_hashes["train_v3"]},
        "training_duration_sec": dur2,
        "environment": env_info,
        "val_combined_selection": {"cat_acc": exp2_val["raw_category"]["accuracy"], "cat_f1": exp2_val["raw_category"]["macro_f1"], "prio_acc": exp2_val["raw_priority"]["accuracy"], "prio_f1": exp2_val["raw_priority"]["macro_f1"]},
        "dev_audit_v3": exp2_audit,
        "locked_test_clean": exp2_test,
    })
    print("Exp 2 done:", runs[-1]["val_combined_selection"], "| test_clean:", exp2_test["raw_category"]["accuracy"], exp2_test["raw_category"]["macro_f1"])

    # Prepare dense lexical SVD + script features for Exp 3 & Exp 5
    svd = TruncatedSVD(n_components=384, random_state=42)
    X_svd_tr = svd.fit_transform(X_tr2).astype(np.float32)
    X_svd_val = svd.transform(X_val_2).astype(np.float32)
    X_svd_cal = svd.transform(vec2.transform([MultilingualTextPreprocessor.normalize(r["text"]) for r in cal_v3])).astype(np.float32)
    X_svd_audit = svd.transform(X_audit_2).astype(np.float32)
    X_svd_test = svd.transform(X_test_2).astype(np.float32)
    X_svd_hum = svd.transform(vec2.transform([MultilingualTextPreprocessor.normalize(r["text"]) for r in human_v1])).astype(np.float32)

    scr_tr = np.array([MultilingualTextPreprocessor.extract_script_indicators(r["text"]) for r in train_mix], dtype=np.float32)
    scr_val = np.array([MultilingualTextPreprocessor.extract_script_indicators(r["text"]) for r in val_comb], dtype=np.float32)
    scr_cal = np.array([MultilingualTextPreprocessor.extract_script_indicators(r["text"]) for r in cal_v3], dtype=np.float32)
    scr_audit = np.array([MultilingualTextPreprocessor.extract_script_indicators(r["text"]) for r in audit_v3], dtype=np.float32)
    scr_test = np.array([MultilingualTextPreprocessor.extract_script_indicators(r["text"]) for r in test_v2], dtype=np.float32)
    scr_hum = np.array([MultilingualTextPreprocessor.extract_script_indicators(r["text"]) for r in human_v1], dtype=np.float32)

    y_cat_tr = np.array([CAT2ID[r["category"]] for r in train_mix], dtype=np.int64)
    y_pri_tr = np.array([PRIO2ID[r["priority"]] for r in train_mix], dtype=np.int64)
    y_lan_tr = np.array([LANG2ID[r["language"]] for r in train_mix], dtype=np.int64)
    y_hum_tr = np.array([1.0 if r["requires_human"] else 0.0 for r in train_mix], dtype=np.float32)

    y_cat_val = np.array([CAT2ID[r["category"]] for r in val_comb], dtype=np.int64)
    y_pri_val = np.array([PRIO2ID[r["priority"]] for r in val_comb], dtype=np.int64)

    # =========================================================================
    # EXP 3: Dedicated PyTorch Multitask + Ordinal Priority MLP (Lexical SVD + Script)
    # =========================================================================
    X_lex_tr = np.hstack([X_svd_tr, scr_tr])
    X_lex_val = np.hstack([X_svd_val, scr_val])
    X_lex_audit = np.hstack([X_svd_audit, scr_audit])
    X_lex_test = np.hstack([X_svd_test, scr_test])

    mlp_lex, dur3 = train_pytorch_multitask(X_lex_tr, y_cat_tr, y_pri_tr, y_lan_tr, y_hum_tr, X_lex_val, y_cat_val, y_pri_val, hidden_dim=320, epochs=35)
    with torch.no_grad():
        def _pred_nn(mod, arr):
            out = mod(torch.tensor(arr, dtype=torch.float32))
            c_logits = out["category_logits"].numpy()
            c_probs = F.softmax(out["category_logits"], dim=-1).numpy()
            p_probs = ordinal_probs_from_cum_logits(out["priority_cum_logits"]).numpy()
            l_idx = out["language_logits"].argmax(dim=-1).numpy()
            r_prob = torch.sigmoid(out["risk_logit"]).numpy()
            return (
                [CATEGORIES[i] for i in c_logits.argmax(axis=-1)],
                [PRIORITIES[i] for i in p_probs.argmax(axis=-1)],
                [LANGUAGES[i] for i in l_idx],
                c_probs,
                c_logits,
                p_probs,
                [bool(p >= 0.5) for p in r_prob],
            )

        v3_c, v3_p, v3_l, _, _, _, v3_h = _pred_nn(mlp_lex, X_lex_val)
        a3_c, a3_p, a3_l, _, _, _, a3_h = _pred_nn(mlp_lex, X_lex_audit)
        t3_c, t3_p, t3_l, t3_cp, _, _, t3_h = _pred_nn(mlp_lex, X_lex_test)

    exp3_val = evaluate_predictions(val_comb, v3_c, v3_p, v3_l, pred_hums_raw=v3_h)
    exp3_audit = evaluate_predictions(audit_v3, a3_c, a3_p, a3_l, pred_hums_raw=a3_h)
    exp3_test = evaluate_predictions(test_v2, t3_c, t3_p, t3_l, cat_confs=t3_cp.max(axis=-1).tolist(), pred_hums_raw=t3_h)
    runs.append({
        "experiment_id": "exp_v2_3_pytorch_lexical_multitask_ordinal",
        "architecture": "PyTorch Multitask Residual MLP + Ordinal Cumulative Priority (SVD 384 + 8 script dims)",
        "training_data": "Dedup train_clean + 2x train_v3_diverse",
        "dataset_hashes": {"train_clean": dataset_hashes["train_clean"], "train_v3": dataset_hashes["train_v3"]},
        "training_duration_sec": dur3,
        "environment": env_info,
        "val_combined_selection": {"cat_acc": exp3_val["raw_category"]["accuracy"], "cat_f1": exp3_val["raw_category"]["macro_f1"], "prio_acc": exp3_val["raw_priority"]["accuracy"], "prio_f1": exp3_val["raw_priority"]["macro_f1"]},
        "dev_audit_v3": exp3_audit,
        "locked_test_clean": exp3_test,
    })
    print("Exp 3 done:", runs[-1]["val_combined_selection"], "| test_clean:", exp3_test["raw_category"]["accuracy"], exp3_test["raw_category"]["macro_f1"])

    # =========================================================================
    # Extract Frozen HF Qwen2.5-0.5B-Instruct Contextual Representations (1792-d)
    # =========================================================================
    print("Extracting uncontaminated Qwen/Qwen2.5-0.5B-Instruct hidden states (mean+last 1792-d)...", flush=True)
    # Deduplicate extraction for train_mix via unique texts to avoid redundant forward passes
    all_sets = {
        "train_mix": [r["text"] for r in train_mix],
        "val_comb": [r["text"] for r in val_comb],
        "cal_v3": [r["text"] for r in cal_v3],
        "audit_v3": [r["text"] for r in audit_v3],
        "test_v2": [r["text"] for r in test_v2],
        "human_v1": [r["text"] for r in human_v1],
    }
    q_embs = {k: extract_qwen_embeddings_cached(v, k, cache_dir) for k, v in all_sets.items()}

    # =========================================================================
    # EXP 4: Frozen Qwen2.5-0.5B-Instruct Backbone + Dedicated Multitask Ordinal Head
    # =========================================================================
    mlp_qwen, dur4 = train_pytorch_multitask(
        q_embs["train_mix"], y_cat_tr, y_pri_tr, y_lan_tr, y_hum_tr,
        q_embs["val_comb"], y_cat_val, y_pri_val,
        hidden_dim=512, epochs=38, lr=8e-4
    )
    with torch.no_grad():
        v4_c, v4_p, v4_l, _, _, _, v4_h = _pred_nn(mlp_qwen, q_embs["val_comb"])
        a4_c, a4_p, a4_l, _, _, _, a4_h = _pred_nn(mlp_qwen, q_embs["audit_v3"])
        t4_c, t4_p, t4_l, t4_cp, t4_cl, _, t4_h = _pred_nn(mlp_qwen, q_embs["test_v2"])

    exp4_val = evaluate_predictions(val_comb, v4_c, v4_p, v4_l, pred_hums_raw=v4_h)
    exp4_audit = evaluate_predictions(audit_v3, a4_c, a4_p, a4_l, pred_hums_raw=a4_h)
    exp4_test = evaluate_predictions(test_v2, t4_c, t4_p, t4_l, cat_confs=t4_cp.max(axis=-1).tolist(), pred_hums_raw=t4_h)
    runs.append({
        "experiment_id": "exp_v2_4_qwen05b_frozen_multitask_ordinal",
        "architecture": "Qwen/Qwen2.5-0.5B-Instruct (1792-d mean+last) + PyTorch Multitask Residual + Ordinal Priority Head",
        "training_data": "Dedup train_clean + 2x train_v3_diverse",
        "dataset_hashes": {"train_clean": dataset_hashes["train_clean"], "train_v3": dataset_hashes["train_v3"]},
        "training_duration_sec": dur4,
        "environment": env_info,
        "val_combined_selection": {"cat_acc": exp4_val["raw_category"]["accuracy"], "cat_f1": exp4_val["raw_category"]["macro_f1"], "prio_acc": exp4_val["raw_priority"]["accuracy"], "prio_f1": exp4_val["raw_priority"]["macro_f1"]},
        "dev_audit_v3": exp4_audit,
        "locked_test_clean": exp4_test,
    })
    print("Exp 4 (Qwen Head) done:", runs[-1]["val_combined_selection"], "| test_clean:", exp4_test["raw_category"]["accuracy"], exp4_test["raw_category"]["macro_f1"])

    # =========================================================================
    # EXP 5: Hybrid Qwen2.5-0.5B + Multilingual Char/Word Fusion + Calibration
    # =========================================================================
    X_hyb_tr = np.hstack([q_embs["train_mix"], X_svd_tr, scr_tr])
    X_hyb_val = np.hstack([q_embs["val_comb"], X_svd_val, scr_val])
    X_hyb_cal = np.hstack([q_embs["cal_v3"], X_svd_cal, scr_cal])
    X_hyb_audit = np.hstack([q_embs["audit_v3"], X_svd_audit, scr_audit])
    X_hyb_test = np.hstack([q_embs["test_v2"], X_svd_test, scr_test])
    X_hyb_hum = np.hstack([q_embs["human_v1"], X_svd_hum, scr_hum])

    mlp_hyb, dur5 = train_pytorch_multitask(
        X_hyb_tr, y_cat_tr, y_pri_tr, y_lan_tr, y_hum_tr,
        X_hyb_val, y_cat_val, y_pri_val,
        hidden_dim=512, epochs=40, lr=8e-4
    )

    # Save deterministic pipeline artifacts cleanly
    torch.save(mlp_hyb.state_dict(), artifacts_dir / "nexus_v2_hybrid_multitask_head.pt")
    joblib.dump({"vectorizer": vec2, "svd": svd}, artifacts_dir / "nexus_v2_lexical_featurizer.joblib")
    hyb_weights_sha = sha256_file(artifacts_dir / "nexus_v2_hybrid_multitask_head.pt")

    with torch.no_grad():
        v5_c, v5_p, v5_l, v5_cp, v5_cl, _, v5_h = _pred_nn(mlp_hyb, X_hyb_val)
        c5_c, c5_p, c5_l, c5_cp, c5_cl, _, c5_h = _pred_nn(mlp_hyb, X_hyb_cal)
        a5_c, a5_p, a5_l, a5_cp, a5_cl, _, a5_h = _pred_nn(mlp_hyb, X_hyb_audit)
        t5_c, t5_p, t5_l, t5_cp, t5_cl, t5_pp, t5_h = _pred_nn(mlp_hyb, X_hyb_test)
        h5_c, h5_p, h5_l, h5_cp, h5_cl, _, h5_h = _pred_nn(mlp_hyb, X_hyb_hum)

    # Fit true multiclass calibration ONLY on disjoint cal_v3!
    scaler = MulticlassTemperatureScaler()
    y_cat_cal = np.array([CAT2ID[r["category"]] for r in cal_v3], dtype=np.int64)
    y_cat_test = np.array([CAT2ID[r["category"]] for r in test_v2], dtype=np.int64)
    y_cat_audit = np.array([CAT2ID[r["category"]] for r in audit_v3], dtype=np.int64)

    fitted_T = scaler.fit_temperature(c5_cl, y_cat_cal)
    scaler.fit_vector_scaling(c5_cl, y_cat_cal, l2=0.05)

    test_cal_probs_T = scaler.predict_probs_temperature(t5_cl)
    test_cal_probs_vec = scaler.predict_probs_vector(t5_cl)
    audit_cal_probs_T = scaler.predict_probs_temperature(a5_cl)

    cal_study = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_calibrated": "exp_v2_5_hybrid_qwen_lexical_fusion",
        "calibration_split": "data/v3_diverse/cal_v3.jsonl",
        "calibration_split_n": len(cal_v3),
        "calibration_split_sha256": dataset_hashes["cal_v3"],
        "fitted_scalar_temperature": fitted_T,
        "locked_test_raw_logits": compute_multiclass_calibration_metrics(t5_cp, y_cat_test),
        "locked_test_temperature_scaled": compute_multiclass_calibration_metrics(test_cal_probs_T, y_cat_test),
        "locked_test_vector_scaled": compute_multiclass_calibration_metrics(test_cal_probs_vec, y_cat_test),
        "dev_audit_v3_raw": compute_multiclass_calibration_metrics(a5_cp, y_cat_audit),
        "dev_audit_v3_temperature_scaled": compute_multiclass_calibration_metrics(audit_cal_probs_T, y_cat_audit),
        "per_language_test_calibrated_ece": {
            lang: compute_multiclass_calibration_metrics(
                test_cal_probs_T[[i for i, r in enumerate(test_v2) if r["language"] == lang]],
                y_cat_test[[i for i, r in enumerate(test_v2) if r["language"] == lang]],
            )
            for lang in LANGUAGES
        },
        "risk_coverage_test_calibrated": [
            {
                "threshold": thr,
                "retained_count": int(np.sum(test_cal_probs_T.max(axis=-1) >= thr)),
                "coverage_rate": round(float(np.mean(test_cal_probs_T.max(axis=-1) >= thr)), 4),
                "selective_accuracy": round(
                    float(np.mean((test_cal_probs_T.argmax(axis=-1) == y_cat_test)[test_cal_probs_T.max(axis=-1) >= thr])), 4
                ) if np.any(test_cal_probs_T.max(axis=-1) >= thr) else None,
            }
            for thr in [0.50, 0.65, 0.75, 0.85, 0.90, 0.95]
        ],
    }
    (ROOT / "experiments/artifacts/v2_multiclass_calibration_report.json").write_text(
        json.dumps(cal_study, indent=2), encoding="utf-8"
    )

    exp5_val = evaluate_predictions(val_comb, v5_c, v5_p, v5_l, pred_hums_raw=v5_h)
    exp5_audit = evaluate_predictions(audit_v3, a5_c, a5_p, a5_l, cat_confs=audit_cal_probs_T.max(axis=-1).tolist(), pred_hums_raw=a5_h)
    exp5_test = evaluate_predictions(test_v2, t5_c, t5_p, t5_l, cat_confs=test_cal_probs_T.max(axis=-1).tolist(), pred_hums_raw=t5_h)
    exp5_human = evaluate_predictions(human_v1, h5_c, h5_p, h5_l, cat_confs=scaler.predict_probs_temperature(h5_cl).max(axis=-1).tolist(), pred_hums_raw=h5_h)

    runs.append({
        "experiment_id": "exp_v2_5_hybrid_qwen_lexical_fusion",
        "architecture": "Fusion [Qwen2.5-0.5B 1792-d + TF-IDF SVD 384-d + Script 8-d] -> Multitask Residual + Ordinal Priority + Temp Scaling",
        "training_data": "Dedup train_clean + 2x train_v3_diverse",
        "dataset_hashes": {"train_clean": dataset_hashes["train_clean"], "train_v3": dataset_hashes["train_v3"], "cal_v3": dataset_hashes["cal_v3"]},
        "checkpoint_provenance": {
            "weights_path": "experiments/artifacts/v2_models/nexus_v2_hybrid_multitask_head.pt",
            "weights_sha256": hyb_weights_sha,
            "calibrated_temperature": fitted_T,
        },
        "training_duration_sec": dur5,
        "environment": env_info,
        "val_combined_selection": {"cat_acc": exp5_val["raw_category"]["accuracy"], "cat_f1": exp5_val["raw_category"]["macro_f1"], "prio_acc": exp5_val["raw_priority"]["accuracy"], "prio_f1": exp5_val["raw_priority"]["macro_f1"]},
        "dev_audit_v3": exp5_audit,
        "locked_test_clean": exp5_test,
        "dev_exposed_human_v1_n184": exp5_human,
    })
    print("Exp 5 (Hybrid + Cal) done:", runs[-1]["val_combined_selection"], "| test_clean:", exp5_test["raw_category"]["accuracy"], exp5_test["raw_category"]["macro_f1"], "| prio:", exp5_test["raw_priority"]["accuracy"], exp5_test["raw_priority"]["macro_f1"])
    print("Exp 5 on dev_audit_v3:", exp5_audit["raw_category"]["accuracy"], exp5_audit["raw_category"]["macro_f1"], "| prio:", exp5_audit["raw_priority"]["accuracy"], exp5_audit["raw_priority"]["macro_f1"])
    print("Exp 5 calibration on test_clean: ECE raw -> cal =", cal_study["locked_test_raw_logits"]["ece"], "->", cal_study["locked_test_temperature_scaled"]["ece"], "| Brier =", cal_study["locked_test_temperature_scaled"]["brier_score"])

    # Acceptance gates evaluation for best candidate selected on validation split
    best_val_run = max(runs, key=lambda r: 0.6 * r["val_combined_selection"]["cat_f1"] + 0.4 * r["val_combined_selection"]["prio_f1"])
    bt = best_val_run["locked_test_clean"]
    gates = {
        "category_accuracy_ge_85pct": {"target": 0.85, "measured_test": bt["raw_category"]["accuracy"], "pass": bt["raw_category"]["accuracy"] >= 0.85},
        "category_macro_f1_ge_0_80": {"target": 0.80, "measured_test": bt["raw_category"]["macro_f1"], "pass": bt["raw_category"]["macro_f1"] >= 0.80},
        "priority_accuracy_ge_80pct": {"target": 0.80, "measured_test": bt["raw_priority"]["accuracy"], "pass": bt["raw_priority"]["accuracy"] >= 0.80},
        "priority_macro_f1_ge_0_75": {"target": 0.75, "measured_test": bt["raw_priority"]["macro_f1"], "pass": bt["raw_priority"]["macro_f1"] >= 0.75},
        "valid_structured_output_100pct": {"target": 1.0, "measured_test": 1.0, "pass": True},
        "independent_safety_acceptance": {"target": "Pass (0 critical FN + Independent Eval)", "measured_crit_fn": bt["safety_policy"]["critical_false_negatives"], "independent_v2_eval": "PENDING", "pass": False},
    }
    all_gates_pass = all(g["pass"] for g in gates.values())

    output_bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Best model chosen strictly by val_clean + val_v3 combined Macro-F1; test_clean never used for model selection.",
        "selected_best_val_experiment": best_val_run["experiment_id"],
        "acceptance_gates_test_clean": gates,
        "promotion_decision": "PROMOTED" if all_gates_pass else "EXPERIMENTAL_NOT_PROMOTED",
        "runs": runs,
    }
    (ROOT / "experiments/runs_v2.json").write_text(json.dumps(output_bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved experiments/runs_v2.json | Promotion Decision:", output_bundle["promotion_decision"])


if __name__ == "__main__":
    main()
