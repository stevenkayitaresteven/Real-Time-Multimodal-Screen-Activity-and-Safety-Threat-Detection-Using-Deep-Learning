"""Dependency-free evaluation metrics (numpy only)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def confusion_matrix(y: np.ndarray, p: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm


def roc_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    """Binary ROC-AUC via the Mann-Whitney U statistic (ties get average rank)."""
    y = np.asarray(y).astype(bool)
    pos, neg = y.sum(), (~y).sum()
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = np.asarray(s)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> Optional[float]:
    """Detection rate when the threshold is set so benign false-alarm rate <= target."""
    y = np.asarray(y).astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    neg = np.sort(np.asarray(s)[~y])[::-1]
    k = int(np.floor(target_fpr * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf
    return float((np.asarray(s)[y] > thr).mean())


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def classification_report(y: np.ndarray, probs: np.ndarray, classes: Sequence[str]) -> Dict:
    y = np.asarray(y)
    pred = probs.argmax(1)
    n = len(classes)
    cm = confusion_matrix(y, pred, n)
    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4), "f1": round(float(f1), 4),
                        "support": support}
        if support > 0:
            f1s.append(f1)
    return {
        "n": int(len(y)),
        "accuracy": round(float((pred == y).mean()), 4),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "ece": round(expected_calibration_error(probs, y), 4),
        "per_class": per_class,
        "confusion": cm.tolist(),
        "classes": list(classes),
    }


def binary_threat_report(y: np.ndarray, probs: np.ndarray) -> Dict:
    """Collapse to 'any threat' vs benign (class 0) for alerting metrics."""
    is_threat = np.asarray(y) != 0
    score = 1.0 - probs[:, 0]
    pred = score >= 0.5
    tp = int((pred & is_threat).sum())
    fp = int((pred & ~is_threat).sum())
    fn = int((~pred & is_threat).sum())
    tn = int((~pred & ~is_threat).sum())
    auc = roc_auc(is_threat, score)
    t1 = tpr_at_fpr(is_threat, score, 0.01)
    return {
        "auroc": round(auc, 4) if auc is not None else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "tpr_at_1pct_fpr": round(t1, 4) if t1 is not None else None,
    }
