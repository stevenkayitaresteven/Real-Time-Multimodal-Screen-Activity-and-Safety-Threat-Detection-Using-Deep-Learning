"""Combine the neural threat distribution with rule evidence.

For every threat class ``c != benign`` with model probability ``p_c`` and
rule evidence ``r_c``:

    q_c      = 1 - (1 - p_c)(1 - r_c)          (noisy-OR: either source can raise risk)
    q_benign = p_benign * prod_c (1 - r_c)     (any rule hit lowers the benign mass)

and ``q`` is renormalised.  ``mode="model"`` and ``mode="rules"`` give the two
ablations evaluated in the experiments.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

MODES = ("model", "rules", "hybrid")


def fuse_threat(p_model: np.ndarray, rule_scores: Dict[str, float], classes: Sequence[str], mode: str = "hybrid",
                rule_scale: float = 1.0) -> np.ndarray:
    """``p_model``: ``[C]`` or ``[N, C]`` probabilities (class 0 must be ``benign``)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if classes[0] != "benign":
        raise ValueError("first threat class must be 'benign'")
    p = np.asarray(p_model, dtype=np.float64)
    single = p.ndim == 1
    p = p[None] if single else p
    r = np.array([min(0.95, rule_scale * rule_scores.get(c, 0.0)) for c in classes[1:]], dtype=np.float64)
    if mode == "model":
        q = p.copy()
    elif mode == "rules":
        q = np.concatenate([np.full((p.shape[0], 1), np.prod(1 - r)), np.broadcast_to(r, (p.shape[0], len(r)))], axis=1)
        q[:, 0] = np.maximum(q[:, 0], 1e-6)
    else:
        q_threat = 1 - (1 - p[:, 1:]) * (1 - r)
        q_benign = p[:, :1] * np.prod(1 - r)
        q = np.concatenate([q_benign, q_threat], axis=1)
    q = q / q.sum(axis=1, keepdims=True)
    return q[0] if single else q
