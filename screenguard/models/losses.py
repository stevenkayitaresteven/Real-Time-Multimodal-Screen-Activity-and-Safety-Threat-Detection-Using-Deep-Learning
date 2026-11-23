"""Masked multi-task loss with optional homoscedastic-uncertainty weighting.

Real training data is *heterogeneous*: a screenshot dataset has activity
labels, an NSFW dataset has only NSFW labels, a phishing corpus has only
threat labels.  Each sample therefore carries ``IGNORE_INDEX`` for tasks it is
not annotated for, and the loss of a task is computed only over the samples
that have that label.

With ``weighting="uncertainty"`` the per-task weights are learned following
Kendall, Gal & Cipolla (CVPR 2018): ``L = sum_i exp(-s_i) * L_i + s_i`` where
``s_i = log(sigma_i^2)`` is a learnable scalar per task.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..taxonomy import IGNORE_INDEX


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: Iterable[str], weighting: str = "uncertainty", weights: Optional[Dict[str, float]] = None,
                 label_smoothing: float = 0.0, class_weights: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.tasks = list(tasks)
        if weighting not in ("uncertainty", "fixed"):
            raise ValueError(f"Unknown weighting {weighting!r}")
        self.weighting = weighting
        self.weights = {t: float((weights or {}).get(t, 1.0)) for t in self.tasks}
        self.label_smoothing = label_smoothing
        self.log_vars = nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in self.tasks})
        self.class_weights = class_weights or {}

    def forward(self, logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = None
        parts: Dict[str, float] = {}
        for t in self.tasks:
            y = targets[t]
            valid = y != IGNORE_INDEX
            if not bool(valid.any()):
                continue
            cw = self.class_weights.get(t)
            loss = F.cross_entropy(logits[t][valid].float(), y[valid], weight=cw.to(logits[t].device) if cw is not None else None,
                                   label_smoothing=self.label_smoothing)
            parts[t] = float(loss.detach())
            if self.weighting == "uncertainty":
                s = self.log_vars[t]
                term = torch.exp(-s) * loss + s
            else:
                term = self.weights[t] * loss
            total = term if total is None else total + term
        if total is None:  # batch without any labels (should not happen with sane data)
            total = sum(l.sum() * 0.0 for l in logits.values())
        return total, parts
