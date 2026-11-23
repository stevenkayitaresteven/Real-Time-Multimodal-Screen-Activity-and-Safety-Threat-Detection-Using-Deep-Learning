"""Post-hoc temperature scaling (Guo et al., ICML 2017).

Alert thresholds only mean something if probabilities are calibrated, so one
temperature per task is fitted on the validation set after training and
stored in the checkpoint.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    if len(labels) == 0:
        return 1.0
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp().clamp(0.05, 20.0))
    return t if np.isfinite(t) else 1.0
