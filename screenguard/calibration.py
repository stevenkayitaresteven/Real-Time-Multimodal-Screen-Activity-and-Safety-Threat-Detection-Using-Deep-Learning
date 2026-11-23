"""Post-hoc temperature scaling (Guo et al., ICML 2017).

Alert thresholds only mean something if probabilities are calibrated, so one
temperature per task is fitted on the validation set after training and
stored in the checkpoint.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200, t_min: float = 0.25,
                    t_max: float = 10.0) -> float:
    """Fit a single temperature by minimising validation NLL.

    The result is clamped to ``[t_min, t_max]``: when the validation set is
    (almost) perfectly classified, NLL keeps decreasing as T -> 0, and an
    unbounded fit would make the model wildly over-confident on
    out-of-distribution screens.
    """
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
    t = float(log_t.detach().exp().clamp(t_min, t_max))
    return t if np.isfinite(t) else 1.0
