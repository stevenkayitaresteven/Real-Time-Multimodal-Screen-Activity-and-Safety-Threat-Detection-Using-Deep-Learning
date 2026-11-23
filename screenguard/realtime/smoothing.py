"""Temporal smoothing and alert hysteresis.

Per-frame predictions flicker (a scrolling page briefly shows a scary word, a
video frame is momentarily ambiguous).  Users lose trust in a system that
fires and retracts alerts every second, so:

1. class probabilities are smoothed with an exponential moving average, and
2. alerts follow a hysteresis state machine: raise only after the risk stays
   above ``on`` for ``min_on`` analysed frames, clear only after it stays
   below ``off`` (< ``on``) for ``min_off`` frames, with a cooldown between
   repeated alerts of the same kind.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


class EMASmoother:
    def __init__(self, alpha: float = 0.6):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.state: Dict[str, np.ndarray] = {}

    def update(self, key: str, probs: np.ndarray) -> np.ndarray:
        prev = self.state.get(key)
        cur = probs if prev is None or prev.shape != probs.shape else self.alpha * probs + (1 - self.alpha) * prev
        self.state[key] = cur
        return cur

    def reset(self, key: Optional[str] = None) -> None:
        if key is None:
            self.state.clear()
        else:
            self.state.pop(key, None)


@dataclass
class AlertEvent:
    kind: str           # e.g. "threat" or "nsfw"
    status: str         # "raised" | "cleared"
    label: str
    score: float
    timestamp: float
    reasons: Optional[List[str]] = None


class AlertStateMachine:
    def __init__(self, kind: str, on: float = 0.7, off: float = 0.4, min_on: int = 2, min_off: int = 3,
                 cooldown_s: float = 30.0):
        if off >= on:
            raise ValueError("off threshold must be lower than on threshold")
        self.kind, self.on, self.off = kind, on, off
        self.min_on, self.min_off, self.cooldown_s = min_on, min_off, cooldown_s
        self.active = False
        self.label: Optional[str] = None
        self._above = self._below = 0
        self._last_raise = -1e18

    def update(self, score: float, label: str, now: float, reasons: Optional[List[str]] = None) -> Optional[AlertEvent]:
        if not self.active:
            self._above = self._above + 1 if score >= self.on else 0
            if self._above >= self.min_on and (now - self._last_raise) >= self.cooldown_s:
                self.active, self.label, self._below, self._last_raise = True, label, 0, now
                return AlertEvent(self.kind, "raised", label, round(float(score), 4), now, reasons)
        else:
            self._below = self._below + 1 if score < self.off else 0
            if self._below >= self.min_off:
                self.active, self._above = False, 0
                ev = AlertEvent(self.kind, "cleared", self.label or label, round(float(score), 4), now)
                self.label = None
                return ev
        return None
