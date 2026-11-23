"""Cheap frame-change detection to skip redundant inference.

Screens are static most of the time (reading, typing).  Comparing a 256-bit
difference hash plus a coarse pixel difference costs well under a
millisecond and lets the pipeline skip the model and OCR entirely when
nothing meaningful changed, which is the single biggest saving on battery
powered devices.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image


def dhash(img: Image.Image, size: int = 16) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def thumb(img: Image.Image, size: Tuple[int, int] = (64, 36)) -> np.ndarray:
    return np.asarray(img.convert("L").resize(size, Image.BILINEAR), dtype=np.float32)


class ChangeDetector:
    def __init__(self, hash_bits: int = 8, pixel_delta: float = 2.0, max_interval_s: float = 10.0):
        self.hash_bits = hash_bits
        self.pixel_delta = pixel_delta
        self.max_interval_s = max_interval_s
        self._hash: Optional[np.ndarray] = None
        self._thumb: Optional[np.ndarray] = None
        self._t_last: float = -1e9

    def update(self, img: Image.Image, now: float) -> Tuple[bool, dict]:
        """Return ``(changed, info)``; the reference frame updates only on change."""
        h, t = dhash(img), thumb(img)
        if self._hash is None:
            changed, bits, delta = True, -1, -1.0
        else:
            bits = int((h != self._hash).sum())
            delta = float(np.abs(t - self._thumb).mean())
            changed = bits >= self.hash_bits or delta >= self.pixel_delta or (now - self._t_last) >= self.max_interval_s
        if changed:
            self._hash, self._thumb, self._t_last = h, t, now
        return changed, {"hash_bits": bits, "pixel_delta": round(delta, 3)}

    def reset(self) -> None:
        self._hash = self._thumb = None
