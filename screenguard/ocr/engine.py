"""OCR engines for screen text.

Tesseract is used by default because it is available on every desktop OS and
on Android (tess-two) without GPU or model downloads.  Two screen-specific
tweaks matter a lot in practice:

* dark-themed UIs (terminals, IDEs, dark mode) are inverted before OCR, since
  Tesseract is trained on dark-on-light text;
* small UI fonts are upscaled, since Tesseract wants ~20-30 px glyph heights.

:class:`AsyncOCR` runs OCR on a background thread with "latest frame wins"
semantics, so the real-time loop never blocks on OCR; the model consumes the
most recent finished OCR result.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


@dataclass
class OCRWord:
    text: str
    conf: float
    box: Tuple[int, int, int, int]  # x0, y0, x1, y1 in original image coordinates


@dataclass
class OCRResult:
    text: str
    words: List[OCRWord] = field(default_factory=list)
    latency_ms: float = 0.0
    engine: str = "none"


class OCREngine:
    name = "base"

    def read(self, image: Image.Image) -> OCRResult:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True


class NullOCR(OCREngine):
    name = "none"

    def read(self, image: Image.Image) -> OCRResult:
        return OCRResult("", engine=self.name)


class TesseractOCR(OCREngine):
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 11, min_conf: float = 30.0, target_height: int = 1080,
                 max_upscale: float = 2.0, cmd: Optional[str] = None, timeout_s: float = 30.0, threads: int = 1):
        self.lang, self.psm, self.min_conf = lang, psm, min_conf
        # Tesseract's internal OpenMP threading oversubscribes the CPU when it runs next to the model or in
        # parallel workers; one thread per call is faster overall in that setting.
        self.env = {**os.environ, "OMP_THREAD_LIMIT": str(threads)}
        self.target_height, self.max_upscale = target_height, max_upscale
        self.cmd = cmd or shutil.which("tesseract")
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return self.cmd is not None

    def _prepare(self, image: Image.Image) -> Tuple[Image.Image, float]:
        g = ImageOps.grayscale(image)
        if np.asarray(g.resize((64, 36))).mean() < 110:  # dark theme -> invert to dark-on-light
            g = ImageOps.invert(g)
        scale = min(self.max_upscale, max(1.0, self.target_height / max(1, g.height)))
        if scale > 1.01:
            g = g.resize((int(g.width * scale), int(g.height * scale)), Image.BICUBIC)
        return g, scale

    def read(self, image: Image.Image) -> OCRResult:
        if not self.available:
            raise RuntimeError("tesseract binary not found; install it (e.g. `apt install tesseract-ocr`, "
                               "`brew install tesseract`, or the UB-Mannheim build on Windows)")
        t0 = time.perf_counter()
        g, scale = self._prepare(image)
        buf = io.BytesIO()
        g.save(buf, format="PNG")
        try:
            proc = subprocess.run([self.cmd, "stdin", "stdout", "-l", self.lang, "--psm", str(self.psm), "tsv"],
                                  input=buf.getvalue(), capture_output=True, timeout=self.timeout_s, env=self.env)
        except subprocess.TimeoutExpired:  # degrade gracefully: the visual pathway still works without text
            return OCRResult("", [], (time.perf_counter() - t0) * 1000, f"{self.name}:timeout")
        words: List[OCRWord] = []
        lines: dict = {}
        for row in proc.stdout.decode("utf-8", errors="replace").splitlines()[1:]:
            cols = row.split("\t")
            if len(cols) < 12 or not cols[11].strip():
                continue
            try:
                conf = float(cols[10])
            except ValueError:
                continue
            if conf < self.min_conf:
                continue
            l, t, w, h = (int(int(c) / scale) for c in cols[6:10])
            words.append(OCRWord(cols[11], conf, (l, t, l + w, t + h)))
            key = (int(cols[2]), int(cols[3]), int(cols[4]))  # block, paragraph, line
            lines.setdefault(key, []).append((l, t, cols[11]))
        ordered = sorted(lines.values(), key=lambda ws: (min(w[1] for w in ws) // 8, min(w[0] for w in ws)))
        text = "\n".join(" ".join(w[2] for w in sorted(ws)) for ws in ordered)
        return OCRResult(text, words, (time.perf_counter() - t0) * 1000, self.name)


def build_ocr(name: str = "auto", **kw) -> OCREngine:
    if name == "none":
        return NullOCR()
    if name in ("auto", "tesseract"):
        eng = TesseractOCR(**kw)
        if eng.available:
            return eng
        if name == "tesseract":
            raise RuntimeError("tesseract requested but not installed")
        return NullOCR()
    raise ValueError(f"Unknown OCR engine {name!r}")


class AsyncOCR:
    """Background OCR worker: ``submit`` never blocks; ``latest`` returns the newest result."""

    def __init__(self, engine: OCREngine):
        self.engine = engine
        self._pending: Optional[Tuple[Image.Image, int]] = None
        self._result: Tuple[OCRResult, int] = (OCRResult("", engine=engine.name), -1)
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, image: Image.Image, frame_id: int) -> None:
        with self._cv:
            self._pending = (image.copy(), frame_id)
            self._cv.notify()

    def latest(self) -> Tuple[OCRResult, int]:
        with self._cv:
            return self._result

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                image, fid = self._pending
                self._pending = None
            try:
                res = self.engine.read(image)
            except Exception as exc:  # keep the worker alive; surface the error in the result
                res = OCRResult("", engine=f"{self.engine.name}:error:{exc.__class__.__name__}")
            with self._cv:
                self._result = (res, fid)

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=2)
