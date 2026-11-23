"""Frame sources: live screen, video file, image folder, scripted synthetic session."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from PIL import Image


@dataclass
class Frame:
    image: Image.Image
    index: int
    timestamp: float
    meta: Optional[dict] = None


class FrameSource:
    live = False  # live sources are paced by wall-clock; offline ones by the requested fps

    def __iter__(self) -> Iterator[Frame]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class ScreenSource(FrameSource):
    """Captures the desktop with `mss` (Windows, macOS, Linux/X11)."""

    live = True

    def __init__(self, monitor: int = 1, region: Optional[Tuple[int, int, int, int]] = None):
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("Screen capture needs `pip install mss`") from exc
        self._sct = mss.mss()
        if region:
            x, y, w, h = region
            self.area = {"left": x, "top": y, "width": w, "height": h}
        else:
            self.area = self._sct.monitors[monitor]

    def __iter__(self) -> Iterator[Frame]:
        i = 0
        while True:
            shot = self._sct.grab(self.area)
            yield Frame(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"), i, time.time())
            i += 1

    def close(self) -> None:
        self._sct.close()


class VideoSource(FrameSource):
    """Reads a screen recording with OpenCV, sub-sampled to ``fps``."""

    def __init__(self, path: str, fps: float = 2.0):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Video input needs `pip install opencv-python`") from exc
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video {path}")
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.step = max(1, int(round(self.src_fps / fps)))

    def __iter__(self) -> Iterator[Frame]:
        i = n = 0
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                return
            if n % self.step == 0:
                rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
                yield Frame(Image.fromarray(rgb), i, n / self.src_fps)
                i += 1
            n += 1

    def close(self) -> None:
        self.cap.release()


class ImageDirSource(FrameSource):
    EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, path: str, loop: bool = False):
        self.paths = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in self.EXTS)
        if not self.paths:
            raise ValueError(f"No images in {path}")
        self.loop = loop

    def __iter__(self) -> Iterator[Frame]:
        i = 0
        while True:
            for p in self.paths:
                with Image.open(p) as im:
                    yield Frame(im.convert("RGB"), i, float(i), {"path": str(p)})
                i += 1
            if not self.loop:
                return


class SyntheticSession(FrameSource):
    """A scripted 'user session' built from synthetic screens, for demos and
    end-to-end tests without screen access.

    Each segment shows one screen for ``hold`` frames.  Consecutive frames in a
    segment are identical except for occasional small changes, which exercises
    the change detector (frame skipping) and the temporal smoother.
    """

    DEFAULT_SCRIPT: Sequence[Tuple[str, str]] = (
        ("document_editing", "benign"), ("web_browsing", "benign"), ("email", "benign"), ("email", "phishing"),
        ("code_editor", "benign"), ("terminal", "benign"), ("terminal", "suspicious_command"),
        ("video_streaming", "benign"), ("web_browsing", "scam"), ("messaging", "benign"),
        ("file_download", "malware_download"), ("social_media", "benign"), ("web_browsing", "phishing"),
    )

    def __init__(self, script: Optional[Sequence[Tuple[str, str]]] = None, hold: int = 6, seed: int = 7,
                 width: int = 1280, height: int = 720, flicker_p: float = 0.15):
        from ..data.synthetic import ScreenGenerator

        self.script = list(script or self.DEFAULT_SCRIPT)
        self.hold = hold
        self.gen = ScreenGenerator(width, height, mobile_p=0.0, seed=seed)
        self.rng = random.Random(seed)
        self.flicker_p = flicker_p

    def __iter__(self) -> Iterator[Frame]:
        i = 0
        for seg, (activity, threat) in enumerate(self.script):
            s = self.gen.sample(random.Random(self.rng.random()), activity=activity, threat=threat, device="desktop")
            base = s.image
            for k in range(self.hold):
                img = base
                if k and self.rng.random() < self.flicker_p:  # e.g. a blinking cursor / clock tick
                    img = base.copy()
                    x, y = self.rng.randrange(base.width - 20), self.rng.randrange(base.height - 20)
                    img.paste((255, 255, 255), (x, y, x + 6, y + 14))
                yield Frame(img, i, float(i), {"segment": seg, "activity": activity, "threat": threat,
                                               "text": s.text, "url": s.url})
                i += 1


def open_source(spec: str, fps: float = 2.0) -> FrameSource:
    """Parse a source spec: ``screen[:N]``, ``video:PATH``, ``dir:PATH``, ``synthetic``."""
    if spec.startswith("screen"):
        mon = int(spec.split(":", 1)[1]) if ":" in spec else 1
        return ScreenSource(mon)
    if spec.startswith("video:"):
        return VideoSource(spec[6:], fps)
    if spec.startswith("dir:"):
        return ImageDirSource(spec[4:])
    if spec.startswith("synthetic"):
        hold = int(spec.split(":", 1)[1]) if ":" in spec else 6
        return SyntheticSession(hold=hold)
    p = Path(spec)
    if p.is_dir():
        return ImageDirSource(spec)
    if p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}:
        return VideoSource(spec, fps)
    raise ValueError(f"Unrecognised source {spec!r}")
