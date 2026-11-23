"""End-to-end near-real-time screen analyser.

Per frame::

    capture ─► change detector ──(unchanged)──► reuse previous result (no OCR, no model)
                     │ changed
                     ▼
                OCR (sync or async) ─► tokenizer ─┐
                letterbox + normalise ────────────┴─► model ─► temperature-scaled probs
                                                            │
                rules + URL analysis ───────────────────────┴─► decision fusion (threat)
                                                                     │
                                                    EMA smoothing ─► alert hysteresis ─► events
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from ..analysis.decision import fuse_threat
from ..analysis.rules import analyze_text
from ..data.preprocess import preprocess
from ..metrics import softmax
from ..ocr.engine import AsyncOCR, OCREngine, NullOCR
from ..privacy import redact
from ..text.tokenizer import ByteTokenizer
from .backends import Backend
from .capture import FrameSource
from .change_detector import ChangeDetector
from .smoothing import AlertStateMachine, EMASmoother


@dataclass
class AnalyzerConfig:
    decision: str = "hybrid"          # model | rules | hybrid
    ema_alpha: float = 0.6
    threat_on: float = 0.70
    threat_off: float = 0.40
    nsfw_on: float = 0.80
    nsfw_off: float = 0.50
    min_on: int = 2
    min_off: int = 3
    cooldown_s: float = 30.0
    async_ocr: bool = False
    change_hash_bits: int = 8
    change_pixel_delta: float = 2.0
    max_interval_s: float = 10.0
    skip_unchanged: bool = True


class ScreenAnalyzer:
    def __init__(self, backend: Backend, ocr: Optional[OCREngine] = None, cfg: Optional[AnalyzerConfig] = None):
        self.backend = backend
        self.cfg = cfg or AnalyzerConfig()
        self.ocr = ocr or NullOCR()
        self.async_ocr = AsyncOCR(self.ocr) if (self.cfg.async_ocr and not isinstance(self.ocr, NullOCR)) else None
        m = backend.meta
        self.tasks: Dict[str, List[str]] = m["tasks"]
        self.trained = set(m.get("tasks_trained", self.tasks.keys()))
        self.temps: Dict[str, float] = m.get("temperatures", {})
        self.img_size = tuple(m["img_size"])
        self.tok = ByteTokenizer(m["max_text_len"])
        self.change = ChangeDetector(self.cfg.change_hash_bits, self.cfg.change_pixel_delta, self.cfg.max_interval_s)
        self.ema = EMASmoother(self.cfg.ema_alpha)
        self.alerts = {"threat": AlertStateMachine("threat", self.cfg.threat_on, self.cfg.threat_off, self.cfg.min_on,
                                                   self.cfg.min_off, self.cfg.cooldown_s),
                       "nsfw": AlertStateMachine("nsfw", self.cfg.nsfw_on, self.cfg.nsfw_off, self.cfg.min_on,
                                                 self.cfg.min_off, self.cfg.cooldown_s)}
        self._last: Optional[Dict] = None
        self._frame_id = 0

    # ------------------------------------------------------------------ #
    def _probs(self, logits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {t: softmax(z[0] / self.temps.get(t, 1.0)) for t, z in logits.items()}

    def analyze(self, image: Image.Image, now: Optional[float] = None, text: Optional[str] = None) -> Dict:
        """Analyse one frame. ``text`` overrides OCR (useful for tests / ground-truth evaluation)."""
        t_start = time.perf_counter()
        now = time.time() if now is None else now
        fid = self._frame_id
        self._frame_id += 1
        lat: Dict[str, float] = {}

        t0 = time.perf_counter()
        changed, info = self.change.update(image, now)
        lat["change"] = (time.perf_counter() - t0) * 1000
        if self.cfg.skip_unchanged and not changed and self._last is not None:
            res = dict(self._last)
            res.update({"frame": fid, "t": now, "analyzed": False, "change": info, "alerts": [],
                        "latency_ms": {"change": round(lat["change"], 2), "total": round((time.perf_counter() - t_start) * 1000, 2)}})
            return res

        # --- OCR --------------------------------------------------------- #
        t0 = time.perf_counter()
        ocr_engine = "override"
        if text is None:
            if self.async_ocr is not None:
                self.async_ocr.submit(image, fid)
                result, _ = self.async_ocr.latest()
                text, ocr_engine = result.text, f"{result.engine}(async)"
            else:
                result = self.ocr.read(image)
                text, ocr_engine = result.text, result.engine
        lat["ocr"] = (time.perf_counter() - t0) * 1000

        # --- model ------------------------------------------------------- #
        t0 = time.perf_counter()
        x = preprocess(image, self.img_size)[None]
        ids = self.tok.encode(text)[None]
        lat["preprocess"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        probs = self._probs(self.backend.infer(x, ids))
        lat["model"] = (time.perf_counter() - t0) * 1000

        # --- rules + decision ------------------------------------------- #
        t0 = time.perf_counter()
        ev = analyze_text(text)
        threat_classes = self.tasks["threat"]
        p_threat_model = probs["threat"] if "threat" in self.trained else np.eye(len(threat_classes))[0]
        p_threat = fuse_threat(p_threat_model, ev.class_scores, threat_classes, self.cfg.decision)
        lat["rules"] = (time.perf_counter() - t0) * 1000

        # --- temporal smoothing + alerts --------------------------------- #
        p_act = self.ema.update("activity", probs["activity"])
        p_thr = self.ema.update("threat", p_threat)
        alerts = []
        thr_label = threat_classes[int(np.argmax(p_thr[1:])) + 1]
        risk = float(1.0 - p_thr[0])
        evt = self.alerts["threat"].update(risk, thr_label, now, ev.reasons())
        if evt:
            alerts.append(asdict(evt))
        nsfw = {"available": False}
        if "nsfw" in self.trained and "nsfw" in probs:
            p_nsfw = self.ema.update("nsfw", probs["nsfw"])
            nsfw_classes = self.tasks["nsfw"]
            nsfw_score = float(1.0 - p_nsfw[0])
            nsfw = {"available": True, "label": nsfw_classes[int(np.argmax(p_nsfw))], "score": round(nsfw_score, 4),
                    "probs": {c: round(float(p), 4) for c, p in zip(nsfw_classes, p_nsfw)}}
            evt = self.alerts["nsfw"].update(nsfw_score, nsfw_classes[int(np.argmax(p_nsfw[1:])) + 1], now)
            if evt:
                alerts.append(asdict(evt))

        act_classes = self.tasks["activity"]
        top = np.argsort(-p_act)[:3]
        res = {
            "frame": fid, "t": now, "analyzed": True, "change": info,
            "activity": {"label": act_classes[int(top[0])], "confidence": round(float(p_act[top[0]]), 4),
                         "top3": [[act_classes[int(i)], round(float(p_act[i]), 4)] for i in top]},
            "threat": {"label": threat_classes[int(np.argmax(p_thr))], "risk": round(risk, 4),
                       "probs": {c: round(float(p), 4) for c, p in zip(threat_classes, p_thr)},
                       "model_probs": {c: round(float(p), 4) for c, p in zip(threat_classes, p_threat_model)},
                       "rule_evidence": {k: round(v, 4) for k, v in ev.class_scores.items()},
                       "reasons": ev.reasons(), "rule_hits": [h.rule_id for h in ev.hits],
                       "urls": [{"domain": u.registered_domain, "score": round(u.score, 3)} for u in ev.urls[:5]]},
            "nsfw": nsfw,
            "alerts": alerts,
            "ocr": {"engine": ocr_engine, "chars": len(text)},
            "latency_ms": {k: round(v, 2) for k, v in lat.items()},
            "_text": text,  # kept in memory only; sinks never persist it unless explicitly asked (and then redacted)
        }
        res["latency_ms"]["total"] = round((time.perf_counter() - t_start) * 1000, 2)
        self._last = res
        return res

    # ------------------------------------------------------------------ #
    def run(self, source: FrameSource, fps: float = 2.0, max_frames: Optional[int] = None,
            on_result: Optional[Callable[[Dict, object], None]] = None) -> Dict:
        interval = 1.0 / fps if fps > 0 else 0.0
        results: List[Dict] = []
        events: List[Dict] = []
        try:
            for frame in source:
                t_loop = time.perf_counter()
                res = self.analyze(frame.image, now=frame.timestamp if not source.live else None)
                results.append(res)
                events.extend(res["alerts"])
                if on_result:
                    on_result(res, frame)
                if max_frames and len(results) >= max_frames:
                    break
                if source.live and interval:
                    time.sleep(max(0.0, interval - (time.perf_counter() - t_loop)))
        except KeyboardInterrupt:
            pass
        finally:
            source.close()
            if self.async_ocr:
                self.async_ocr.close()
        return summarize(results, events)


def summarize(results: List[Dict], events: List[Dict]) -> Dict:
    analyzed = [r for r in results if r["analyzed"]]
    tot = [r["latency_ms"]["total"] for r in analyzed]

    def mean_of(key):
        vals = [r["latency_ms"].get(key) for r in analyzed if key in r["latency_ms"]]
        return round(statistics.mean(vals), 2) if vals else None

    return {
        "frames": len(results), "analyzed": len(analyzed), "skipped_unchanged": len(results) - len(analyzed),
        "latency_ms_mean": {k: mean_of(k) for k in ("change", "ocr", "preprocess", "model", "rules", "total")},
        "latency_ms_p95_total": round(float(np.percentile(tot, 95)), 2) if tot else None,
        "alerts": events,
    }


class ConsoleSink:
    COLORS = {"benign": "\033[32m", "alert": "\033[1;31m", "reset": "\033[0m", "dim": "\033[2m"}

    def __init__(self, show_skipped: bool = False, stream=sys.stdout):
        self.show_skipped = show_skipped
        self.stream = stream
        self.color = stream.isatty()

    def _c(self, k: str) -> str:
        return self.COLORS[k] if self.color else ""

    def __call__(self, res: Dict, frame=None) -> None:
        if not res["analyzed"] and not self.show_skipped:
            return
        thr = res["threat"]
        tag = "" if res["analyzed"] else f"{self._c('dim')}(unchanged){self._c('reset')} "
        nsfw = f" nsfw={res['nsfw']['label']}" if res["nsfw"].get("available") else ""
        col = self._c("benign") if thr["label"] == "benign" else self._c("alert")
        print(f"[{res['frame']:05d}] {tag}activity={res['activity']['label']:<17s} ({res['activity']['confidence']:.2f}) "
              f"threat={col}{thr['label']}{self._c('reset')} risk={thr['risk']:.2f}{nsfw} "
              f"| {res['latency_ms'].get('total', 0):.0f} ms", file=self.stream)
        for a in res["alerts"]:
            if a["status"] == "raised":
                print(f"{self._c('alert')}  ⚠ ALERT [{a['kind']}] {a['label']} (score {a['score']:.2f}){self._c('reset')}", file=self.stream)
                for r in (a.get("reasons") or [])[:3]:
                    print(f"      - {r}", file=self.stream)
            else:
                print(f"  ✓ cleared [{a['kind']}] {a['label']}", file=self.stream)


class JsonlSink:
    """Privacy-preserving event log: no screen text unless ``log_text`` (then redacted)."""

    def __init__(self, path: str, log_text: bool = False, only_alerts: bool = False):
        self.fh = open(path, "a", encoding="utf-8")
        self.log_text = log_text
        self.only_alerts = only_alerts

    def __call__(self, res: Dict, frame=None) -> None:
        if self.only_alerts and not res["alerts"]:
            return
        rec = {k: res[k] for k in ("frame", "t", "analyzed", "activity", "threat", "nsfw", "alerts", "latency_ms")}
        if self.log_text:
            rec["text_redacted"] = redact(res.get("_text", ""))
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()
