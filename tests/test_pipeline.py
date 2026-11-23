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
