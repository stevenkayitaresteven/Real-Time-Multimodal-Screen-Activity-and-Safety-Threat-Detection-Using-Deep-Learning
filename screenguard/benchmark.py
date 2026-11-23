"""Latency / size benchmarks for edge deployment decisions.

Latency does not depend on trained weights, so architecture variants are
benchmarked with randomly initialised models; this allows comparing
backbones and runtimes without training each one.
"""
from __future__ import annotations

import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Config
from .models.multitask import ExportWrapper, build_model, count_parameters
from .taxonomy import TASKS


def time_fn(fn: Callable[[], object], n: int = 50, warmup: int = 5) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"mean_ms": round(statistics.mean(ts), 2), "p50_ms": round(ts[len(ts) // 2], 2),
            "p95_ms": round(ts[min(len(ts) - 1, int(0.95 * len(ts)))], 2)}


def _inputs(img_size: Tuple[int, int], max_len: int, batch: int = 1):
    img = np.random.randn(batch, 3, *img_size).astype(np.float32)
    ids = np.random.randint(4, 200, (batch, max_len)).astype(np.int64)
    ids[:, 0] = 1
    ids[:, 600:] = 0
    return img, ids


def benchmark_variant(vision: str, fusion: str = "attention", modalities: Sequence[str] = ("image", "text"),
                      img_size: Tuple[int, int] = (144, 256), max_len: int = 1024, threads: int = 1, n: int = 50,
                      runtimes: Sequence[str] = ("torch", "torch-int8", "onnx", "onnx-int8"),
                      model: Optional[torch.nn.Module] = None) -> Dict:
    torch.set_num_threads(threads)
    if model is None:
        cfg = Config()
        cfg.model.vision, cfg.model.fusion, cfg.model.modalities = vision, fusion, list(modalities)
        model = build_model(cfg.model, TASKS, img_size, max_len)
    model.eval()
    img, ids = _inputs(img_size, max_len)
    ti, tids = torch.from_numpy(img), torch.from_numpy(ids)
    row: Dict = {"vision": vision, "fusion": fusion, "modalities": "+".join(modalities), "threads": threads,
                 "params_m": round(count_parameters(model) / 1e6, 3)}
    wrapper = ExportWrapper(model).eval()
    if "torch" in runtimes:
        with torch.inference_mode():
            row["torch"] = time_fn(lambda: wrapper(ti, tids), n)
    if "torch-int8" in runtimes:
        try:
            qm = torch.ao.quantization.quantize_dynamic(wrapper, {torch.nn.Linear}, dtype=torch.qint8)
            with torch.inference_mode():
                row["torch-int8"] = time_fn(lambda: qm(ti, tids), n)
        except Exception as exc:  # pragma: no cover
            row["torch-int8"] = {"error": str(exc)[:80]}
    if any(r.startswith("onnx") for r in runtimes):
        import warnings

        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.onnx"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(wrapper, (ti, tids), str(p), input_names=["image", "text_ids"], opset_version=17,
                                  dynamic_axes={"image": {0: "b"}, "text_ids": {0: "b"}}, dynamo=False)
            row["onnx_mb"] = round(p.stat().st_size / 2**20, 2)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            feeds = {"image": img, "text_ids": ids}
            if "onnx" in runtimes:
                s = ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
                row["onnx"] = time_fn(lambda: s.run(None, feeds), n)
            if "onnx-int8" in runtimes:
                from onnxruntime.quantization import QuantType, quantize_dynamic

                q = Path(d) / "m.int8.onnx"
                quantize_dynamic(str(p), str(q), weight_type=QuantType.QInt8)
                row["onnx_int8_mb"] = round(q.stat().st_size / 2**20, 2)
                s = ort.InferenceSession(str(q), opts, providers=["CPUExecutionProvider"])
                row["onnx-int8"] = time_fn(lambda: s.run(None, feeds), n)
    return row


def benchmark_components(n: int = 20, with_ocr: bool = True) -> Dict:
    """Per-stage cost of the non-neural parts of the pipeline on a 1280x720 frame."""
    from .analysis.rules import analyze_text
    from .data.preprocess import preprocess
    from .data.synthetic import ScreenGenerator
    from .ocr.engine import TesseractOCR
    from .realtime.change_detector import ChangeDetector
    from .text.tokenizer import ByteTokenizer

    gen = ScreenGenerator(1280, 720, mobile_p=0.0, seed=5)
    samples = [gen.generate(i) for i in range(8)]
    cd = ChangeDetector()
    tok = ByteTokenizer(1024)
    k = iter(range(10**9))
    res = {
        "change_detector": time_fn(lambda: cd.update(samples[next(k) % 8].image, 0.0), n),
        "preprocess_letterbox": time_fn(lambda: preprocess(samples[0].image, (144, 256)), n),
        "tokenize": time_fn(lambda: tok.encode(samples[0].text), n),
        "rules_and_urls": time_fn(lambda: analyze_text(samples[0].text), n),
    }
    if with_ocr:
        ocr = TesseractOCR()
        if ocr.available:
            res["ocr_tesseract_720p"] = time_fn(lambda: ocr.read(samples[1].image), max(3, n // 5), warmup=1)
    return res


def to_markdown(rows: List[Dict]) -> str:
    cols = ["vision", "fusion", "modalities", "threads", "params_m", "onnx_mb", "onnx_int8_mb"]
    rts = ["torch", "torch-int8", "onnx", "onnx-int8"]
    head = cols + [f"{r} mean/p95 ms" for r in rts]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(r.get(c, "")) for c in cols]
        for rt in rts:
            v = r.get(rt)
            cells.append(f"{v['mean_ms']:.1f} / {v['p95_ms']:.1f}" if isinstance(v, dict) and "mean_ms" in v else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
