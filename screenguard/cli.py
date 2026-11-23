"""Command-line interface: ``screenguard <command> --help``."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional


def _cmd_generate(a: argparse.Namespace) -> None:
    from .data.generate import generate_split

    for split, n, seed in (("train", a.train, a.seed), ("val", a.val, a.seed + 1), ("test", a.test, a.seed + 2)):
        if n <= 0:
            continue
        t = time.time()
        m, counts = generate_split(str(Path(a.out) / split), n, seed * 101 + 1, a.width, a.height, a.mobile_p, a.workers)
        print(f"{split}: {n} screens -> {m} ({time.time() - t:.1f}s)")


def _cmd_train(a: argparse.Namespace) -> None:
    from .config import Config
    from .training import train

    train(Config.load(a.config, a.set))


def _cmd_evaluate(a: argparse.Namespace) -> None:
    from .evaluate import evaluate_checkpoint, rows_to_markdown, summary_row

    rep = evaluate_checkpoint(a.checkpoint, a.manifest, a.text_field, a.ocr_noise, a.drop_text, threads=a.threads)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=2))
    print(rows_to_markdown([summary_row(Path(a.checkpoint).parent.name, rep)]))


def _cmd_export(a: argparse.Namespace) -> None:
    from .export import export_onnx, export_torchscript

    if a.format == "onnx":
        rep = export_onnx(a.checkpoint, a.out, quantize=not a.no_quantize)
    else:
        rep = export_torchscript(a.checkpoint, a.out)
    print(json.dumps(rep, indent=2))


def _cmd_benchmark(a: argparse.Namespace) -> None:
    from .benchmark import benchmark_components, benchmark_variant, to_markdown

    rows = []
    for v in a.variants:
        vision, _, fusion = v.partition("/")
        for th in a.threads:
            rows.append(benchmark_variant(vision, fusion or "attention", threads=th, n=a.n))
            print(f"  done {v} threads={th}", file=sys.stderr)
    md = to_markdown(rows)
    comp = benchmark_components(with_ocr=not a.no_ocr)
    print(md)
    print("\nPipeline components (1280x720 frame):")
    for k, v in comp.items():
        print(f"  {k:24s} mean {v['mean_ms']:8.2f} ms   p95 {v['p95_ms']:8.2f} ms")
    if a.out:
        Path(a.out).write_text(json.dumps({"models": rows, "components": comp}, indent=2))


def _build_analyzer(a: argparse.Namespace):
    from .ocr.engine import build_ocr
    from .realtime.backends import load_backend
    from .realtime.pipeline import AnalyzerConfig, ScreenAnalyzer

    backend = load_backend(a.model, threads=a.threads)
    cfg = AnalyzerConfig(decision=a.decision, async_ocr=getattr(a, "async_ocr", False))
    if hasattr(a, "min_on"):
        cfg.min_on = a.min_on
    return ScreenAnalyzer(backend, build_ocr(a.ocr), cfg)


def _cmd_analyze(a: argparse.Namespace) -> None:
    from PIL import Image

    an = _build_analyzer(a)
    an.cfg.skip_unchanged = False
    for path in a.images:
        with Image.open(path) as im:
            img = im.convert("RGB")
        text = Path(a.text_file).read_text() if a.text_file else None
        res = an.analyze(img, text=text)
        res.pop("_text", None)
        if a.json:
            print(json.dumps({"image": path, **res}, indent=2))
        else:
            thr = res["threat"]
            print(f"{path}\n  activity : {res['activity']['label']} ({res['activity']['confidence']:.2f})"
                  f"  top3={res['activity']['top3']}\n  threat   : {thr['label']} (risk {thr['risk']:.2f})"
                  f"  model={max(thr['model_probs'], key=thr['model_probs'].get)}  rules={thr['rule_evidence']}")
            for r in thr["reasons"]:
                print(f"     - {r}")
            print(f"  nsfw     : {res['nsfw'] if res['nsfw'].get('available') else 'n/a (model not trained on NSFW data)'}")
            print(f"  latency  : {res['latency_ms']}")
        if a.explain:
            _explain_to(a, img, an, path)


def _explain_to(a, img, an, path) -> None:
    from .explain.gradcam import explain
    from .realtime.backends import TorchBackend

    if not isinstance(an.backend, TorchBackend):
        print("  (explanations need a PyTorch checkpoint, not ONNX)")
        return
    text = an.ocr.read(img).text
    out = explain(an.backend.model, img, text, an.img_size, an.tok.max_len, task=a.explain_task)
    dst = Path(a.explain)
    if len(a.images) > 1:
        dst = dst.with_name(f"{dst.stem}_{Path(path).stem}{dst.suffix}")
    if "overlay" in out:
        out["overlay"].save(dst)
        print(f"  grad-cam : {dst}")
    if "top_words" in out:
        print(f"  top words: {out['top_words'][:8]}")


def _cmd_monitor(a: argparse.Namespace) -> None:
    from .realtime.capture import open_source
    from .realtime.pipeline import ConsoleSink, JsonlSink

    an = _build_analyzer(a)
    console = ConsoleSink(show_skipped=a.show_skipped)
    sinks = [console]
    if a.log:
        sinks.append(JsonlSink(a.log, log_text=a.log_text, only_alerts=a.only_alerts))

    def on_result(res, frame):
        for s in sinks:
            s(res, frame)

    src = open_source(a.source, a.fps)
    print(f"Monitoring {a.source} at <= {a.fps} fps with {an.backend.meta.get('backend')} / OCR={an.ocr.name} "
          f"(Ctrl+C to stop). Frames are processed on-device and never stored.")
    summary = an.run(src, a.fps, a.max_frames, on_result)
    print(json.dumps({k: v for k, v in summary.items() if k != "alerts"}, indent=2))
    print(f"alerts raised: {sum(1 for e in summary['alerts'] if e['status'] == 'raised')}")


def _cmd_ocr_manifest(a: argparse.Namespace) -> None:
    """Add real OCR output (field ``ocr_text``) to a manifest, for OCR-gap experiments."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    from .data.datasets import read_manifest
    from .ocr.engine import TesseractOCR

    ocr = TesseractOCR()
    recs = read_manifest(a.manifest)[: a.limit or None]

    out_dir = Path(a.out).resolve().parent

    def run(r):
        with Image.open(r["image"]) as im:
            res = ocr.read(im.convert("RGB"))
        rel = os.path.relpath(r["image"], out_dir)  # keep the manifest relocatable
        return {**r, "image": rel, "ocr_text": res.text, "ocr_ms": round(res.latency_ms, 1)}

    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(run, recs))
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} records with OCR text to {a.out}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser("screenguard", description="Real-time multimodal screen activity & safety threat detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="render a synthetic labelled screen dataset")
    g.add_argument("--out", default="data/synth")
    g.add_argument("--train", type=int, default=8000)
    g.add_argument("--val", type=int, default=1000)
    g.add_argument("--test", type=int, default=1500)
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=360)
    g.add_argument("--mobile-p", type=float, default=0.2)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--workers", type=int, default=0)
    g.set_defaults(fn=_cmd_generate)

    t = sub.add_parser("train", help="train a model")
    t.add_argument("--config", default="configs/default.yaml")
    t.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=5 model.fusion=gated")
    t.set_defaults(fn=_cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a checkpoint on manifests")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--manifest", nargs="+", default=["data/synth/test/manifest.jsonl"])
    e.add_argument("--text-field", default="text")
    e.add_argument("--ocr-noise", type=float, default=0.0)
    e.add_argument("--drop-text", action="store_true")
    e.add_argument("--threads", type=int, default=None)
    e.add_argument("--out", default=None)
    e.set_defaults(fn=_cmd_evaluate)

    x = sub.add_parser("export", help="export to ONNX (+int8) or TorchScript")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--out", default="exports/screenguard.onnx")
    x.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    x.add_argument("--no-quantize", action="store_true")
    x.set_defaults(fn=_cmd_export)

    b = sub.add_parser("benchmark", help="latency benchmark of model variants and pipeline stages")
    b.add_argument("--variants", nargs="+", default=["tiny_cnn/attention", "tiny_cnn/gated", "mobilenet_v3_small/attention",
                                                     "vit_tiny/attention"])
    b.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    b.add_argument("--n", type=int, default=30)
    b.add_argument("--no-ocr", action="store_true")
    b.add_argument("--out", default=None)
    b.set_defaults(fn=_cmd_benchmark)

    def add_runtime_args(sp):
        sp.add_argument("--model", required=True, help="checkpoint (.pt) or exported model (.onnx)")
        sp.add_argument("--ocr", default="auto", choices=["auto", "tesseract", "none"])
        sp.add_argument("--decision", default="hybrid", choices=["model", "rules", "hybrid"])
        sp.add_argument("--threads", type=int, default=None)

    an = sub.add_parser("analyze", help="analyse screenshot(s)")
    an.add_argument("images", nargs="+")
    add_runtime_args(an)
    an.add_argument("--text-file", default=None, help="use this text instead of OCR")
    an.add_argument("--json", action="store_true")
    an.add_argument("--explain", default=None, help="write a Grad-CAM overlay PNG here")
    an.add_argument("--explain-task", default="threat", choices=["activity", "threat", "nsfw"])
    an.set_defaults(fn=_cmd_analyze)

    m = sub.add_parser("monitor", help="real-time monitoring of the screen / a video / a folder / a synthetic session")
    m.add_argument("--source", default="screen", help="screen[:N] | video:PATH | dir:PATH | synthetic[:HOLD]")
    add_runtime_args(m)
    m.add_argument("--fps", type=float, default=2.0)
    m.add_argument("--async-ocr", action="store_true")
    m.add_argument("--max-frames", type=int, default=None)
    m.add_argument("--min-on", type=int, default=2, help="consecutive analysed frames above threshold before alerting")
    m.add_argument("--log", default=None, help="JSONL event log (no screen text by default)")
    m.add_argument("--log-text", action="store_true", help="also log OCR text (PII-redacted)")
    m.add_argument("--only-alerts", action="store_true")
    m.add_argument("--show-skipped", action="store_true")
    m.set_defaults(fn=_cmd_monitor)

    o = sub.add_parser("ocr-manifest", help="run Tesseract over a manifest and store `ocr_text`")
    o.add_argument("--manifest", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--limit", type=int, default=0)
    o.add_argument("--workers", type=int, default=4)
    o.set_defaults(fn=_cmd_ocr_manifest)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
