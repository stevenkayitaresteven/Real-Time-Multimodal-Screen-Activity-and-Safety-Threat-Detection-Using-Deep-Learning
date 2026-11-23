"""Export trained models for on-device deployment.

* ONNX (fp32) – runs with ONNX Runtime on Windows / macOS / Linux / Android / iOS
  (and in browsers via onnxruntime-web).
* ONNX dynamic int8 – int8 MatMul/Gemm weights (Transformer + MLPs); convolutions stay fp32.
* TorchScript (optional) – for PyTorch Mobile style deployment.

Every export is verified numerically against the PyTorch model.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .models.multitask import ExportWrapper
from .training import load_checkpoint


def _dummy_inputs(meta_cfg, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    h, w = meta_cfg.data.img_size
    img = torch.randn(2, 3, h, w, generator=g)
    ids = torch.randint(4, 200, (2, meta_cfg.data.max_text_len), generator=g)
    ids[:, 0] = 1
    ids[1, 300:] = 0  # include padding
    return img, ids


def export_onnx(checkpoint: str, out_path: str, opset: int = 17, quantize: bool = True, verify: bool = True) -> Dict:
    model, ck = load_checkpoint(checkpoint)
    cfg = ck["config_obj"]
    wrapper = ExportWrapper(model).eval()
    img, ids = _dummy_inputs(cfg)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [f"{t}_logits" for t in model.task_names]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(wrapper, (img, ids), str(out), input_names=["image", "text_ids"], output_names=names,
                          dynamic_axes={"image": {0: "batch"}, "text_ids": {0: "batch"}, **{n: {0: "batch"} for n in names}},
                          opset_version=opset, dynamo=False)
    meta = {"tasks": ck["tasks"], "tasks_trained": ck.get("tasks_trained"), "temperatures": ck.get("temperatures", {}),
            "img_size": list(cfg.data.img_size), "max_text_len": cfg.data.max_text_len,
            "modalities": list(cfg.model.modalities), "vision": cfg.model.vision, "fusion": cfg.model.fusion,
            "inputs": {"image": "float32 [B,3,H,W], RGB, letterboxed, ImageNet-normalised",
                       "text_ids": "int64 [B,L], byte tokens: 0=PAD 1=CLS 2=newline, byte+4"}}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    report: Dict = {"onnx": str(out), "onnx_mb": round(out.stat().st_size / 2**20, 3)}

    if verify or quantize:
        import onnxruntime as ort

        with torch.no_grad():
            ref = [t.numpy() for t in wrapper(img, ids)]
        feeds = {"image": img.numpy(), "text_ids": ids.numpy()}

        def check(path: str) -> Dict:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            got = sess.run(None, feeds)
            diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
            agree = float(np.mean([np.mean(a.argmax(-1) == b.argmax(-1)) for a, b in zip(ref, got)]))
            return {"max_abs_diff": round(diff, 6), "argmax_agreement": agree}

        report["onnx_check"] = check(str(out))
        if quantize:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            q = out.with_name(out.stem + ".int8.onnx")
            # Quantise only MatMul/Gemm (Transformer + MLP weights). Dynamic ConvInteger kernels are slower than
            # fp32 convolutions on most CPUs, so convolutions stay in fp32.
            quantize_dynamic(str(out), str(q), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
            report["onnx_int8"] = str(q)
            report["onnx_int8_mb"] = round(q.stat().st_size / 2**20, 3)
            report["onnx_int8_check"] = check(str(q))
    return report


def export_torchscript(checkpoint: str, out_path: str) -> Dict:
    model, ck = load_checkpoint(checkpoint)
    wrapper = ExportWrapper(model).eval()
    img, ids = _dummy_inputs(ck["config_obj"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(wrapper, (img, ids), check_trace=False)
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile

            traced = optimize_for_mobile(traced)
        except Exception:  # pragma: no cover - optional optimisation
            pass
        traced.save(out_path)
    with torch.no_grad():
        ref = wrapper(img, ids)
        got = torch.jit.load(out_path)(img, ids)
    diff = max(float((a - b).abs().max()) for a, b in zip(ref, got))
    return {"torchscript": out_path, "mb": round(Path(out_path).stat().st_size / 2**20, 3), "max_abs_diff": round(diff, 6)}
