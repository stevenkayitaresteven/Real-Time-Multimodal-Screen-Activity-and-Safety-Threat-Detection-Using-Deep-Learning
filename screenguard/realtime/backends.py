"""Inference backends with a common ``infer(image, ids) -> {task: logits}`` API.

* :class:`TorchBackend` – PyTorch checkpoint (development, explanations).
* :class:`OnnxBackend`  – ONNX Runtime (fp32 or int8), the deployment path for
  Windows/macOS/Linux/Android/iOS without a PyTorch dependency.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class Backend:
    meta: Dict

    def infer(self, image: np.ndarray, ids: np.ndarray) -> Dict[str, np.ndarray]:  # pragma: no cover
        raise NotImplementedError


class TorchBackend(Backend):
    def __init__(self, checkpoint: str, device: str = "cpu", threads: Optional[int] = None):
        import torch

        from ..training import load_checkpoint

        if threads:
            torch.set_num_threads(threads)
        model, ck = load_checkpoint(checkpoint, map_location=device)
        self.torch = torch
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        cfg = ck["config_obj"]
        self.meta = {"tasks": ck["tasks"], "tasks_trained": ck.get("tasks_trained", list(ck["tasks"])),
                     "temperatures": ck.get("temperatures", {}), "img_size": list(cfg.data.img_size),
                     "max_text_len": cfg.data.max_text_len, "modalities": list(cfg.model.modalities),
                     "vision": cfg.model.vision, "fusion": cfg.model.fusion, "backend": "torch"}

    def infer(self, image: np.ndarray, ids: np.ndarray) -> Dict[str, np.ndarray]:
        t = self.torch
        with t.inference_mode():
            out = self.model(t.from_numpy(image).to(self.device), t.from_numpy(ids).to(self.device))
        return {k: v.float().cpu().numpy() for k, v in out.items() if not k.startswith("_")}


class OnnxBackend(Backend):
    def __init__(self, path: str, threads: Optional[int] = None):
        import onnxruntime as ort

        meta_path = Path(path).with_suffix(".json")
        if not meta_path.exists():
            meta_path = Path(str(path).replace(".int8.onnx", ".onnx")).with_suffix(".json")
        self.meta = json.loads(meta_path.read_text())
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.meta["backend"] = "onnx" + ("-int8" if ".int8." in str(path) else "")

    def infer(self, image: np.ndarray, ids: np.ndarray) -> Dict[str, np.ndarray]:
        feeds = {}
        for name in self.input_names:
            feeds[name] = image.astype(np.float32) if name == "image" else ids.astype(np.int64)
        outs = self.sess.run(None, feeds)
        return dict(zip(self.meta["tasks"].keys(), outs))


def load_backend(path: str, **kw) -> Backend:
    if str(path).endswith(".onnx"):
        return OnnxBackend(path, threads=kw.get("threads"))
    return TorchBackend(path, device=kw.get("device", "cpu"), threads=kw.get("threads"))
