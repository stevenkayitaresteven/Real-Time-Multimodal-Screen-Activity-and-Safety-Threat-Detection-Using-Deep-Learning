"""Explanations for a single prediction.

* **Grad-CAM** (Selvaraju et al., 2017) on the visual tokens: works for every
  backbone because all encoders expose a token grid.
* **Gradient x input** saliency on byte embeddings, aggregated to words: shows
  which words / URL fragments drove the decision.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..data.preprocess import letterbox, preprocess
from ..text.tokenizer import BYTE_OFFSET, SEP_ID, ByteTokenizer


def _jet(x: np.ndarray) -> np.ndarray:
    """Map [0,1] -> RGB uint8 (approximation of the 'jet' colormap)."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def _letterbox_geometry(w: int, h: int, size_hw: Tuple[int, int]):
    th, tw = size_hw
    s = min(tw / w, th / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return (tw - nw) // 2, (th - nh) // 2, nw, nh


def word_saliency(ids: np.ndarray, scores: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
    words: Dict[str, float] = {}
    cur, cur_s = bytearray(), 0.0

    def flush():
        nonlocal cur, cur_s
        if cur:
            w = cur.decode("utf-8", errors="replace").strip(".,;:!?()[]{}\"'")
            if len(w) > 1:
                words[w] = max(words.get(w, 0.0), cur_s)
        cur, cur_s = bytearray(), 0.0

    for i, s in zip(ids, scores):
        i = int(i)
        if i == SEP_ID or i < BYTE_OFFSET or (i - BYTE_OFFSET) in (32, 9):
            flush()
            continue
        cur.append(i - BYTE_OFFSET)
        cur_s += float(s)
    flush()
    total = sum(words.values()) or 1.0
    return [(w, round(s / total, 4)) for w, s in sorted(words.items(), key=lambda kv: -kv[1])[:top_k]]


def explain(model, image: Image.Image, text: str, img_size: Tuple[int, int], max_text_len: int, task: str = "threat",
            target: Optional[int] = None, alpha: float = 0.45) -> Dict:
    model.eval()
    tok = ByteTokenizer(max_text_len)
    x = torch.from_numpy(preprocess(image, img_size))[None]
    ids_np = tok.encode(text)
    ids = torch.from_numpy(ids_np)[None]
    with torch.enable_grad():
        out = model(x if model.use_image else None, ids if model.use_text else None, return_features=True)
        vt, emb = out["_vision_tokens"], out["_text_embeddings"]
        if vt is not None:
            vt.retain_grad()
        if emb is not None:
            emb.retain_grad()
        logits = out[task][0]
        cls = int(logits.argmax()) if target is None else int(target)
        model.zero_grad(set_to_none=True)
        logits[cls].backward()
    result: Dict = {"task": task, "class_index": cls, "logit": float(logits[cls].detach())}
    if vt is not None and vt.grad is not None:
        a, g = vt[0].detach(), vt.grad[0]
        cam = torch.relu((a * g.mean(0)).sum(-1))
        gh, gw = out["_grid"]
        cam = cam.reshape(gh, gw).numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        th, tw = img_size
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
        ox, oy, nw, nh = _letterbox_geometry(image.width, image.height, img_size)
        cam_img = cam_img.crop((ox, oy, ox + nw, oy + nh)).resize(image.size, Image.BILINEAR)
        heat = np.asarray(cam_img, dtype=np.float32) / 255.0
        base = np.asarray(image.convert("RGB"), dtype=np.float32)
        overlay = (base * (1 - alpha * heat[..., None]) + _jet(heat).astype(np.float32) * alpha * heat[..., None])
        result["heatmap"] = heat
        result["overlay"] = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    if emb is not None and emb.grad is not None:
        sal = (emb.grad[0] * emb[0].detach()).sum(-1).abs().numpy()
        result["top_words"] = word_saliency(ids_np, sal)
    return result
