"""ScreenThreatNet: multimodal, multi-task screen understanding network.

    frame ──► vision encoder ──► visual tokens ─┐
                                                ├─► fusion ──► task features ─► activity head
    OCR text ─► byte encoder ──► text tokens ───┘                           ├─► NSFW head
                                                                            └─► threat head
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig
from .fusion import build_fusion
from .text import ByteTextEncoder
from .vision import build_vision_encoder


class TaskHead(nn.Module):
    def __init__(self, dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, n_classes))

    def forward(self, x):
        return self.net(x)


class ScreenThreatNet(nn.Module):
    def __init__(self, cfg: ModelConfig, tasks: Dict[str, List[str]], img_size: Tuple[int, int] = (144, 256),
                 max_text_len: int = 512):
        super().__init__()
        self.cfg = cfg
        self.task_names = list(tasks.keys())
        self.use_image = "image" in cfg.modalities
        self.use_text = "text" in cfg.modalities
        if not (self.use_image or self.use_text):
            raise ValueError("At least one modality is required")
        self.vision = build_vision_encoder(cfg.vision, cfg.dim, cfg.pretrained, img_size,
                                           vit_patch=cfg.vit_patch, vit_depth=cfg.vit_depth, vit_dim=cfg.vit_dim,
                                           vit_heads=cfg.vit_heads, tiny_cnn_width=cfg.tiny_cnn_width) if self.use_image else None
        self.text = ByteTextEncoder(cfg.dim, cfg.text_depth, cfg.text_heads, max_text_len, cfg.text_downsample,
                                    cfg.dropout, stem=cfg.text_stem) if self.use_text else None
        self.fusion = build_fusion(cfg.fusion, cfg.dim, len(self.task_names), self.use_image, self.use_text,
                                   cfg.fusion_depth, cfg.fusion_heads, cfg.dropout)
        self.heads = nn.ModuleDict({t: TaskHead(cfg.dim, len(c), cfg.dropout) for t, c in tasks.items()})

    def _modality_dropout(self, b: int, device, has_text: Optional[torch.Tensor]):
        """Randomly hide one modality per sample (never both) to build robustness."""
        p = self.cfg.modality_dropout
        v_present = torch.ones(b, dtype=torch.bool, device=device)
        t_present = has_text if has_text is not None else torch.ones(b, dtype=torch.bool, device=device)
        if self.training and p > 0 and self.use_image and self.use_text:
            r = torch.rand(b, device=device)
            drop_text = r < p / 2
            drop_img = (r >= p / 2) & (r < p) & t_present  # only drop the image if text is there
            t_present = t_present & ~drop_text
            v_present = v_present & ~drop_img
        return v_present, t_present

    def forward(self, image: Optional[torch.Tensor] = None, text_ids: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        v_tokens = t_tokens = t_valid = has_text = emb = None
        grid = None
        if self.use_image:
            v_tokens, grid = self.vision(image)
        if self.use_text:
            if return_features:
                t_tokens, t_valid, has_text, emb = self.text(text_ids, return_embeddings=True)
            else:
                t_tokens, t_valid, has_text = self.text(text_ids)
        b = (image if image is not None else text_ids).shape[0]
        device = (image if image is not None else text_ids).device
        v_present, t_present = self._modality_dropout(b, device, has_text)
        feats = self.fusion(v_tokens, t_tokens, t_valid, v_present, t_present)
        out = {t: self.heads[t](feats[:, i]) for i, t in enumerate(self.task_names)}
        if return_features:
            out["_vision_tokens"] = v_tokens
            out["_grid"] = grid
            out["_text_embeddings"] = emb
        return out


class ExportWrapper(nn.Module):
    """Tensor-in / tuple-out wrapper used for ONNX and TorchScript export."""

    def __init__(self, model: ScreenThreatNet):
        super().__init__()
        self.model = model
        self.task_names = model.task_names

    def forward(self, image: torch.Tensor, text_ids: torch.Tensor):
        out = self.model(image if self.model.use_image else None, text_ids if self.model.use_text else None)
        return tuple(out[t] for t in self.task_names)


def build_model(cfg: ModelConfig, tasks: Dict[str, List[str]], img_size=(144, 256), max_text_len: int = 512) -> ScreenThreatNet:
    return ScreenThreatNet(cfg, tasks, tuple(img_size), max_text_len)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
