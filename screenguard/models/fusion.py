"""Multimodal fusion strategies (compared in the ablation study).

All fusion modules share one signature and return one feature vector per task
(``[B, T, D]``) so the multi-task heads are agnostic to the fusion choice.

* ``concat``    – late fusion: concatenate pooled modality vectors, then an MLP.
* ``gated``     – Gated Multimodal Unit (Arevalo et al., 2017): a learned,
                  per-dimension gate decides how much to trust each modality.
* ``attention`` – single-stream Transformer over [task tokens | visual tokens |
                  text tokens] with modality-type embeddings.  Each task owns
                  a query token, so e.g. the threat head can attend to URL bytes
                  while the activity head attends to layout tokens.

Missing modalities (no OCR text on a video frame, modality dropout during
training, or a single-modality ablation) are handled explicitly: pooled
fusions substitute a learned "missing" embedding, attention fusion masks the
tokens out.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def masked_mean(tokens: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
    if valid is None:
        return tokens.mean(1)
    w = valid.unsqueeze(-1).to(tokens.dtype)
    return (tokens * w).sum(1) / w.sum(1).clamp_min(1.0)


class _PooledFusion(nn.Module):
    def __init__(self, dim: int, num_tasks: int, use_image: bool, use_text: bool):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_image, self.use_text = use_image, use_text
        self.null_image = nn.Parameter(torch.zeros(dim))
        self.null_text = nn.Parameter(torch.zeros(dim))

    def pooled(self, v_tokens, t_tokens, t_valid, v_present, t_present):
        v = t = None
        if self.use_image:
            v = masked_mean(v_tokens, None)
            if v_present is not None:
                v = torch.where(v_present.unsqueeze(-1), v, self.null_image.expand_as(v))
        if self.use_text:
            t = masked_mean(t_tokens, t_valid)
            if t_present is not None:
                t = torch.where(t_present.unsqueeze(-1), t, self.null_text.expand_as(t))
        return v, t


class ConcatFusion(_PooledFusion):
    def __init__(self, dim: int, num_tasks: int, use_image: bool = True, use_text: bool = True, dropout: float = 0.1):
        super().__init__(dim, num_tasks, use_image, use_text)
        n_in = dim * (int(use_image) + int(use_text))
        self.mlp = nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, v_tokens, t_tokens, t_valid, v_present=None, t_present=None):
        v, t = self.pooled(v_tokens, t_tokens, t_valid, v_present, t_present)
        h = self.mlp(torch.cat([x for x in (v, t) if x is not None], dim=-1))
        return h.unsqueeze(1).expand(-1, self.num_tasks, -1)


class GatedFusion(_PooledFusion):
    def __init__(self, dim: int, num_tasks: int, use_image: bool = True, use_text: bool = True, dropout: float = 0.1):
        super().__init__(dim, num_tasks, use_image, use_text)
        self.hv = nn.Linear(dim, dim)
        self.ht = nn.Linear(dim, dim)
        self.gate = nn.Linear(2 * dim, dim)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout))
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, v_tokens, t_tokens, t_valid, v_present=None, t_present=None):
        v, t = self.pooled(v_tokens, t_tokens, t_valid, v_present, t_present)
        if v is None or t is None:  # single-modality ablation degenerates to a projection
            x = v if v is not None else t
            h = torch.tanh(self.hv(x) if v is not None else self.ht(x))
        else:
            z = torch.sigmoid(self.gate(torch.cat([v, t], dim=-1)))
            self.last_gate = z.detach()
            h = z * torch.tanh(self.hv(v)) + (1 - z) * torch.tanh(self.ht(t))
        return self.out(h).unsqueeze(1).expand(-1, self.num_tasks, -1)


class AttentionFusion(nn.Module):
    def __init__(self, dim: int, num_tasks: int, depth: int = 2, heads: int = 4, dropout: float = 0.1,
                 use_image: bool = True, use_text: bool = True, max_vision_tokens: int = 1024):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_image, self.use_text = use_image, use_text
        self.task_tokens = nn.Parameter(torch.zeros(1, num_tasks, dim))
        self.type_embed = nn.Parameter(torch.zeros(3, dim))  # 0 task, 1 image, 2 text
        nn.init.trunc_normal_(self.task_tokens, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, v_tokens, t_tokens, t_valid, v_present=None, t_present=None):
        b = (v_tokens if v_tokens is not None else t_tokens).shape[0]
        device = (v_tokens if v_tokens is not None else t_tokens).device
        seq = [self.task_tokens.expand(b, -1, -1) + self.type_embed[0]]
        masks = [torch.zeros(b, self.num_tasks, dtype=torch.bool, device=device)]  # True = ignore
        if self.use_image:
            seq.append(v_tokens + self.type_embed[1])
            vm = torch.zeros(b, v_tokens.shape[1], dtype=torch.bool, device=device)
            if v_present is not None:
                vm = vm | ~v_present.unsqueeze(1)
            masks.append(vm)
        if self.use_text:
            seq.append(t_tokens + self.type_embed[2])
            tm = ~t_valid
            if t_present is not None:
                tm = tm | ~t_present.unsqueeze(1)
            masks.append(tm)
        x = self.encoder(torch.cat(seq, 1), src_key_padding_mask=torch.cat(masks, 1))
        return self.norm(x[:, : self.num_tasks])


def build_fusion(name: str, dim: int, num_tasks: int, use_image: bool, use_text: bool, depth: int = 2,
                 heads: int = 4, dropout: float = 0.1) -> nn.Module:
    if name == "concat":
        return ConcatFusion(dim, num_tasks, use_image, use_text, dropout)
    if name == "gated":
        return GatedFusion(dim, num_tasks, use_image, use_text, dropout)
    if name == "attention":
        return AttentionFusion(dim, num_tasks, depth, heads, dropout, use_image, use_text)
    raise ValueError(f"Unknown fusion {name!r}")
