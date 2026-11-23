"""Byte-level Transformer text encoder for OCR output.

Bytes are embedded, character n-gram detectors (a stride-1 convolution) fire
on cues such as ``xn--``, ``.exe`` or ``| bash``, and max-pooling downsamples
the sequence so the Transformer runs on ``L / downsample`` tokens.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..text.tokenizer import BYTE_OFFSET, PAD_ID, VOCAB_SIZE


class ByteTextEncoder(nn.Module):
    """Byte embedding -> n-gram convolution -> max-pool downsampling -> Transformer.

    ``stem="ngram"`` (default) uses a stride-1 convolution (shift-equivariant
    n-gram detectors) followed by max-pooling (local shift invariance), as in
    character-CNN text classifiers.  A keyword such as ``verify`` then fires
    the same detector wherever it occurs.  ``stem="strided"`` is the cheaper
    patch-style alternative (a single strided convolution), kept for ablations.
    """

    def __init__(self, dim: int = 128, depth: int = 2, heads: int = 4, max_len: int = 1024, downsample: int = 8,
                 dropout: float = 0.1, stem: str = "ngram", emb_dim: int = 64, ngram: int = 7):
        super().__init__()
        self.downsample = max(1, downsample)
        self.stem = stem
        if stem == "ngram":
            self.embed = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=PAD_ID)
            self.conv = nn.Conv1d(emb_dim, dim, kernel_size=ngram, padding=ngram // 2)
            self.k, self.pad = self.downsample, 0
        elif stem == "strided":
            self.embed = nn.Embedding(VOCAB_SIZE, dim, padding_idx=PAD_ID)
            k = self.downsample * 2 if self.downsample > 1 else 3
            pad = (k - self.downsample) // 2 if self.downsample > 1 else 1
            self.conv = nn.Conv1d(dim, dim, kernel_size=k, stride=self.downsample, padding=pad)
            self.k, self.pad = k, pad
        else:
            raise ValueError(f"Unknown text stem {stem!r}")
        self.conv_norm = nn.LayerNorm(dim)
        n_pos = max_len // self.downsample + 2
        self.pos = nn.Parameter(torch.zeros(1, n_pos, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample == 1:
            return x
        return F.max_pool1d(x, kernel_size=self.k, stride=self.downsample, padding=self.pad, ceil_mode=self.stem == "ngram")

    def forward(self, ids: torch.Tensor, return_embeddings: bool = False):
        """Returns ``(tokens [B, N, D], valid_mask [B, N], has_text [B])``."""
        emb = self.embed(ids)
        x = self.conv(emb.transpose(1, 2))
        if self.stem == "ngram":
            x = self._pool(F.gelu(x))
        x = F.gelu(self.conv_norm(x.transpose(1, 2)))
        n = x.shape[1]
        if n > self.pos.shape[1]:
            raise ValueError(f"Text too long for positional table ({n} > {self.pos.shape[1]})")
        x = x + self.pos[:, :n]
        valid = self._pool((ids != PAD_ID).float().unsqueeze(1)).squeeze(1)[:, :n] > 0.5
        valid[:, 0] = True  # first token always covers CLS -> never an all-masked row
        x = self.encoder(x, src_key_padding_mask=~valid)
        x = self.norm(x)
        has_text = (ids >= BYTE_OFFSET).any(dim=1)
        if return_embeddings:
            return x, valid, has_text, emb
        return x, valid, has_text
