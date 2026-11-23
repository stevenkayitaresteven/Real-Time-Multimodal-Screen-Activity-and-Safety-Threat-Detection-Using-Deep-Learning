"""Vision encoders.

Every encoder maps an image batch ``[B, 3, H, W]`` to a sequence of visual
tokens ``[B, N, D]`` plus the spatial grid ``(h, w)`` with ``N = h * w``.
Emitting tokens (not just a pooled vector) lets the attention-fusion module
look at *where* things are on screen, and lets Grad-CAM produce heat-maps for
any backbone, CNN or ViT.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    out_dim: int

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:  # pragma: no cover - interface
        raise NotImplementedError


def _make_divisible(v: float, d: int = 8) -> int:
    return max(d, int(v + d / 2) // d * d)


class SqueezeExcite(nn.Module):
    def __init__(self, c: int, r: int = 4):
        super().__init__()
        h = _make_divisible(c / r)
        self.fc1, self.fc2 = nn.Conv2d(c, h, 1), nn.Conv2d(h, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        return x * F.hardsigmoid(self.fc2(F.relu(self.fc1(s))))


class InvertedResidual(nn.Module):
    """MobileNetV2/V3 block: expand (1x1) -> depthwise (kxk) -> SE -> project (1x1)."""

    def __init__(self, cin: int, cout: int, stride: int, expand: int, k: int = 3, se: bool = False):
        super().__init__()
        hid = cin * expand
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.Hardswish()]
        layers += [nn.Conv2d(hid, hid, k, stride, k // 2, groups=hid, bias=False), nn.BatchNorm2d(hid), nn.Hardswish()]
        if se:
            layers.append(SqueezeExcite(hid))
        layers += [nn.Conv2d(hid, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.residual else y


class TinyCNN(VisionEncoder):
    """A ~0.4M-parameter MobileNetV3-style CNN designed for on-device inference
    and training from scratch (no pretrained weights required)."""

    # (out_channels, stride, expand, kernel, squeeze-excite)
    # Downsamples early (like MobileNetV3-Small) so most compute happens at 1/8-1/32 resolution.
    CFG = [(16, 2, 1, 3, True), (24, 2, 4, 3, False), (24, 1, 3, 3, False), (40, 2, 4, 5, True),
           (40, 1, 4, 5, True), (64, 1, 4, 5, True), (96, 2, 4, 5, True), (96, 1, 4, 5, True)]

    def __init__(self, dim: int = 128, width: float = 1.0):
        super().__init__()
        c = _make_divisible(16 * width)
        layers = [nn.Conv2d(3, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.Hardswish()]
        for out, s, e, k, se in self.CFG:
            o = _make_divisible(out * width)
            layers.append(InvertedResidual(c, o, s, e, k, se))
            c = o
        self.features = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Conv2d(c, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.Hardswish())
        self.out_dim = dim

    def forward(self, x):
        f = self.proj(self.features(x))
        b, d, h, w = f.shape
        return f.flatten(2).transpose(1, 2), (h, w)


class ViTTiny(VisionEncoder):
    """Compact Vision Transformer (patch embedding + pre-norm encoder).

    Learned position embeddings are bilinearly interpolated when the input
    resolution changes, so one model can serve phones and laptops.
    """

    def __init__(self, dim: int = 128, img_size=(144, 256), patch: int = 16, width: int = 192, depth: int = 4,
                 heads: int = 3, dropout: float = 0.0):
        super().__init__()
        self.patch = patch
        self.grid = (img_size[0] // patch, img_size[1] // patch)
        self.embed = nn.Conv2d(3, width, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, width, *self.grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, dim)
        self.out_dim = dim

    def forward(self, x):
        f = self.embed(x)
        h, w = f.shape[-2:]
        pos = self.pos if (h, w) == self.grid else F.interpolate(self.pos, size=(h, w), mode="bilinear", align_corners=False)
        t = (f + pos).flatten(2).transpose(1, 2)
        t = self.proj(self.norm(self.encoder(t)))
        return t, (h, w)


class TorchvisionEncoder(VisionEncoder):
    """Wraps a torchvision classification backbone (optionally ImageNet-pretrained)."""

    SUPPORTED = ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18", "convnext_tiny")

    def __init__(self, name: str, dim: int = 128, pretrained: bool = False):
        super().__init__()
        import torchvision.models as tvm

        if name not in self.SUPPORTED:
            raise ValueError(f"Unsupported torchvision backbone {name!r}; choose from {self.SUPPORTED}")
        weights = "DEFAULT" if pretrained else None
        net = getattr(tvm, name)(weights=weights)
        if name.startswith("resnet"):
            self.features = nn.Sequential(*list(net.children())[:-2])
        else:
            self.features = net.features
        with torch.no_grad():
            c = self.features(torch.zeros(1, 3, 64, 64)).shape[1]
        self.proj = nn.Sequential(nn.Conv2d(c, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.Hardswish())
        self.out_dim = dim

    def forward(self, x):
        f = self.proj(self.features(x))
        b, d, h, w = f.shape
        return f.flatten(2).transpose(1, 2), (h, w)


class TimmEncoder(VisionEncoder):
    """Any `timm` model (e.g. ``timm:mobilevitv2_050``, ``timm:efficientvit_b0``)."""

    def __init__(self, name: str, dim: int = 128, pretrained: bool = False):
        super().__init__()
        import timm  # optional dependency

        self.net = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
        with torch.no_grad():
            f = self.net.forward_features(torch.zeros(1, 3, 128, 128))
        c = f.shape[1] if f.ndim == 4 else f.shape[-1]
        self.is_tokens = f.ndim == 3
        self.proj = nn.Linear(c, dim)
        self.out_dim = dim

    def forward(self, x):
        f = self.net.forward_features(x)
        if f.ndim == 4:
            h, w = f.shape[-2:]
            t = f.flatten(2).transpose(1, 2)
        else:
            t = f
            n = t.shape[1]
            h = int(math.sqrt(n))
            w = n // max(1, h)
            if h * w != n:
                h, w = 1, n
        return self.proj(t), (h, w)


def build_vision_encoder(name: str, dim: int, pretrained: bool = False, img_size=(144, 256), **kw) -> VisionEncoder:
    if name == "tiny_cnn":
        return TinyCNN(dim, width=kw.get("tiny_cnn_width", 1.0))
    if name == "vit_tiny":
        return ViTTiny(dim, img_size=img_size, patch=kw.get("vit_patch", 16), width=kw.get("vit_dim", 192),
                       depth=kw.get("vit_depth", 4), heads=kw.get("vit_heads", 3))
    if name.startswith("timm:"):
        return TimmEncoder(name.split(":", 1)[1], dim, pretrained)
    return TorchvisionEncoder(name, dim, pretrained)
