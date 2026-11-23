"""Typed experiment configuration with YAML files and dotted CLI overrides."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    train_manifests: List[str] = field(default_factory=lambda: ["data/synth/train/manifest.jsonl"])
    val_manifests: List[str] = field(default_factory=lambda: ["data/synth/val/manifest.jsonl"])
    test_manifests: List[str] = field(default_factory=lambda: ["data/synth/test/manifest.jsonl"])
    # Optional ImageFolder root for the NSFW head: <root>/{train,val}/{safe,suggestive,explicit}/*.jpg
    nsfw_root: Optional[str] = None
    nsfw_composite_p: float = 0.7      # probability of pasting an NSFW-dataset image into a synthetic screen
    img_size: List[int] = field(default_factory=lambda: [144, 256])  # (H, W)
    max_text_len: int = 1024
    text_field: str = "text"           # manifest field with screen text ("text" = ground truth, "ocr_text" = OCR)
    ocr_noise_p: float = 0.03          # simulated OCR character error rate during training
    preload: bool = True               # decode + letterbox all images once into RAM
    augment: bool = True


@dataclass
class ModelConfig:
    vision: str = "tiny_cnn"           # tiny_cnn | vit_tiny | mobilenet_v3_small | mobilenet_v3_large | efficientnet_b0 | resnet18 | timm:<name>
    pretrained: bool = False
    dim: int = 128
    text_depth: int = 2
    text_heads: int = 4
    text_downsample: int = 8
    text_stem: str = "ngram"          # ngram (stride-1 conv + max-pool) | strided (single strided conv)
    fusion: str = "attention"          # concat | gated | attention
    fusion_depth: int = 2
    fusion_heads: int = 4
    modalities: List[str] = field(default_factory=lambda: ["image", "text"])
    dropout: float = 0.1
    modality_dropout: float = 0.15     # per-sample probability of dropping one modality while training
    vit_patch: int = 16
    vit_depth: int = 4
    vit_dim: int = 192
    vit_heads: int = 3
    tiny_cnn_width: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    label_smoothing: float = 0.05
    task_weighting: str = "uncertainty"  # uncertainty | fixed
    task_weights: Dict[str, float] = field(default_factory=lambda: {"activity": 1.0, "nsfw": 1.0, "threat": 1.0})
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 0
    device: str = "auto"
    threads: Optional[int] = None
    amp: bool = True
    out_dir: str = "runs/default"
    init_checkpoint: Optional[str] = None  # warm-start (fine-tuning) from an existing best.pt
    early_stop_patience: int = 4
    log_every: int = 50


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        def build(dc, values):
            names = {f.name for f in dataclasses.fields(dc)}
            unknown = set(values) - names
            if unknown:
                raise KeyError(f"Unknown config keys for {dc.__name__}: {sorted(unknown)}")
            return dc(**values)

        d = d or {}
        return cls(data=build(DataConfig, d.get("data", {})), model=build(ModelConfig, d.get("model", {})),
                   train=build(TrainConfig, d.get("train", {})))

    @classmethod
    def load(cls, path: Optional[str] = None, overrides: Optional[List[str]] = None) -> "Config":
        d: Dict[str, Any] = {}
        if path:
            d = yaml.safe_load(Path(path).read_text()) or {}
        for ov in overrides or []:
            if "=" not in ov:
                raise ValueError(f"Override must look like section.key=value, got {ov!r}")
            key, raw = ov.split("=", 1)
            parts = key.split(".")
            cur = d
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = yaml.safe_load(raw)
        return cls.from_dict(d)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
