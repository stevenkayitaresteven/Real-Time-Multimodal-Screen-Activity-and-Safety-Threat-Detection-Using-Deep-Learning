"""Training, prediction, evaluation and checkpoint I/O."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .calibration import fit_temperature
from .config import Config
from .data.datasets import ImageFolderComposite, ManifestDataset
from .metrics import binary_threat_report, classification_report, softmax
from .models.losses import MultiTaskLoss
from .models.multitask import ScreenThreatNet, build_model, count_parameters
from .taxonomy import IGNORE_INDEX, TASKS
from .text.tokenizer import ByteTokenizer

CHECKPOINT_VERSION = 1


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_dataset(cfg: Config, split: str, return_text: bool = False, manifests: Optional[List[str]] = None,
                  ocr_noise_p: Optional[float] = None, drop_text: bool = False) -> Dataset:
    d = cfg.data
    tok = ByteTokenizer(d.max_text_len)
    train = split == "train"
    manifests = manifests if manifests is not None else {"train": d.train_manifests, "val": d.val_manifests,
                                                          "test": d.test_manifests}[split]
    parts: List[Dataset] = []
    if manifests:
        parts.append(ManifestDataset(manifests, tuple(d.img_size), tok, d.text_field,
                                     ocr_noise_p=d.ocr_noise_p if (train and ocr_noise_p is None) else (ocr_noise_p or 0.0),
                                     augment=train and d.augment, preload=d.preload, return_text=return_text,
                                     drop_text=drop_text))
    if d.nsfw_root and split in ("train", "val"):
        root = Path(d.nsfw_root) / split
        if root.is_dir():
            parts.append(ImageFolderComposite(str(root), tuple(d.img_size), tok, composite_p=d.nsfw_composite_p,
                                              seed=cfg.train.seed + (0 if train else 1), augment=train and d.augment,
                                              return_text=return_text))
    if not parts:
        raise ValueError(f"No data configured for split {split!r}")
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def tasks_with_labels(ds: Dataset) -> List[str]:
    found = set()
    sets = ds.datasets if isinstance(ds, ConcatDataset) else [ds]
    for s in sets:
        if isinstance(s, ManifestDataset):
            for t, c in s.label_counts().items():
                if c:
                    found.add(t)
        elif isinstance(s, ImageFolderComposite):
            found.add(s.task)
            if s.composite_p > 0:
                found.update({"activity", "threat"})
        else:
            found.update({"activity", "threat"})
    return [t for t in TASKS if t in found]


# --------------------------------------------------------------------------- #
# Prediction / evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model: ScreenThreatNet, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    logits: Dict[str, List[np.ndarray]] = {t: [] for t in model.task_names}
    labels: Dict[str, List[np.ndarray]] = {t: [] for t in model.task_names}
    texts: List[str] = []
    for batch in loader:
        out = model(batch["image"].to(device), batch["text_ids"].to(device))
        for t in model.task_names:
            logits[t].append(out[t].float().cpu().numpy())
            labels[t].append(batch[t].numpy())
        if "text" in batch:
            texts.extend(batch["text"])
    return {"logits": {t: np.concatenate(v) for t, v in logits.items()},
            "labels": {t: np.concatenate(v) for t, v in labels.items()}, "texts": texts}


def evaluate_predictions(pred: Dict, temperatures: Optional[Dict[str, float]] = None,
                         tasks: Optional[List[str]] = None) -> Dict:
    report: Dict = {}
    for t in tasks or list(pred["logits"].keys()):
        y = pred["labels"][t]
        m = y != IGNORE_INDEX
        if not m.any():
            continue
        z = pred["logits"][t][m] / (temperatures or {}).get(t, 1.0)
        probs = softmax(z)
        report[t] = classification_report(y[m], probs, TASKS[t])
        if t == "threat":
            report[t]["binary"] = binary_threat_report(y[m], probs)
    scores = [r["macro_f1"] for r in report.values()]
    report["score"] = round(float(np.mean(scores)), 4) if scores else 0.0
    return report


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model: ScreenThreatNet, cfg: Config, meta: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": CHECKPOINT_VERSION, "state_dict": model.state_dict(), "config": cfg.to_dict(),
                "tasks": {t: TASKS[t] for t in model.task_names}, **meta}, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Tuple[ScreenThreatNet, Dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = Config.from_dict(ck["config"])
    cfg.model.pretrained = False  # weights come from the checkpoint
    model = build_model(cfg.model, ck["tasks"], tuple(cfg.data.img_size), cfg.data.max_text_len)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ck["config_obj"] = cfg
    return model, ck


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def _loader(ds: Dataset, cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle, num_workers=cfg.train.num_workers,
                      drop_last=shuffle and len(ds) > cfg.train.batch_size, persistent_workers=cfg.train.num_workers > 0)


def train(cfg: Config, log=print) -> Dict:
    tc = cfg.train
    set_seed(tc.seed)
    if tc.threads:
        torch.set_num_threads(tc.threads)
    device = resolve_device(tc.device)
    out = Path(tc.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(str(out / "config.json"))

    t0 = time.time()
    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    trained_tasks = tasks_with_labels(train_ds)
    log(f"[data] train={len(train_ds)} val={len(val_ds)} tasks_with_labels={trained_tasks} ({time.time() - t0:.1f}s)")
    train_dl, val_dl = _loader(train_ds, cfg, True), _loader(val_ds, cfg, False)

    model = build_model(cfg.model, TASKS, tuple(cfg.data.img_size), cfg.data.max_text_len).to(device)
    if tc.init_checkpoint:
        init = torch.load(tc.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(init["state_dict"])
        log(f"[init] warm-started from {tc.init_checkpoint}")
    criterion = MultiTaskLoss(TASKS, tc.task_weighting, tc.task_weights, tc.label_smoothing).to(device)
    n_params = count_parameters(model)
    log(f"[model] vision={cfg.model.vision} fusion={cfg.model.fusion} modalities={cfg.model.modalities} "
        f"params={n_params / 1e6:.2f}M device={device}")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n.endswith(("pos", "task_tokens", "type_embed")) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0},
                             {"params": criterion.parameters(), "weight_decay": 0.0}], lr=tc.lr)
    total_steps = max(1, tc.epochs * len(train_dl))
    warmup = int(tc.warmup_frac * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = tc.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, best_epoch, bad_epochs = -1.0, -1, 0
    history = []
    step = 0
    for epoch in range(1, tc.epochs + 1):
        model.train()
        te = time.time()
        running: Dict[str, float] = {}
        n_batches = 0
        for batch in train_dl:
            img = batch["image"].to(device, non_blocking=True)
            ids = batch["text_ids"].to(device, non_blocking=True)
            targets = {t: batch[t].to(device) for t in TASKS}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(img, ids)
            loss, parts = criterion(logits, targets)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            n_batches += 1
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            if tc.log_every and step % tc.log_every == 0:
                log(f"  step {step}/{total_steps} loss={float(loss.detach()):.4f} " +
                    " ".join(f"{k}={v / n_batches:.3f}" for k, v in running.items()))
        pred = predict(model, val_dl, device)
        rep = evaluate_predictions(pred, tasks=trained_tasks)
        train_losses = {k: round(v / max(1, n_batches), 4) for k, v in running.items()}
        rec = {"epoch": epoch, "time_s": round(time.time() - te, 1), "train_loss": train_losses, "val_score": rep["score"],
               **{f"val_{t}_acc": rep[t]["accuracy"] for t in trained_tasks if t in rep},
               **{f"val_{t}_f1": rep[t]["macro_f1"] for t in trained_tasks if t in rep},
               "task_sigma": {t: round(float(criterion.log_vars[t].detach().exp().sqrt()), 3) for t in TASKS}}
        history.append(rec)
        with open(out / "train_log.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(f"[epoch {epoch}] {rec['time_s']}s loss={train_losses} val_score={rep['score']:.4f} " +
            " ".join(f"{t}:acc={rep[t]['accuracy']:.3f}/f1={rep[t]['macro_f1']:.3f}" for t in trained_tasks if t in rep))
        if rep["score"] > best_score:
            best_score, best_epoch, bad_epochs = rep["score"], epoch, 0
            torch.save(model.state_dict(), out / "_best_weights.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= tc.early_stop_patience:
                log(f"[early stop] no improvement for {bad_epochs} epochs")
                break

    model.load_state_dict(torch.load(out / "_best_weights.pt", map_location=device))
    (out / "_best_weights.pt").unlink(missing_ok=True)
    pred = predict(model, val_dl, device)
    temps = {t: fit_temperature(pred["logits"][t][pred["labels"][t] != IGNORE_INDEX],
                                pred["labels"][t][pred["labels"][t] != IGNORE_INDEX]) for t in trained_tasks}
    val_rep = evaluate_predictions(pred, temps, trained_tasks)
    meta = {"tasks_trained": trained_tasks, "temperatures": temps, "best_epoch": best_epoch,
            "val_metrics": val_rep, "n_params": n_params, "history": history}
    save_checkpoint(str(out / "best.pt"), model.cpu(), cfg, meta)
    summary = {"best_epoch": best_epoch, "val_score": val_rep["score"], "temperatures": temps, "n_params": n_params,
               "train_time_s": round(time.time() - t0, 1), "val": val_rep}
    (out / "val_metrics.json").write_text(json.dumps(summary, indent=2))
    log(f"[done] best_epoch={best_epoch} val_score={val_rep['score']:.4f} temps={ {k: round(v, 3) for k, v in temps.items()} } "
        f"-> {out / 'best.pt'}")
    return summary
