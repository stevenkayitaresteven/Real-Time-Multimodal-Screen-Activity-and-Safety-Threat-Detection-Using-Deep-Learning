"""Checkpoint evaluation, including decision-fusion and robustness ablations."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .analysis.decision import fuse_threat
from .analysis.rules import analyze_text
from .data.datasets import ManifestDataset
from .metrics import binary_threat_report, classification_report, softmax
from .taxonomy import IGNORE_INDEX, TASKS
from .text.tokenizer import ByteTokenizer
from .training import evaluate_predictions, load_checkpoint, predict


def evaluate_checkpoint(checkpoint: str, manifests: Sequence[str], text_field: str = "text", ocr_noise_p: float = 0.0,
                        drop_text: bool = False, decision_modes: Sequence[str] = ("model", "rules", "hybrid"),
                        batch_size: int = 64, threads: Optional[int] = None, seed: int = 0) -> Dict:
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model, ck = load_checkpoint(checkpoint)
    cfg = ck["config_obj"]
    ds = ManifestDataset(list(manifests), tuple(cfg.data.img_size), ByteTokenizer(cfg.data.max_text_len), text_field,
                         ocr_noise_p=ocr_noise_p, augment=False, preload=True, return_text=True, drop_text=drop_text)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    pred = predict(model, dl, torch.device("cpu"))
    temps = ck.get("temperatures", {})
    trained = ck.get("tasks_trained", list(TASKS))
    report: Dict = {"checkpoint": checkpoint, "n": len(ds), "text_field": text_field, "ocr_noise_p": ocr_noise_p,
                    "drop_text": drop_text, "model": evaluate_predictions(pred, temps, trained)}

    if "threat" in trained and decision_modes:
        y = pred["labels"]["threat"]
        m = y != IGNORE_INDEX
        p_model = softmax(pred["logits"]["threat"] / temps.get("threat", 1.0))
        evidence = [analyze_text(t).class_scores for t in pred["texts"]]
        classes = TASKS["threat"]
        report["decision"] = {}
        for mode in decision_modes:
            q = np.stack([fuse_threat(p_model[i], evidence[i], classes, mode) for i in range(len(y))])
            r = classification_report(y[m], q[m], classes)
            r["binary"] = binary_threat_report(y[m], q[m])
            report["decision"][mode] = r
    # breakdown by device (desktop / mobile) for the activity task
    devices = [r.get("device") for r in ds.records]
    if any(devices) and "activity" in trained:
        report["by_device"] = {}
        for dev in sorted({d for d in devices if d}):
            idx = np.array([d == dev for d in devices])
            sub = {"logits": {t: v[idx] for t, v in pred["logits"].items()},
                   "labels": {t: v[idx] for t, v in pred["labels"].items()}}
            report["by_device"][dev] = {t: {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "n": r["n"]}
                                        for t, r in evaluate_predictions(sub, temps, trained).items() if t != "score"}
    return report


def summary_row(name: str, rep: Dict) -> Dict:
    row = {"run": name}
    for t in ("activity", "threat", "nsfw"):
        if t in rep["model"]:
            row[f"{t}_acc"] = rep["model"][t]["accuracy"]
            row[f"{t}_f1"] = rep["model"][t]["macro_f1"]
    if "threat" in rep["model"]:
        b = rep["model"]["threat"]["binary"]
        row["threat_auroc"] = b["auroc"]
        row["threat_tpr@1%fpr"] = b["tpr_at_1pct_fpr"]
        row["threat_ece"] = rep["model"]["threat"]["ece"]
    for mode, r in rep.get("decision", {}).items():
        row[f"{mode}_threat_f1"] = r["macro_f1"]
        row[f"{mode}_fpr"] = r["binary"]["false_positive_rate"]
        row[f"{mode}_recall"] = r["binary"]["recall"]
    return row


def rows_to_markdown(rows: List[Dict], cols: Optional[List[str]] = None) -> str:
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else (f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c)))
                                     for c in cols) + " |")
    return "\n".join(out)
