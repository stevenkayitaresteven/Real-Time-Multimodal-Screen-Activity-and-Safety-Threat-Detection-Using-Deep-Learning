"""Train and evaluate the ablation grid, then write results to docs/results/.

Usage:
    python scripts/run_ablations.py --epochs 5            # all variants
    python scripts/run_ablations.py --only text_only gated
    python scripts/run_ablations.py --eval-only           # re-evaluate existing runs

Each variant changes one factor of the main configuration so its effect can be
isolated.  All variants are evaluated on the same held-out test split under
three text conditions: clean (ground-truth) text, simulated OCR noise (10 %
character error rate) and no text at all (OCR failure / pure-visual content).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screenguard.config import Config  # noqa: E402
from screenguard.evaluate import evaluate_checkpoint, rows_to_markdown, summary_row  # noqa: E402
from screenguard.training import train  # noqa: E402

VARIANTS = {
    "attention (main)": [],
    "image_only": ["model.modalities=[image]"],
    "text_only": ["model.modalities=[text]"],
    "concat_fusion": ["model.fusion=concat"],
    "gated_fusion": ["model.fusion=gated"],
    "vit_tiny": ["model.vision=vit_tiny"],
    "mobilenet_v3_small": ["model.vision=mobilenet_v3_small"],
    "no_modality_dropout": ["model.modality_dropout=0.0"],
    "fixed_task_weights": ["train.task_weighting=fixed"],
    "strided_text_stem": ["model.text_stem=strided"],
}
# All ablations share one training budget (8k screens) so differences come from the factor being varied.
BUDGET = ["data.train_manifests=[data/synth/train/manifest.jsonl]"]
CONDITIONS = {"clean": {}, "ocr_noise_10%": {"ocr_noise_p": 0.10}, "no_text": {"drop_text": True}}


def slug(name: str) -> str:
    return name.split(" ")[0].replace("/", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs/results"))
    ap.add_argument("--runs", default=str(ROOT / "runs/ablations"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "ablations.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name, overrides in VARIANTS.items():
        if a.only and slug(name) not in a.only:
            continue
        run_dir = Path(a.runs) / slug(name)
        ckpt = run_dir / "best.pt"
        if not a.eval_only:
            cfg = Config.load(a.config, BUDGET + overrides + [f"train.epochs={a.epochs}", f"train.out_dir={run_dir}"])
            t = time.time()
            summary = train(cfg)
            print(f"== {name}: trained in {time.time() - t:.0f}s, val_score={summary['val_score']}", flush=True)
        if not ckpt.exists():
            print(f"skip {name}: no checkpoint", flush=True)
            continue
        cfg = Config.load(a.config)
        entry = {"overrides": overrides, "conditions": {}}
        for cond, kw in CONDITIONS.items():
            rep = evaluate_checkpoint(str(ckpt), cfg.data.test_manifests, **kw)
            entry["conditions"][cond] = summary_row(name, rep)
            entry.setdefault("full", {})[cond] = rep
        train_meta = json.loads((run_dir / "val_metrics.json").read_text())
        entry["n_params"] = train_meta["n_params"]
        entry["train_time_s"] = train_meta["train_time_s"]
        entry["best_epoch"] = train_meta["best_epoch"]
        results[name] = entry
        results_path.write_text(json.dumps(results, indent=2))
        print(f"== {name}: {entry['conditions']['clean']}", flush=True)

    rows = []
    for name, e in results.items():
        c = e["conditions"]
        rows.append({
            "variant": name, "params_M": round(e["n_params"] / 1e6, 2),
            "activity_acc": c["clean"].get("activity_acc"), "activity_f1": c["clean"].get("activity_f1"),
            "threat_f1": c["clean"].get("threat_f1"), "threat_auroc": c["clean"].get("threat_auroc"),
            "threat_f1 (OCR noise)": c["ocr_noise_10%"].get("threat_f1"),
            "activity_acc (no text)": c["no_text"].get("activity_acc"), "threat_f1 (no text)": c["no_text"].get("threat_f1"),
            "hybrid_threat_f1": c["clean"].get("hybrid_threat_f1"),
        })
    md = rows_to_markdown(rows)
    (out / "ablations.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
