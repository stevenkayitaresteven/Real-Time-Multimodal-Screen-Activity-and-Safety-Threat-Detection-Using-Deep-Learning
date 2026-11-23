"""Generate report figures into docs/figures/ from results in docs/results/.

Each figure is skipped when its inputs are missing, so the script can be run
after any subset of experiments.  Requires matplotlib (dev-only dependency).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RES = ROOT / "docs/results"
FIG = ROOT / "docs/figures"
FIG.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (fixed order) + text/surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#9cc3ef", "#2a78d6", "#0f3f7a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def confusion(rep, task, title):
    r = rep["model"].get(task)
    if not r:
        return
    cm = np.array(r["confusion"], dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    labels = [c.replace("_", "\n") for c in r["classes"]]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.9 * n + 2, 0.75 * n + 1.5))
    ax.grid(False)
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{norm[i, j]:.0%}" if norm[i, j] >= 0.005 else "<1%", ha="center", va="center",
                    fontsize=8, color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(n), labels, fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}  (acc {r['accuracy']:.3f}, macro-F1 {r['macro_f1']:.3f})", loc="left")
    save(fig, f"confusion_{task}.png")


def ablation_bars(abl):
    names = list(abl.keys())
    metrics = [("threat_f1", "Threat macro-F1 (clean text)"), ("activity_acc", "Activity accuracy (clean text)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(names) + 1.4), sharey=True)
    for ax, (key, title) in zip(axes, metrics):
        vals = [abl[n]["conditions"]["clean"].get(key) or 0 for n in names]
        y = np.arange(len(names))
        ax.barh(y, vals, height=0.62, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        lo = max(0.0, min(vals) - 0.1)
        ax.set_xlim(lo, 1.02)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
    save(fig, "ablations.png")


def robustness(abl):
    keep = [n for n in ("attention (main config)", "no_modality_dropout", "gated_fusion", "concat_fusion", "text_only", "image_only") if n in abl]
    conds = [("clean", "clean text"), ("real_ocr", "real Tesseract OCR"), ("no_text", "no text")]
    for key, title, fname in [("threat_f1", "Threat macro-F1 under text degradation", "robustness_threat.png"),
                              ("activity_acc", "Activity accuracy under text degradation", "robustness_activity.png")]:
        fig, ax = plt.subplots(figsize=(10, 3.6))
        x = np.arange(len(keep))
        w = 0.26
        for k, (c, label) in enumerate(conds):
            vals = [abl[n]["conditions"][c].get(key) or 0 for n in keep]
            ax.bar(x + (k - 1) * w, vals, width=w, color=SERIES[k], edgecolor=SURFACE, linewidth=2, label=label)
        ax.set_xticks(x, [n.replace("_", " ") for n in keep], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.3)
        ax.set_title(title, loc="left", pad=30)
        save(fig, fname)


def latency(bench):
    rows = [r for r in bench["models"] if r["threads"] == 1]
    if not rows:
        return
    rts = [("torch", "PyTorch fp32"), ("onnx", "ONNX Runtime fp32"), ("onnx-int8", "ONNX Runtime int8")]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.6))
    y = np.arange(len(rows))
    h = 0.26
    for k, (rt, label) in enumerate(rts):
        vals = [r.get(rt, {}).get("mean_ms", 0) for r in rows]
        ax.barh(y + (k - 1) * h, vals, height=h, color=SERIES[k], edgecolor=SURFACE, linewidth=2, label=label)
        for yi, v in zip(y + (k - 1) * h, vals):
            ax.text(v + 0.2, yi, f"{v:.1f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [f"{r['vision']} / {r['fusion']}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("mean latency per frame, ms (1 CPU thread, batch 1)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.3)
    ax.set_title("Model inference latency by backbone and runtime", loc="left", pad=30)
    save(fig, "latency.png")


def samples():
    man = ROOT / "data/synth/test_hires/manifest.jsonl"
    if not man.exists():
        return
    from PIL import Image

    recs = [json.loads(l) for l in man.read_text().splitlines()]
    want = [("web_browsing", "phishing"), ("web_browsing", "scam"), ("email", "phishing"), ("terminal", "suspicious_command"),
            ("file_download", "malware_download"), ("messaging", "scam"), ("code_editor", "benign"), ("video_streaming", "benign"),
            ("social_media", "benign"), ("document_editing", "benign"), ("web_browsing", "suspicious_command"), ("email", "benign")]
    picks = [next((r for r in recs if (r["activity"], r["threat"]) == w), None) for w in want]
    picks = [p for p in picks if p]
    fig, axes = plt.subplots(3, 4, figsize=(16, 7.4))
    for ax, r in zip(axes.flat, picks):
        ax.imshow(Image.open(man.parent / r["image"]))
        ax.set_title(f"{r['activity']}  |  {r['threat']}", fontsize=9, loc="left", fontweight="normal", color=INK)
        ax.axis("off")
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")
    fig.suptitle("Synthetic screens (activity | threat) rendered by screenguard.data.synthetic", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, "synthetic_samples.png")


def gradcam():
    ckpt = ROOT / "runs/main/best.pt"
    man = ROOT / "data/synth/test_hires/manifest.jsonl"
    if not (ckpt.exists() and man.exists()):
        return
    from PIL import Image

    from screenguard.explain.gradcam import explain
    from screenguard.metrics import softmax
    from screenguard.taxonomy import TASKS
    from screenguard.training import load_checkpoint

    model, ck = load_checkpoint(str(ckpt))
    cfg = ck["config_obj"]
    recs = [json.loads(l) for l in man.read_text().splitlines()]
    cases = [("web_browsing", "phishing", "threat"), ("web_browsing", "scam", "threat"), ("code_editor", "benign", "activity")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [0.6, 1, 1]})
    for ax, (act, thr, task) in zip(axes, cases):
        r = next(r for r in recs if (r["activity"], r["threat"]) == (act, thr))
        img = Image.open(man.parent / r["image"]).convert("RGB")
        out = explain(model, img, r["text"], tuple(cfg.data.img_size), cfg.data.max_text_len, task=task)
        import torch

        with torch.no_grad():
            from screenguard.data.preprocess import preprocess
            from screenguard.text.tokenizer import ByteTokenizer

            z = model(torch.from_numpy(preprocess(img, tuple(cfg.data.img_size)))[None],
                      torch.from_numpy(ByteTokenizer(cfg.data.max_text_len).encode(r["text"]))[None])[task][0].numpy()
        p = softmax(z / ck.get("temperatures", {}).get(task, 1.0))
        label = TASKS[task][out["class_index"]]
        ax.imshow(out["overlay"])
        ax.axis("off")
        words = ", ".join((w if len(w) <= 28 else w[:26] + "…") for w, _ in out.get("top_words", [])[:4])
        import textwrap

        ax.set_title(f"{task}: {label} (p={p[out['class_index']]:.2f})\n" + "\n".join(textwrap.wrap(f"top words: {words}", 60)),
                     fontsize=9, loc="left", fontweight="normal")
    fig.suptitle("Grad-CAM (image) and gradient-x-input word saliency (text)", x=0.01, y=1.06, ha="left", fontsize=12,
                 fontweight="bold")
    save(fig, "gradcam_examples.png")


if __name__ == "__main__":
    rep = load("test_main.json")
    if rep:
        confusion(rep, "activity", "Activity recognition")
        confusion(rep, "threat", "Threat detection (model only)")
    abl = load("ablations.json")
    if abl:
        ablation_bars(abl)
        robustness(abl)
    bench = load("benchmark.json")
    if bench:
        latency(bench)
    samples()
    gradcam()
