"""Architecture diagrams and extra result figures for the project report (300 dpi PNG)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

for f in Path("/usr/share/fonts/truetype/crosextra").glob("Carlito*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Carlito", "font.size": 10, "savefig.dpi": 300, "figure.facecolor": "white",
                     "axes.facecolor": "white", "mathtext.fontset": "dejavusans"})

NAVY, BLUE, LIGHT, RED = "#163A63", "#0B4F9C", "#EAF2FA", "#E4002B"
ORANGE, AQUA, GREY, INK, MUTED = "#EB6834", "#1BAF7A", "#8A8F98", "#1A1A1A", "#52514E"
VFILL, TFILL, FFILL, HFILL = "#DCEBFA", "#FDE7DC", "#E9E4FB", "#DFF5EC"


def canvas(w_mm: float, h_mm: float, scale: float = 1.0):
    fig = plt.figure(figsize=(w_mm / 25.4 * scale, h_mm / 25.4 * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc=LIGHT, ec=NAVY, lw=1.0, fs=8.5, bold_first=True, color=INK, r=1.6, ls="-", align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        lines = text.split("\n")
        n = len(lines)
        lh = fs * 0.42
        y0 = y + h / 2 + (n - 1) * lh / 2
        for i, line in enumerate(lines):
            weight = "bold" if (i == 0 and bold_first) else "normal"
            size = fs if i == 0 else fs - 1
            col = color if i == 0 else (color if color != INK else MUTED)
            xx = x + w / 2 if align == "center" else x + 1.5
            ax.text(xx, y0 - i * lh, line, ha=align, va="center", fontsize=size, fontweight=weight, color=col)


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.1, ls="-", text=None, tx=0, ty=1.6, fs=7.5, style="-|>", rad=0.0, tcolor=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha="center", va="center", fontsize=fs, color=tcolor or MUTED,
                style="italic")


def poly_arrow(ax, pts, color=NAVY, lw=1.1, ls="-"):
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, solid_capstyle="butt")
    arrow(ax, *pts[-2], *pts[-1], color=color, lw=lw, ls=ls)


def tokens(ax, x, y, n, size, color, ec="white", gap=0.35, vertical=False):
    for i in range(n):
        xx, yy = (x, y + i * (size + gap)) if vertical else (x + i * (size + gap), y)
        ax.add_patch(Rectangle((xx, yy), size, size, fc=color, ec=ec, lw=0.4))


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- #
# Figure: end-to-end real-time pipeline
# --------------------------------------------------------------------------- #
def fig_pipeline():
    fig, ax = canvas(190, 98)
    box(ax, 1, 36, 25, 18, "Frame source\nscreen (mss)\nvideo · image folder\nphone capture API", fc="white")
    box(ax, 33, 36, 27, 18, "Change detector\n256-bit dHash\n+ 64×36 pixel Δ\n≈ 3 ms")
    box(ax, 33, 70, 27, 16, "Reuse cached verdict\nno OCR · no model\nstill fed to smoother", fc="#F3F3F1", ec=GREY, ls="--")
    box(ax, 68, 56, 33, 18, "OCR\nTesseract 5, sync or async\nML Kit / Vision on phones\n≈ 330 ms at 720p", fc=TFILL, ec=ORANGE)
    box(ax, 68, 16, 33, 18, "Pre-processing\nletterbox to 144 × 256\nImageNet normalisation\n≈ 1.5 ms", fc=VFILL, ec=BLUE)
    box(ax, 110, 28, 30, 34, "ScreenThreatNet\nvision encoder\nbyte-level text encoder\nattention fusion\n3 task heads\n≈ 7 ms, 1 CPU thread", fc=NAVY,
        ec=NAVY, color="white", fs=9.5)
    box(ax, 110, 70, 30, 16, "Rules + URL analysis\n45 signatures · 16 brands\n≈ 1.3 ms", fc=TFILL, ec=ORANGE)
    box(ax, 150, 58, 26, 18, "Decision fusion\nnoisy-OR of model\nand rule evidence", fc=HFILL, ec=AQUA)
    box(ax, 150, 30, 26, 18, "Temporal logic\nEMA smoothing\nalert hysteresis", fc=HFILL, ec=AQUA)
    box(ax, 145, 3, 44, 17, "Alerts with reasons\nconsole · JSONL log · UI\n“paypa1-secure.net imitates PayPal”", fc="white", ec=RED)
    arrow(ax, 26, 45, 33, 45)
    arrow(ax, 46.5, 54, 46.5, 70, text="unchanged\n(≈ 80 % of frames)", tx=-11, ty=0)
    arrow(ax, 60, 49, 68, 63, text="changed", tx=-5.5, ty=2)
    arrow(ax, 60, 41, 68, 27, text="changed", tx=-5.5, ty=-2)
    arrow(ax, 101, 61, 110, 55)
    ax.text(105.5, 62.5, "byte tokens", fontsize=7, color=MUTED, style="italic", ha="center")
    arrow(ax, 101, 29, 110, 35)
    ax.text(105.5, 26.5, "pixels", fontsize=7, color=MUTED, style="italic", ha="center")
    arrow(ax, 101, 70, 110, 76)
    arrow(ax, 140, 55, 150, 63)
    ax.text(145.5, 55.5, "threat", fontsize=7, color=MUTED, style="italic", ha="left")
    arrow(ax, 140, 78, 150, 71)
    arrow(ax, 140, 36, 150, 38)
    ax.text(145, 31.5, "activity /\nNSFW", fontsize=6.8, color=MUTED, style="italic", ha="center", linespacing=0.95)
    arrow(ax, 163, 58, 163, 48)
    arrow(ax, 163, 30, 163, 20)
    poly_arrow(ax, [(46.5, 86), (46.5, 92), (184, 92), (184, 39), (176, 39)], color=GREY, ls="--")
    ax.text(115, 94, "the cached prediction keeps feeding the temporal logic", fontsize=7.5, color=MUTED, style="italic", ha="center")
    save(fig, "fig_pipeline.png")


# --------------------------------------------------------------------------- #
# Figure: ScreenThreatNet
# --------------------------------------------------------------------------- #
def fig_model():
    from PIL import Image

    fig, ax = canvas(190, 85)
    rec = man = None
    for cand in ("data/synth/test_hires/manifest.jsonl", "data/synth/train_hires/manifest.jsonl"):
        m = ROOT / cand
        hits = [json.loads(l) for l in m.read_text().splitlines()]
        hits = [r for r in hits if (r["activity"], r["threat"]) == ("web_browsing", "phishing") and r["device"] == "desktop"]
        if hits:
            rec, man = hits[0], m
            break
    img = Image.open(man.parent / rec["image"])
    ax.imshow(img, extent=(1, 37, 58, 78.25), zorder=2)
    ax.add_patch(Rectangle((1, 58), 36, 20.25, fc="none", ec=GREY, lw=0.6, zorder=3))
    ax.text(19, 81, "Screen frame", ha="center", fontsize=9, fontweight="bold")
    box(ax, 1, 9, 36, 30, "", fc="#FBFBF8", ec=GREY)
    lines = [l for l in rec["text"].splitlines() if len(l.strip()) > 3]
    keep = [l for l in lines if "." in l and "/" in l][:1] + [l for l in lines if "/" not in l][2:9]
    for i, line in enumerate(keep[:8]):
        line = line if len(line) <= 25 else line[:24].rsplit(" ", 1)[0] + " …"
        ax.text(2.6, 36 - i * 3.4, line, fontsize=6.3, family="DejaVu Sans Mono", color=INK, va="center")
    ax.text(19, 42, "OCR text (excerpt)", ha="center", fontsize=9, fontweight="bold")
    # vision path (y centre 68)
    box(ax, 43, 60, 24, 16, "Letterbox\n3 × 144 × 256", fc=VFILL, ec=BLUE)
    box(ax, 72, 60, 31, 16, "Vision encoder\nTinyCNN / MobileNetV3\n/ ViT-tiny", fc=VFILL, ec=BLUE)
    arrow(ax, 37, 68, 43, 68)
    arrow(ax, 67, 68, 72, 68)
    for k in range(3):
        tokens(ax, 107, 65.6 + k * 2.4, 8, 2.0, BLUE, gap=0.35)
    ax.text(107, 62.6, "40 visual tokens × 128", fontsize=6.8, color=MUTED)
    arrow(ax, 103, 68, 106.6, 68)
    # text path (y centre 22)
    box(ax, 43, 13, 24, 18, "Byte tokenizer\n≤ 1 024 bytes\nhead + tail cut", fc=TFILL, ec=ORANGE)
    box(ax, 72, 12, 31, 20, "Byte encoder\nn-gram Conv1d (k = 7)\nmax-pool × 8\nTransformer × 2", fc=TFILL, ec=ORANGE)
    arrow(ax, 37, 22, 43, 22)
    arrow(ax, 67, 22, 72, 22)
    tokens(ax, 107, 22, 10, 1.65, ORANGE, gap=0.25)
    tokens(ax, 107, 19.9, 10, 1.65, ORANGE, gap=0.25)
    ax.text(107, 16.6, "≤ 128 text tokens × 128", fontsize=6.8, color=MUTED)
    arrow(ax, 103, 22, 106.6, 22)
    # fusion
    fx, fy, fw, fh = 128, 29, 32, 34
    box(ax, fx, fy, fw, fh, "", fc=FFILL, ec="#6250D6")
    ax.text(fx + fw / 2, fy + fh - 4, "Attention fusion", ha="center", fontsize=9, fontweight="bold")
    ax.text(fx + fw / 2, fy + fh - 7.8, "Transformer × 2, 4 heads", ha="center", fontsize=7.3, color=MUTED)
    sx = fx + 3.4
    for i in range(3):
        ax.add_patch(Rectangle((sx + i * 2.4, fy + 15), 2.1, 2.1, fc="#6250D6", ec="white", lw=0.4))
    for i in range(4):
        ax.add_patch(Rectangle((sx + 8 + i * 2.4, fy + 15), 2.1, 2.1, fc=BLUE, ec="white", lw=0.4))
    for i in range(3):
        ax.add_patch(Rectangle((sx + 18.4 + i * 2.4, fy + 15), 2.1, 2.1, fc=ORANGE, ec="white", lw=0.4))
    ax.text(fx + fw / 2, fy + 11.5, "[task | visual | text]", ha="center", fontsize=6.6, color=MUTED)
    ax.text(fx + fw / 2, fy + 8.2, "+ modality-type embedding", ha="center", fontsize=6.6, color=MUTED)
    ax.text(fx + fw / 2, fy + 4.9, "absent text is masked", ha="center", fontsize=6.6, color=MUTED)
    ax.text(fx + fw / 2, fy + 1.9, "modality dropout p = 0.15", ha="center", fontsize=6.6, color=MUTED)
    poly_arrow(ax, [(124.5, 68), (144, 68), (144, 63)], color=BLUE)
    poly_arrow(ax, [(125.5, 22), (144, 22), (144, 29)], color=ORANGE)
    for name, sub, y in [("Activity head", "9 classes", 74), ("NSFW head", "3 classes", 46), ("Threat head", "5 classes", 18)]:
        box(ax, 167, y - 7, 22, 14, f"{name}\n{sub}\nsoftmax(z / T)", fc=HFILL, ec=AQUA)
    arrow(ax, 160, 56, 167, 71, color="#6250D6")
    arrow(ax, 160, 46, 167, 46, color="#6250D6")
    arrow(ax, 160, 36, 167, 21, color="#6250D6")
    ax.text(163.5, 66.5, "token 1", fontsize=6.3, color=MUTED, ha="right")
    ax.text(163.5, 47.6, "token 2", fontsize=6.3, color=MUTED, ha="center")
    ax.text(163.5, 26, "token 3", fontsize=6.3, color=MUTED, ha="right")
    ax.text(95, 2.5, "Masked multi-task cross-entropy with learned uncertainty weights:  L = Σ_t exp(−s_t) · L_t + s_t      (1.28 M parameters in total)",
            ha="center", fontsize=7.8, color=NAVY)
    save(fig, "fig_model.png")


# --------------------------------------------------------------------------- #
# Figure: vision encoders
# --------------------------------------------------------------------------- #
def fig_vision():
    fig, ax = canvas(190, 112)
    ax.text(1, 108, "(a) TinyCNN (0.34 M parameters): MobileNetV3-style inverted residual stages", fontsize=9.5, fontweight="bold")
    stages = [("Input", "3", "144×256", 26, "#F3F3F1"), ("Stem\n3×3, s2", "16", "72×128", 22, VFILL),
              ("IR-SE\ns2", "16", "36×64", 18, VFILL), ("IR × 2\ns2, e4", "24", "18×32", 15, VFILL),
              ("IR-SE × 3\ns2, k5", "40→64", "9×16", 13, VFILL), ("IR-SE × 2\ns2, k5", "96", "5×8", 11, VFILL),
              ("1×1 conv\nproject", "128", "5×8", 11, BLUE)]
    x = 2
    cy = 86
    for i, (name, ch, res, h, fc) in enumerate(stages):
        w = 20 if i else 18
        ec = BLUE if fc != "#F3F3F1" else GREY
        ax.add_patch(FancyBboxPatch((x, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        tc = "white" if fc == BLUE else INK
        parts = name.split("\n")
        for j, line in enumerate(parts):
            ax.text(x + w / 2, cy + (len(parts) - 1) * 1.7 - j * 3.4, line, ha="center", va="center", fontsize=7.3,
                    fontweight="bold" if j == 0 else "normal", color=tc)
        ax.text(x + w / 2, cy - h / 2 - 3, f"C = {ch}", ha="center", fontsize=6.8, color=MUTED)
        ax.text(x + w / 2, cy - h / 2 - 6, res, ha="center", fontsize=6.8, color=MUTED)
        if i < len(stages) - 1:
            arrow(ax, x + w, cy, x + w + 5.5, cy, lw=0.9)
        x += w + 5.5
    tokens(ax, x - 2, cy - 4, 4, 2.0, BLUE)
    tokens(ax, x - 2, cy - 1.6, 4, 2.0, BLUE)
    tokens(ax, x - 2, cy + 0.8, 4, 2.0, BLUE)
    ax.text(x + 2.5, cy - 7.5, "40 tokens", ha="center", fontsize=6.8, color=MUTED)
    # (b) inverted residual block
    ax.text(1, 58, "(b) Inverted residual block with squeeze-excite", fontsize=9.5, fontweight="bold")
    blocks = [("1×1 expand", "BN · h-swish"), ("k×k depthwise", "stride s · BN · HS"), ("Squeeze-excite", "GAP · FC · gate"),
              ("1×1 project", "BN, linear")]
    bx = 4
    for i, (n, s) in enumerate(blocks):
        box(ax, bx, 34, 17, 14, f"{n}\n{s}", fc=VFILL if i != 2 else "#E6F4FF", ec=BLUE, fs=7.6)
        if i < 3:
            arrow(ax, bx + 17, 41, bx + 20, 41, lw=0.9)
        bx += 20
    ax.add_patch(Circle((bx + 3.5, 41), 2.6, fc="white", ec=NAVY, lw=0.9))
    ax.text(bx + 3.5, 41, "+", ha="center", va="center", fontsize=10, color=NAVY)
    arrow(ax, bx - 3, 41, bx + 0.9, 41, lw=0.9)
    poly_arrow(ax, [(1, 41), (1, 52), (bx + 3.5, 52), (bx + 3.5, 43.6)], color=GREY, ls="--")
    ax.text((bx + 4) / 2, 54, "identity skip when stride = 1 and C_in = C_out", ha="center", fontsize=7, color=MUTED, style="italic")
    arrow(ax, -0.5, 41, 4, 41, lw=0.9)
    # (c) ViT-tiny
    ax.text(100, 58, "(c) ViT-tiny (1.98 M parameters, encoder only)", fontsize=9.5, fontweight="bold")
    box(ax, 101, 34, 18, 14, "Patchify\n16×16 conv\n9×16 = 144", fc=VFILL, ec=BLUE, fs=7.6)
    box(ax, 123, 34, 19, 14, "+ position emb.\nbilinear-interp.\nfor any size", fc=VFILL, ec=BLUE, fs=7.6)
    box(ax, 146, 30, 22, 22, "Transformer\nencoder × 4\npre-norm · 3 heads\nwidth 192", fc=FFILL, ec="#6250D6", fs=7.8)
    box(ax, 172, 34, 17, 14, "LN · Linear\n192 → 128\n144 tokens", fc=BLUE, ec=BLUE, color="white", fs=7.6)
    arrow(ax, 119, 41, 123, 41, lw=0.9)
    arrow(ax, 142, 41, 146, 41, lw=0.9)
    arrow(ax, 168, 41, 172, 41, lw=0.9)
    # (d) comparison strip
    ax.text(1, 22, "(d) Backbones compared in this project (same 128-d token interface)", fontsize=9.5, fontweight="bold")
    rows = [("TinyCNN", "0.34 M", "from scratch", "best speed/size"), ("MobileNetV3-Small", "1.00 M", "ImageNet-ready", "+ threat F1 at 4k"),
            ("ViT-tiny", "1.98 M", "from scratch", "data-hungry"), ("EfficientNet-B0", "4.17 M", "ImageNet-ready", "accuracy option")]
    for i, (n, p, w, note) in enumerate(rows):
        x0 = 2 + i * 47
        box(ax, x0, 2, 44, 15, f"{n}\nencoder {p} · {w}\n{note}", fc=LIGHT if i else VFILL, ec=BLUE, fs=8)
    save(fig, "fig_vision.png")


# --------------------------------------------------------------------------- #
# Figure: byte text encoder, strided vs n-gram stem
# --------------------------------------------------------------------------- #
def fig_text():
    fig, ax = canvas(190, 96)
    s = "verify your account at paypa1-login.xyz/verify"
    cw = 2.95
    x0 = 9
    def bytes_row(y, highlight=()):
        for i, ch in enumerate(s):
            fc = "#FFF4CC" if any(a <= i < b for a, b in highlight) else "white"
            ax.add_patch(Rectangle((x0 + i * cw, y), cw, 4.2, fc=fc, ec="#C9CDD3", lw=0.4))
            ax.text(x0 + i * cw + cw / 2, y + 2.1, ch if ch != " " else "␣", ha="center", va="center", fontsize=6.3,
                    family="DejaVu Sans Mono", color=INK if ch != " " else GREY)
    hl = [(0, 6), (40, 46)]
    # (a) strided
    ax.text(1, 92, "(a) Strided stem (patch-style): one Conv1d, kernel 16, stride 8", fontsize=9.5, fontweight="bold")
    bytes_row(80, hl)
    for j, k in enumerate(range(0, len(s) - 8, 8)):
        yy = 77.2 if j % 2 == 0 else 74.6
        w16 = min(16, len(s) - k) * cw
        ax.add_patch(FancyBboxPatch((x0 + k * cw + 0.2, yy), w16 - 0.4, 1.8, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=BLUE if j % 2 == 0 else "#7FA8D9", ec="none"))
    ax.text(x0 + len(s) * cw + 4, 80, "“verify” falls at a different\noffset in each window, so a\nseparate detector must be\nlearned for every alignment",
            fontsize=7.3, color=RED, va="center")
    ax.text(x0, 70.5, "overlapping 16-byte windows, stride 8 → one token via position-specific weights (not shift-invariant)", fontsize=7.3, color=MUTED, style="italic")
    # (b) n-gram + max-pool
    ax.text(1, 64, "(b) n-gram stem (used): Conv1d kernel 7, stride 1 → GELU → max-pool ×8", fontsize=9.5, fontweight="bold")
    bytes_row(52, hl)
    for k in range(0, len(s), 8):
        ax.plot([x0 + k * cw, x0 + k * cw], [46, 57], color="#C9CDD3", lw=0.5)
    import numpy as np

    rng = np.random.default_rng(3)
    act = rng.uniform(0.05, 0.25, len(s))
    for a, b in hl:
        act[a:b] = [0.4, 0.7, 1.0, 0.85, 0.6, 0.35][: b - a]
    for i, v in enumerate(act):
        ax.add_patch(Rectangle((x0 + i * cw + 0.3, 46), cw - 0.6, 5 * v, fc=ORANGE, ec="none", alpha=0.85))
    ax.text(x0 - 1, 47.5, "detector\n“verify”", fontsize=6.5, color=MUTED, ha="right", va="center")
    pooled = [max(act[k:k + 8]) for k in range(0, len(s), 8)]
    for j, v in enumerate(pooled):
        wj = min(8, len(s) - j * 8) * cw
        ax.add_patch(Rectangle((x0 + j * 8 * cw + 1, 38.5), wj - 2, 4.2, fc=ORANGE, ec="white", alpha=0.25 + 0.75 * v))
    ax.text(x0 + len(s) * cw + 4, 44, "max-pooled tokens:\nboth occurrences of\n“verify” light up", fontsize=7.3, color=ORANGE, va="center")
    ax.text(x0, 35.3, "max-pool × 8: the same detector fires wherever the n-gram occurs (local shift invariance); 1 024 bytes → 128 tokens",
            fontsize=7.3, color=MUTED, style="italic")
    # (c) rest of encoder + evidence
    ax.text(1, 27, "(c) Remainder of the byte encoder", fontsize=9.5, fontweight="bold")
    items = [("Byte embedding", "260 ids → 64-d"), ("n-gram Conv1d", "64 → 128, k = 7"), ("Max-pool ×8", "+ padding mask"),
             ("+ learned positions", "130 × 128"), ("Transformer ×2", "pre-norm, 4 heads"), ("Text tokens", "N × 128 + valid mask")]
    bx = 2
    for i, (n, sub) in enumerate(items):
        box(ax, bx, 9, 27, 12, f"{n}\n{sub}", fc=TFILL if i < 5 else ORANGE, ec=ORANGE, color="white" if i == 5 else INK, fs=7.8)
        if i < 5:
            arrow(ax, bx + 27, 15, bx + 31, 15, lw=0.9)
        bx += 31
    ax.text(95, 3, "Ablation (4 000 screens, 3 epochs): threat macro-F1 0.578 with the strided stem vs 0.912 with the n-gram stem at equal cost",
            ha="center", fontsize=7.8, color=NAVY, fontweight="bold")
    save(fig, "fig_text_encoder.png")


# --------------------------------------------------------------------------- #
# Figure: fusion strategies
# --------------------------------------------------------------------------- #
def fig_fusion():
    fig, ax = canvas(190, 72)
    titles = ["(a) Concat (late fusion)", "(b) Gated multimodal unit", "(c) Attention fusion (used)"]
    for i, t in enumerate(titles):
        ax.text(2 + i * 64, 68, t, fontsize=9.5, fontweight="bold")
    # (a)
    tokens(ax, 4, 60, 6, 2.0, BLUE)
    tokens(ax, 34, 60, 6, 2.0, ORANGE)
    ax.text(10, 63.2, "visual tokens", ha="center", fontsize=6.8, color=MUTED)
    ax.text(40, 63.2, "text tokens", ha="center", fontsize=6.8, color=MUTED)
    box(ax, 4, 44, 13, 8, "mean-pool", fc=VFILL, ec=BLUE, fs=7.5)
    box(ax, 32.5, 44, 16, 8, "masked mean", fc=TFILL, ec=ORANGE, fs=7.5)
    arrow(ax, 10.5, 59.5, 10.5, 52)
    arrow(ax, 40.5, 59.5, 40.5, 52)
    box(ax, 15, 28, 22, 9, "concat [v ; t]", fc="white", fs=7.8)
    arrow(ax, 10.5, 44, 20, 37)
    arrow(ax, 40.5, 44, 32, 37)
    box(ax, 13, 12, 26, 10, "LN · Linear · GELU\nshared feature h", fc=FFILL, ec="#6250D6", fs=7.8)
    arrow(ax, 26, 28, 26, 22)
    ax.text(26, 5, "3 heads share h\n0.92 M params · 4.4 ms", ha="center", fontsize=7.2, color=MUTED)
    # (b)
    ox = 64
    tokens(ax, ox + 4, 60, 6, 2.0, BLUE)
    tokens(ax, ox + 34, 60, 6, 2.0, ORANGE)
    box(ax, ox + 0.5, 44, 21, 9, "h_v = tanh(W_v v)", fc=VFILL, ec=BLUE, fs=7.3)
    box(ax, ox + 30.5, 44, 21, 9, "h_t = tanh(W_t t)", fc=TFILL, ec=ORANGE, fs=7.3)
    arrow(ax, ox + 10.5, 59.5, ox + 10.5, 53)
    arrow(ax, ox + 40.5, 59.5, ox + 40.5, 53)
    box(ax, ox + 15, 30, 22, 9, "z = σ(W_z [v ; t])", fc="white", fs=7.3)
    arrow(ax, ox + 13, 59.5, ox + 22, 39, color=GREY, lw=0.8)
    arrow(ax, ox + 38, 59.5, ox + 30, 39, color=GREY, lw=0.8)
    box(ax, ox + 8, 12, 36, 10, "h = z · h_v + (1 − z) · h_t\nelement-wise modality gate", fc=FFILL, ec="#6250D6", fs=7.6)
    arrow(ax, ox + 10.5, 44, ox + 14, 22)
    arrow(ax, ox + 40.5, 44, ox + 38, 22)
    arrow(ax, ox + 26, 30, ox + 26, 22)
    ax.text(ox + 26, 5, "best at small data (threat F1 0.976)\n0.95 M params · 3.9 ms", ha="center", fontsize=7.2, color=MUTED)
    # (c)
    ox = 128
    seq_y = 60
    for i in range(3):
        ax.add_patch(Rectangle((ox + 2 + i * 3.2, seq_y), 2.8, 2.8, fc="#6250D6", ec="white", lw=0.4))
    for i in range(5):
        ax.add_patch(Rectangle((ox + 13 + i * 3.2, seq_y), 2.8, 2.8, fc=BLUE, ec="white", lw=0.4))
    for i in range(5):
        ax.add_patch(Rectangle((ox + 31.5 + i * 3.2, seq_y), 2.8, 2.8, fc=ORANGE, ec="white", lw=0.4))
    ax.text(ox + 6, 57, "task", ha="center", fontsize=6.8, color=MUTED)
    ax.text(ox + 20.5, 57, "visual (40)", ha="center", fontsize=6.8, color=MUTED)
    ax.text(ox + 39, 57, "text (≤128)", ha="center", fontsize=6.8, color=MUTED)
    ax.text(ox + 29, 64.5, "+ modality-type embeddings", ha="center", fontsize=6.8, color=MUTED)
    box(ax, ox + 4, 34, 50, 16, "Transformer encoder ×2\nself-attention over all tokens\nkey-padding mask hides absent text", fc=FFILL, ec="#6250D6", fs=7.8)
    arrow(ax, ox + 29, 56, ox + 29, 50)
    for i, (lab, col) in enumerate([("activity", AQUA), ("NSFW", AQUA), ("threat", AQUA)]):
        box(ax, ox + 4 + i * 17, 14, 15, 9, f"{lab}\nhead", fc=HFILL, ec=col, fs=7.4)
        arrow(ax, ox + 11.5 + i * 17, 34, ox + 11.5 + i * 17, 23)
    ax.text(ox + 29, 5, "each task reads its own query token\n1.28 M params · 7.3 ms · best at 16k screens",
            ha="center", fontsize=7.2, color=MUTED)
    for xx in (63, 127):
        ax.plot([xx, xx], [3, 68], color="#E4E3DF", lw=0.8)
    save(fig, "fig_fusion.png")


# --------------------------------------------------------------------------- #
# Figure: decision layer and alert state machine
# --------------------------------------------------------------------------- #
def fig_decision():
    fig, ax = canvas(190, 86)
    ax.text(1, 82, "(a) Explainable evidence from OCR text", fontsize=9.5, fontweight="bold")
    box(ax, 2, 56, 38, 20, "URL analysis\nregistered-domain parsing\nhomoglyph normalisation\nDamerau–Levenshtein to 16 brands\nIP host · punycode · abused TLD", fc=TFILL, ec=ORANGE, fs=8)
    box(ax, 2, 30, 38, 20, "Signature rules (45)\nencoded PowerShell · curl | sh\nLOLBins · reverse shells\nClickFix · double extensions\ntech-support & crypto scams", fc=TFILL, ec=ORANGE, fs=8)
    box(ax, 2, 8, 38, 16, "Per-class evidence\nr_c = 1 − Π_i (1 − w_i)\nURL score s = σ(Σ w_k f_k − 3)", fc="white", ec=ORANGE, fs=8)
    arrow(ax, 21, 56, 21, 50, color=ORANGE)
    arrow(ax, 21, 30, 21, 24, color=ORANGE)
    ax.text(70, 82, "(b) Decision fusion", fontsize=9.5, fontweight="bold")
    box(ax, 50, 46, 50, 28, "Noisy-OR fusion\nq_c = 1 − (1 − p_c)(1 − r_c),  c ≠ benign\nq_benign = p_benign · Π_c (1 − r_c)\nrenormalise · risk = 1 − q_benign", fc=HFILL, ec=AQUA, fs=8.2)
    box(ax, 50, 14, 50, 22, "Calibrated model probabilities p\nsoftmax(z / T), T fitted on validation\nand clamped to [0.25, 10]", fc=FFILL, ec="#6250D6", fs=8)
    arrow(ax, 40, 16, 50, 52, color=ORANGE)
    arrow(ax, 75, 36, 75, 46, color="#6250D6")
    ax.text(127, 82, "(c) Alert hysteresis per alert kind", fontsize=9.5, fontweight="bold")
    ax.add_patch(Circle((128, 48), 11, fc=LIGHT, ec=NAVY, lw=1.1))
    ax.text(128, 50, "IDLE", ha="center", fontsize=9, fontweight="bold", color=NAVY)
    ax.text(128, 45.5, "EMA α = 0.6", ha="center", fontsize=7, color=MUTED)
    ax.add_patch(Circle((172, 48), 11, fc="#FDE3E7", ec=RED, lw=1.1))
    ax.text(172, 50, "ALERT", ha="center", fontsize=9, fontweight="bold", color=RED)
    ax.text(172, 45.5, "label L", ha="center", fontsize=7, color=MUTED)
    arrow(ax, 137, 55, 163, 55, color=RED, rad=-0.35, text="risk ≥ 0.70 for 2 frames\nand label cooled (30 s)", ty=9, tcolor=RED)
    arrow(ax, 163, 41, 137, 41, color=NAVY, rad=-0.35, text="risk < 0.40 for 3 frames\n→ cleared", ty=-9.5)
    ax.add_patch(FancyArrowPatch((180, 56), (183, 47), connectionstyle="arc3,rad=-1.8", arrowstyle="-|>", mutation_scale=9, color=RED, lw=1))
    ax.text(186, 62, "new label ≥ 0.70\nfor 2 frames\n→ re-raise", fontsize=6.8, color=RED, ha="center")
    arrow(ax, 100, 60, 117, 50, color=AQUA, text="risk per frame", ty=3, tx=2)
    ax.text(150, 16, "Skipped (unchanged) frames re-feed the cached prediction,\nso a static threat still accumulates the frames it needs.",
            ha="center", fontsize=7.4, color=MUTED, style="italic")
    ax.text(150, 6, "Cooldown is per label: a new kind of threat alerts immediately.", ha="center", fontsize=7.4, color=MUTED, style="italic")
    save(fig, "fig_decision.png")


# --------------------------------------------------------------------------- #
# Result figures specific to the report
# --------------------------------------------------------------------------- #
def fig_timeline():
    rs = [json.loads(l) for l in (ROOT / "runs/monitor_events.jsonl").read_text().splitlines()][-65:]
    from screenguard.realtime.capture import SyntheticSession

    script = SyntheticSession.DEFAULT_SCRIPT
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    for k, (act, thr) in enumerate(script):
        x0 = k * 5 - 0.5
        ax.axvspan(x0, x0 + 5, color="#FDE3E7" if thr != "benign" else ("#F6F6F4" if k % 2 else "white"), lw=0)
        lab = act.replace("_", " ").replace("document editing", "document").replace("video streaming", "video") \
            .replace("file download", "download").replace("social media", "social").replace("web browsing", "web") \
            .replace("code editor", "code")
        ax.text(x0 + 2.5, 1.04, lab, rotation=90, ha="center", va="bottom", fontsize=6.6, color=RED if thr != "benign" else MUTED,
                transform=ax.get_xaxis_transform())
    xs = [r["frame"] for r in rs]
    risk = [r["threat"]["risk"] for r in rs]
    ax.plot(xs, risk, color=NAVY, lw=1.4, zorder=3)
    ax.scatter([x for x, r in zip(xs, rs) if r["analyzed"]], [k for k, r in zip(risk, rs) if r["analyzed"]], s=16, color=NAVY,
               zorder=4, label="analysed (OCR + model)")
    ax.scatter([x for x, r in zip(xs, rs) if not r["analyzed"]], [k for k, r in zip(risk, rs) if not r["analyzed"]], s=12,
               facecolor="white", edgecolor=NAVY, lw=0.8, zorder=4, label="skipped (cached)")
    ax.axhline(0.70, color=RED, lw=0.9, ls="--")
    ax.axhline(0.40, color=GREY, lw=0.9, ls="--")
    ax.text(64.6, 0.71, "on 0.70", fontsize=7, color=RED, ha="right", va="bottom")
    ax.text(64.6, 0.41, "off 0.40", fontsize=7, color=GREY, ha="right", va="bottom")
    first = True
    for r in rs:
        for a in r["alerts"]:
            if a["status"] == "raised":
                ax.scatter([r["frame"]], [r["threat"]["risk"] + 0.06], marker="v", s=40, color=RED, zorder=5,
                           label="alert raised" if first else None)
                first = False
    ax.set_xlim(-0.5, 64.5)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("frame index (scripted session, 13 screens × 5 frames)", fontsize=8.5)
    ax.set_ylabel("smoothed threat risk", fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.8, frameon=True, framealpha=0.95, edgecolor="#E4E3DF", loc="upper left", bbox_to_anchor=(0.003, 0.985),
              handletextpad=0.3, borderpad=0.4)
    fig.tight_layout()
    save(fig, "fig_timeline.png")


def fig_ocr_gap():
    import numpy as np

    main = json.loads((ROOT / "docs/results/test_hires_ocr_text.json").read_text())
    ft = json.loads((ROOT / "docs/results/ocr_finetune/test_hires_ocr_text.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), gridspec_kw={"width_ratios": [9, 5]})
    for ax, task, title in [(axes[0], "activity", "Activity recall"), (axes[1], "threat", "Threat recall")]:
        cls = list(main["model"][task]["per_class"].keys())
        a = [main["model"][task]["per_class"][c]["recall"] for c in cls]
        b = [ft["model"][task]["per_class"][c]["recall"] for c in cls]
        x = np.arange(len(cls))
        ax.bar(x - 0.2, a, 0.38, color=BLUE, label="main model", edgecolor="white", lw=0.6)
        ax.bar(x + 0.2, b, 0.38, color=ORANGE, label="+ OCR-aware fine-tune", edgecolor="white", lw=0.6)
        short = {"web_browsing": "web", "video_streaming": "video", "messaging": "chat", "email": "email",
                 "document_editing": "doc", "code_editor": "code", "terminal": "terminal", "social_media": "social",
                 "file_download": "download", "benign": "benign", "phishing": "phishing", "scam": "scam",
                 "malware_download": "malware", "suspicious_command": "command"}
        ax.set_xticks(x, [short[c] for c in cls], fontsize=7.2, rotation=30)
        ax.set_ylim(0.6, 1.03)
        ax.set_title(title + " (720p, real OCR)", fontsize=8.5, loc="left", fontweight="bold", pad=6)
        ax.tick_params(labelsize=7.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#E4E3DF", lw=0.6)
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.4, frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig_ocr_gap.png")


def fig_training():
    log = [json.loads(l) for l in (ROOT / "runs/main/train_log.jsonl").read_text().splitlines()]
    ep = [r["epoch"] for r in log]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    axes[0].plot(ep, [r["train_loss"]["activity"] for r in log], marker="o", ms=3.5, color=BLUE, label="activity")
    axes[0].plot(ep, [r["train_loss"]["threat"] for r in log], marker="o", ms=3.5, color=ORANGE, label="threat")
    axes[0].set_title("Training loss (label-smoothed CE)", fontsize=8.5, loc="left", fontweight="bold")
    axes[0].legend(fontsize=7.2, frameon=False)
    axes[1].plot(ep, [r["val_activity_acc"] for r in log], marker="o", ms=3.5, color=BLUE, label="activity accuracy")
    axes[1].plot(ep, [r["val_threat_f1"] for r in log], marker="o", ms=3.5, color=ORANGE, label="threat macro-F1")
    axes[1].set_title("Validation metrics", fontsize=8.5, loc="left", fontweight="bold")
    axes[1].set_ylim(0.96, 1.002)
    axes[1].legend(fontsize=7.2, frameon=False, loc="lower right")
    for ax in axes:
        ax.set_xlabel("epoch", fontsize=8)
        ax.tick_params(labelsize=7.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color="#E4E3DF", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_training.png")


def fig_console():
    """Render real monitor output as a terminal-style image."""
    from PIL import Image, ImageDraw, ImageFont

    lines = (ROOT / "docs/results/monitor_demo.txt").read_text().splitlines()
    keep = [l for l in lines[1:] if l.strip() and not l.startswith(("{", "}", '  "', "alerts raised"))][:24]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 26)
    W, lh = 2300, 38
    H = 90 + lh * len(keep) + 30
    im = Image.new("RGB", (W, H), (24, 26, 33))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 54), fill=(44, 47, 56))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((24 + i * 38, 15, 48 + i * 38, 39), fill=c)
    d.text((W // 2 - 330, 13), "screenguard monitor --source synthetic:5 --ocr tesseract", font=font, fill=(200, 200, 205))
    y = 74
    for l in keep:
        col = (215, 218, 224)
        if "ALERT" in l:
            col = (255, 107, 107)
        elif l.strip().startswith("- "):
            col = (255, 196, 140)
        elif "cleared" in l:
            col = (120, 220, 150)
        elif "threat=benign" not in l and "threat=" in l:
            col = (255, 160, 160)
        d.text((28, y), l.replace("⚠", "!").replace("✓", "v"), font=font, fill=col)
        y += lh
    im.save(OUT / "fig_console.png")
    print("wrote fig_console.png")


def fig_composites():
    """Copy/crop/compose figures from docs/figures/ (made by scripts/make_figures.py) for the report."""
    import numpy as np
    from PIL import Image

    D = ROOT / "docs/figures"

    def hcat(paths, out, gap):
        ims = [Image.open(q).convert("RGB") for q in paths]
        h = max(i.height for i in ims)
        ims = [i.resize((int(i.width * h / i.height), h), Image.LANCZOS) for i in ims]
        c = Image.new("RGB", (sum(i.width for i in ims) + gap * (len(ims) - 1), h), "white")
        x = 0
        for i in ims:
            c.paste(i, (x, 0))
            x += i.width + gap
        c.save(out)

    def vcat(paths, out, gap):
        ims = [Image.open(q).convert("RGB") for q in paths]
        w = max(i.width for i in ims)
        ims = [i.resize((w, int(i.height * w / i.width)), Image.LANCZOS) for i in ims]
        c = Image.new("RGB", (w, sum(i.height for i in ims) + gap * (len(ims) - 1)), "white")
        y = 0
        for i in ims:
            c.paste(i, (0, y))
            y += i.height + gap
        c.save(out)

    def crop_suptitle(img):
        a = np.asarray(img).astype(int)
        nonwhite = (np.abs(255 - a).sum(axis=2) > 60).any(axis=1)
        y = int(np.where(nonwhite)[0][0])
        while y < len(nonwhite) and nonwhite[y]:
            y += 1
        while y < len(nonwhite) and not nonwhite[y]:
            y += 1
        return img.crop((0, max(0, y - 8), img.width, img.height))

    hcat([D / "confusion_activity.png", D / "confusion_threat.png"], OUT / "fig_confusion.png", 60)
    vcat([D / "robustness_threat.png", D / "robustness_activity.png"], OUT / "fig_robustness.png", 40)
    for n in ("ablations.png", "latency.png"):
        Image.open(D / n).convert("RGB").save(OUT / n)
    for n in ("synthetic_samples.png", "gradcam_examples.png"):
        crop_suptitle(Image.open(D / n).convert("RGB")).save(OUT / n)
    print("wrote composites")


if __name__ == "__main__":
    which = sys.argv[1:] or ["pipeline", "model", "vision", "text", "fusion", "decision", "timeline", "ocr_gap", "training", "console",
                             "composites"]
    for w in which:
        globals()[f"fig_{w}"]()
