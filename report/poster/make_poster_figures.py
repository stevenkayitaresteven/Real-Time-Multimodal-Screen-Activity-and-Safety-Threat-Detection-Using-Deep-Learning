"""Figures for the A1 poster, drawn at their exact size on the poster (inches), so font sizes are poster points."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/poster/figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

for f in Path("/usr/share/fonts/truetype/crosextra").glob("Carlito*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Carlito", "font.size": 13, "savefig.dpi": 300, "figure.facecolor": "white",
                     "axes.facecolor": "white", "mathtext.fontset": "dejavusans"})

# poster palette (template navy / blue / red + modality colours used in the report)
NAVY, BLUE, RED = "#0B2240", "#005FAD", "#D92B3A"
ORANGE, AQUA, PURPLE, GREY, INK, MUTED = "#E0662B", "#16996B", "#5B48C9", "#8A8F98", "#1E293B", "#475569"
VFILL, TFILL, FFILL, HFILL, SOFT = "#DCEBFA", "#FDE7DC", "#E9E4FB", "#DFF5EC", "#F1F5F9"


def canvas(w_in: float, h_in: float):
    """Axes in millimetres over a figure of exactly w_in x h_in inches."""
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_in * 25.4)
    ax.set_ylim(0, h_in * 25.4)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, lines=(), fc=SOFT, ec=NAVY, lw=1.6, fs=13, color=INK, r=2.2, ls="-", sub_fs=None, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if isinstance(lines, str):
        lines = lines.split("\n")
    n = len(lines)
    sub_fs = sub_fs or fs - 2
    lh = [fs * 0.42 if i == 0 else sub_fs * 0.42 for i in range(n)]
    total = sum(lh)
    yy = y + h / 2 + total / 2
    for i, line in enumerate(lines):
        yy -= lh[i] / 2
        ax.text(x + w / 2, yy, line, ha="center", va="center", fontsize=fs if i == 0 else sub_fs,
                fontweight="bold" if i == 0 else "normal", color=color if (i == 0 or color != INK) else MUTED, zorder=z + 1)
        yy -= lh[i] / 2


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.8, ls="-", rad=0.0, ms=16, z=1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def poly_arrow(ax, pts, color=NAVY, lw=1.8, ls="-"):
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)
    arrow(ax, *pts[-2], *pts[-1], color=color, lw=lw, ls=ls)


def tokens(ax, x, y, n, size, color, gap=0.5):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * (size + gap), y), size, size, fc=color, ec="white", lw=0.6, zorder=3))


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", name)


def find(man, image):
    return ROOT / "data/synth" / man / image


# --------------------------------------------------------------------------- #
def fig_threats(w=8.6, h=1.62):
    """Four rendered threat screens the system has to catch (Background card)."""
    from PIL import Image

    cases = [("train_hires", "images/000068.jpg", "Phishing", "fake Google sign-in page"),
             ("test_hires", "images/000004.jpg", "Scam", "“send 1 BTC, get 2 back”"),
             ("train_hires", "images/000102.jpg", "Malware download", "fake update → .exe file"),
             ("test_hires", "images/000246.jpg", "Suspicious command", "ClickFix “verify” paste")]
    fig, ax = canvas(w, h)
    W = w * 25.4
    gap = 3.0
    tw = (W - 3 * gap) / 4
    th = tw * 9 / 16
    y0 = h * 25.4 - th - 0.6
    for k, (man, img, title, sub) in enumerate(cases):
        x = k * (tw + gap)
        ax.imshow(Image.open(find(man, img)).convert("RGB"), extent=(x, x + tw, y0, y0 + th), zorder=2, interpolation="lanczos")
        ax.add_patch(Rectangle((x, y0), tw, th, fc="none", ec=RED, lw=2.2, zorder=3))
        ax.text(x + 1.2, y0 - 3.4, title, fontsize=13.5, fontweight="bold", color=RED, va="center")
        ax.text(x + 1.2, y0 - 8.4, sub, fontsize=11.5, color=MUTED, va="center")
    fig.patch.set_alpha(0)
    save(fig, "poster_threats.png")


# --------------------------------------------------------------------------- #
def fig_architecture(w=8.95, h=4.75):
    """Top: the real-time pipeline. Bottom: zoom into the ScreenThreatNet multimodal multi-task network."""
    fig, ax = canvas(w, h)
    W, H = w * 25.4, h * 25.4  # 227 x 120.6 mm
    # ---- (a) pipeline -------------------------------------------------------
    ax.text(1, H - 4, "(a) Real-time pipeline (per frame, 1 CPU thread)", fontsize=14, fontweight="bold", color=NAVY, va="center")
    ty, bh = H - 30, 19
    steps = [("Screen frame", "laptop / phone", "white", GREY),
             ("Change test", "dHash · 3 ms", SOFT, NAVY),
             ("OCR", "Tesseract · 330 ms", TFILL, ORANGE),
             ("ScreenThreatNet", "int8 ONNX · 7 ms", NAVY, NAVY),
             ("Rules + URLs", "noisy-OR · 1 ms", TFILL, ORANGE),
             ("Alert", "EMA + hysteresis", "#FDE3E7", RED)]
    bw, gap = 32.8, 5.9
    xs = [1 + i * (bw + gap) for i in range(len(steps))]
    for x, (t, s, fc, ec) in zip(xs, steps):
        box(ax, x, ty, bw, bh, [t, s], fc=fc, ec=ec, fs=13, sub_fs=11.5, color="white" if fc == NAVY else INK)
    for x in xs[:-1]:
        arrow(ax, x + bw + 0.3, ty + bh / 2, x + bw + gap - 0.3, ty + bh / 2)
    ax.text(W - 1, H - 4, "80 % of frames are unchanged → cached verdict", fontsize=11.5, color=MUTED, style="italic",
            ha="right", va="center")
    # zoom guides from ScreenThreatNet box to panel (b)
    zx = xs[3]
    py_top = ty - 8.5
    ax.plot([zx, 1], [ty, py_top], color=NAVY, lw=1.0, ls=(0, (3, 2)), zorder=0)
    ax.plot([zx + bw, W - 1], [ty, py_top], color=NAVY, lw=1.0, ls=(0, (3, 2)), zorder=0)
    ax.add_patch(FancyBboxPatch((1, 1), W - 2, py_top - 1, boxstyle="round,pad=0,rounding_size=3", fc="#FAFBFD", ec=NAVY,
                                lw=1.0, ls=(0, (3, 2)), zorder=0))
    # ---- (b) ScreenThreatNet ------------------------------------------------
    from PIL import Image

    top = py_top - 5.5
    ax.text(4, top, "(b) ScreenThreatNet: multimodal multi-task network (1.28 M parameters)", fontsize=14, fontweight="bold",
            color=NAVY, va="center")
    lo, hi = 5.0, top - 6.0  # usable band inside the panel
    vc, tc = hi - 12.5, lo + 12.5  # centres of the vision row and the text row
    # inputs: the frame itself and an excerpt of its OCR text
    iw, ih = 36, 20.25
    ax.imshow(Image.open(find("train_hires", "images/000068.jpg")).convert("RGB"), extent=(4, 4 + iw, vc - ih / 2, vc + ih / 2),
              zorder=2, interpolation="lanczos")
    ax.add_patch(Rectangle((4, vc - ih / 2), iw, ih, fc="none", ec=BLUE, lw=1.6, zorder=3))
    ax.text(4 + iw / 2, vc - ih / 2 - 3.2, "pixels, 144 × 256", fontsize=11, color=MUTED, ha="center", va="center")
    box(ax, 4, tc - 10, iw, 20, [], fc="white", ec=ORANGE)
    for i, line in enumerate(["an-googl-4bs.com", "Sign in to Google", "Unusual sign-in …"]):
        ax.text(6, tc + 5.6 - i * 5.6, line, fontsize=11.5, color=RED if i == 0 else INK, va="center",
                fontweight="bold" if i == 0 else "normal", zorder=4)
    ax.text(4 + iw / 2, tc + 13.2, "OCR text, ≤ 1,024 bytes", fontsize=11, color=MUTED, ha="center", va="center")
    # encoders
    ex, ew, eh = 46, 50, 22
    box(ax, ex, vc - eh / 2, ew, eh, ["Vision encoder", "TinyCNN · MobileNetV3", "or ViT-tiny → feature map"], fc=VFILL, ec=BLUE,
        sub_fs=11.5)
    box(ax, ex, tc - eh / 2, ew, eh, ["Byte encoder", "n-gram CNN (k = 7), pool ×8", "+ Transformer ×2"], fc=TFILL, ec=ORANGE,
        sub_fs=11.5)
    arrow(ax, 4 + iw + 0.3, vc, ex - 0.3, vc, color=BLUE)
    arrow(ax, 4 + iw + 0.3, tc, ex - 0.3, tc, color=ORANGE)
    ax.text(ex + ew / 2, (vc + tc) / 2 + 2.6, "modality dropout: text hidden", fontsize=11, color=PURPLE, ha="center",
            va="center", style="italic")
    ax.text(ex + ew / 2, (vc + tc) / 2 - 2.6, "on 15 % of training screens", fontsize=11, color=PURPLE, ha="center", va="center",
            style="italic")
    # token strips
    kx = ex + ew + 5
    for c, col, lab in [(vc, BLUE, "40 visual tokens"), (tc, ORANGE, "≤ 128 text tokens")]:
        tokens(ax, kx, c + 0.4, 6, 3.0, col, gap=0.55)
        tokens(ax, kx, c - 3.4, 6, 3.0, col, gap=0.55)
        arrow(ax, ex + ew + 0.3, c, kx - 0.4, c, color=col)
    ax.text(kx + 10.5, vc - 7.0, "40 visual tokens", fontsize=11, color=MUTED, ha="center", va="center")
    ax.text(kx + 10.5, tc + 7.0, "≤ 128 text tokens", fontsize=11, color=MUTED, ha="center", va="center")
    # fusion block
    fx, fw = kx + 27, 50
    fy, fh = lo, hi - lo
    box(ax, fx, fy, fw, fh, [], fc=FFILL, ec=PURPLE)
    cx = fx + fw / 2
    ax.text(cx, fy + fh - 6, "Attention fusion", fontsize=13.5, fontweight="bold", color=INK, ha="center", va="center")
    ax.text(cx, fy + fh - 12, "Transformer ×2, 4 heads", fontsize=11.5, color=MUTED, ha="center", va="center")
    seq_y = fy + fh / 2 - 1.5
    sx = fx + 5
    for i in range(3):
        ax.add_patch(Rectangle((sx + i * 3.6, seq_y), 3.1, 3.1, fc=PURPLE, ec="white", lw=0.6, zorder=3))
    for i in range(4):
        ax.add_patch(Rectangle((sx + 12.5 + i * 3.6, seq_y), 3.1, 3.1, fc=BLUE, ec="white", lw=0.6, zorder=3))
    for i in range(4):
        ax.add_patch(Rectangle((sx + 28.5 + i * 3.6, seq_y), 3.1, 3.1, fc=ORANGE, ec="white", lw=0.6, zorder=3))
    ax.text(cx, seq_y + 7.2, "one query token per task", fontsize=11, color=MUTED, ha="center", va="center")
    ax.text(cx, seq_y - 4.2, "[ task | visual | text ]", fontsize=11.5, color=MUTED, ha="center", va="center")
    ax.text(cx, fy + 11, "+ modality-type embeddings", fontsize=11, color=MUTED, ha="center", va="center")
    ax.text(cx, fy + 5, "missing text is masked", fontsize=11, color=MUTED, ha="center", va="center", style="italic")
    arrow(ax, kx + 21.5, vc, fx - 0.3, vc, color=BLUE)
    arrow(ax, kx + 21.5, tc, fx - 0.3, tc, color=ORANGE)
    # heads
    hx = fx + fw + 7
    hw, hh = W - 4 - hx, 14
    heads = [("Activity", "9 classes"), ("NSFW", "3 classes"), ("Threat", "5 classes")]
    for k, (t, s_) in enumerate(heads):
        yy = fy + fh - (k + 1) * (fh / 3) + (fh / 3 - hh) / 2
        box(ax, hx, yy, hw, hh, [t, s_], fc=HFILL, ec=AQUA, fs=13, sub_fs=11)
        arrow(ax, fx + fw + 0.3, fy + fh / 2, hx - 0.3, yy + hh / 2, color=PURPLE, lw=1.5)
    save(fig, "poster_architecture.png")


# --------------------------------------------------------------------------- #
def fig_timeline(w=8.8, h=2.75):
    """Smoothed risk over the scripted session with real OCR: alerts fire on all 5 threats, none on benign screens."""
    rs = [json.loads(l) for l in (ROOT / "runs/monitor_events.jsonl").read_text().splitlines()][-65:]
    from screenguard.realtime.capture import SyntheticSession

    script = SyntheticSession.DEFAULT_SCRIPT
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0.075, 0.2, 0.915, 0.6])
    short = {"web_browsing": "web", "video_streaming": "video", "messaging": "chat", "email": "email", "document_editing": "docs",
             "code_editor": "code", "terminal": "term.", "social_media": "social", "file_download": "files"}
    for k, (act, thr) in enumerate(script):
        x0 = k * 5 - 0.5
        ax.axvspan(x0, x0 + 5, color="#FDE3E7" if thr != "benign" else ("#F3F5F8" if k % 2 else "white"), lw=0)
        ax.text(x0 + 2.5, 1.03, short.get(act, act), ha="center", va="bottom", fontsize=11,
                color=RED if thr != "benign" else MUTED, fontweight="bold" if thr != "benign" else "normal",
                transform=ax.get_xaxis_transform())
    xs = [r["frame"] for r in rs]
    risk = [r["threat"]["risk"] for r in rs]
    ax.plot(xs, risk, color=NAVY, lw=2.0, zorder=3)
    ax.scatter([x for x, r in zip(xs, rs) if r["analyzed"]], [k for k, r in zip(risk, rs) if r["analyzed"]], s=38, color=NAVY,
               zorder=4, label="analysed (OCR + model)")
    ax.scatter([x for x, r in zip(xs, rs) if not r["analyzed"]], [k for k, r in zip(risk, rs) if not r["analyzed"]], s=24,
               facecolor="white", edgecolor=NAVY, lw=1.0, zorder=4, label="skipped (cached)")
    ax.axhline(0.70, color=RED, lw=1.2, ls="--")
    ax.axhline(0.40, color=GREY, lw=1.2, ls="--")
    ax.text(0.3, 0.72, "alert on ≥ 0.70", fontsize=11, color=RED, ha="left", va="bottom")
    ax.text(0.3, 0.42, "clear below 0.40", fontsize=11, color=GREY, ha="left", va="bottom")
    first = True
    for r in rs:
        for a in r["alerts"]:
            if a["status"] == "raised":
                ax.scatter([r["frame"]], [r["threat"]["risk"] + 0.08], marker="v", s=110, color=RED, zorder=5,
                           label="alert raised" if first else None)
                first = False
    ax.set_xlim(-0.5, 64.5)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("frame (13 screens × 5 frames; red = threat screen)", fontsize=12, color=MUTED, labelpad=2)
    ax.set_ylabel("threat risk", fontsize=12, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CBD5E1")
    ax.tick_params(labelsize=11, colors=MUTED)
    ax.set_yticks([0, 0.5, 1.0])
    fig.legend(fontsize=11, frameon=False, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncols=3, handletextpad=0.3,
               columnspacing=1.4)
    save(fig, "poster_timeline.png")


# --------------------------------------------------------------------------- #
def fig_fusion(w=8.95, h=2.3):
    """Three fusion strategies that were compared (concat, gated, attention)."""
    fig, ax = canvas(w, h)
    W, H = w * 25.4, h * 25.4  # 227 x 65 mm
    cw = W / 3
    titles = ["Concat", "Gated (GMU)", "Attention (ours)"]
    notes = ["threat F1 0.955 (4k screens)", "threat F1 0.976 (4k, best)", "0.912 at 4k → 0.994 at 16k"]
    for i, (t, n) in enumerate(zip(titles, notes)):
        cx = i * cw + cw / 2
        ax.text(cx, H - 4, t, ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY)
        ax.text(cx, 3.5, n, ha="center", va="center", fontsize=11.5, color=MUTED, style="italic")
        if i:
            ax.plot([i * cw, i * cw], [2, H - 2], color="#CBD5E1", lw=1)
    top = H - 13
    # (a) concat
    cx = cw / 2
    tokens(ax, cx - 30, top, 5, 3.0, BLUE, gap=0.5)
    tokens(ax, cx + 13, top, 5, 3.0, ORANGE, gap=0.5)
    box(ax, cx - 32, top - 15, 21, 9, "mean", fc=VFILL, ec=BLUE, fs=12)
    box(ax, cx + 11, top - 15, 21, 9, "mean", fc=TFILL, ec=ORANGE, fs=12)
    arrow(ax, cx - 21.5, top - 0.4, cx - 21.5, top - 5.8, color=BLUE, ms=12)
    arrow(ax, cx + 21.5, top - 0.4, cx + 21.5, top - 5.8, color=ORANGE, ms=12)
    box(ax, cx - 17, top - 36, 34, 10, "[ v ; t ] → MLP", fc="white", fs=12)
    arrow(ax, cx - 21.5, top - 15.2, cx - 10, top - 25.8, ms=12)
    arrow(ax, cx + 21.5, top - 15.2, cx + 10, top - 25.8, ms=12)
    # (b) gated
    cx = cw + cw / 2
    tokens(ax, cx - 30, top, 5, 3.0, BLUE, gap=0.5)
    tokens(ax, cx + 13, top, 5, 3.0, ORANGE, gap=0.5)
    box(ax, cx - 33, top - 15, 23, 9, "tanh(W v)", fc=VFILL, ec=BLUE, fs=12)
    box(ax, cx + 10, top - 15, 23, 9, "tanh(W t)", fc=TFILL, ec=ORANGE, fs=12)
    box(ax, cx - 7, top - 15, 14, 9, "z = σ", fc="white", ec=PURPLE, fs=12)
    arrow(ax, cx - 21.5, top - 0.4, cx - 21.5, top - 5.8, color=BLUE, ms=12)
    arrow(ax, cx + 21.5, top - 0.4, cx + 21.5, top - 5.8, color=ORANGE, ms=12)
    box(ax, cx - 26, top - 36, 52, 10, "z · h_v + (1 − z) · h_t", fc=FFILL, ec=PURPLE, fs=12)
    arrow(ax, cx - 21.5, top - 15.2, cx - 14, top - 25.8, ms=12)
    arrow(ax, cx + 21.5, top - 15.2, cx + 14, top - 25.8, ms=12)
    arrow(ax, cx, top - 15.2, cx, top - 25.8, color=PURPLE, ms=12)
    # (c) attention
    cx = 2 * cw + cw / 2
    sx = cx - 31
    for i in range(3):
        ax.add_patch(Rectangle((sx + i * 3.6, top), 3.1, 3.1, fc=PURPLE, ec="white", lw=0.6, zorder=3))
    for i in range(6):
        ax.add_patch(Rectangle((sx + 13 + i * 3.6, top), 3.1, 3.1, fc=BLUE, ec="white", lw=0.6, zorder=3))
    for i in range(6):
        ax.add_patch(Rectangle((sx + 37 + i * 3.6, top), 3.1, 3.1, fc=ORANGE, ec="white", lw=0.6, zorder=3))
    box(ax, cx - 31, top - 15, 62, 9.5, "self-attention ×2 over all tokens", fc=FFILL, ec=PURPLE, fs=12)
    arrow(ax, cx, top - 0.4, cx, top - 5.3, color=PURPLE, ms=12)
    for k, lab in enumerate(["act.", "NSFW", "threat"]):
        bx = cx - 31 + k * 22
        box(ax, bx, top - 36, 18, 10, lab, fc=HFILL, ec=AQUA, fs=12)
        arrow(ax, bx + 9, top - 15.2, bx + 9, top - 25.8, color=AQUA, ms=12)
    save(fig, "poster_fusion.png")


if __name__ == "__main__":
    fig_threats()
    fig_architecture()
    fig_timeline()
    fig_fusion()
