"""Fill the KDU A1 poster template with the ScreenGuard project (python-pptx).

    python report/poster/build_poster.py [template.pptx]

The template (with the university logos) and personal details stay out of git: the template is copied once to
report/assets/poster_template.pptx, and names come from report/assets/personal.json (placeholders otherwise).
Output: report/poster/KDU_Poster_ScreenGuard.pptx
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "report/poster"
FIGS = HERE / "figures"
ASSETS = ROOT / "report/assets"
TEMPLATE = ASSETS / "poster_template.pptx"
OUT = HERE / "KDU_Poster_ScreenGuard.pptx"

PERSON = {"leader": "Group Leader", "course": "Course", "department": "Department", "professor": "Course Professor",
          "category": "Innovative Solutions: Business, Industry and Social Impact"}
if (ASSETS / "personal.json").exists():
    PERSON.update(json.loads((ASSETS / "personal.json").read_text()))

TITLE = ["Real-Time Multimodal Screen Activity and", "Safety Threat Detection Using Deep Learning"]

NAVY, BLUE, RED, INK, MUTED, LINE = "0B2240", "005FAD", "D92B3A", "1E293B", "475569", "CBD5E1"
ORANGE, GREEN = "E0662B", "16996B"
FONT = "Times New Roman"


def rgb(h):
    return RGBColor.from_string(h)


# --------------------------------------------------------------------------- text helpers
def runs_of(text, color=INK, bold_color=NAVY):
    """'**bold** plain *italic*' -> list of (text, bold, italic, color)."""
    out = []
    for part in re.split(r"(\*\*.+?\*\*|\*.+?\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            out.append((part[2:-2], True, False, bold_color))
        elif part.startswith("*"):
            out.append((part[1:-1], False, True, color))
        else:
            out.append((part, False, False, color))
    return out


def set_bullet(p, size, indent_in=0.24, char="•", color=BLUE):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(indent_in)))
    pPr.set("indent", str(-Inches(indent_in)))
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", color)
    etree.SubElement(pPr, qn("a:buSzPct")).set("val", "100000")
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(pPr, qn("a:buChar")).set("char", char)


def no_bullet(p):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", "0")
    pPr.set("indent", "0")
    etree.SubElement(pPr, qn("a:buNone"))


def textbox(slide, x, y, w, h, paras, name, anchor=MSO_ANCHOR.TOP):
    """paras: list of dicts {text, size, color, align, bullet, after, before, spacing, bold, italic}."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        size = para.get("size", 18)
        p.alignment = para.get("align", PP_ALIGN.LEFT)
        p.line_spacing = para.get("spacing", 1.0)
        p.space_before = Pt(para.get("before", 0))
        p.space_after = Pt(para.get("after", 0))
        if para.get("bullet"):
            set_bullet(p, size)
        else:
            no_bullet(p)
        for text, bold, italic, color in runs_of(para["text"], para.get("color", INK), para.get("bold_color", NAVY)):
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = para.get("font", FONT)
            f.size = Pt(size)
            f.bold = bold or para.get("bold", False)
            f.italic = italic or para.get("italic", False)
            f.color.rgb = rgb(color)
    return tb


def drop_style(shape):
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def tile(slide, x, y, w, h, value, label, name, vsize=34, lsize=13, vcolor=NAVY):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.name = name
    drop_style(s)
    s.adjustments[0] = 0.12
    s.fill.solid()
    s.fill.fore_color.rgb = rgb("FFFFFF")
    s.line.color.rgb = rgb(LINE)
    s.line.width = Pt(1)
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, (txt, size, bold, color) in enumerate([(value, vsize, True, vcolor), (label, lsize, False, MUTED)]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.line_spacing = 0.92
        no_bullet(p)
        r = p.add_run()
        r.text = txt
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = False
        r.font.color.rgb = rgb(color)
    return s


def icon(slide, x, y, d, png, name, fill=BLUE):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.name = name + " circle"
    drop_style(c)
    c.fill.solid()
    c.fill.fore_color.rgb = rgb(fill)
    c.line.fill.background()
    pad = d * 0.22
    pic = slide.shapes.add_picture(str(FIGS / png), Inches(x + pad), Inches(y + pad), Inches(d - 2 * pad), Inches(d - 2 * pad))
    pic.name = name + " icon"
    return c


def picture(slide, png, x, y, w, name, descr):
    pic = slide.shapes.add_picture(str(FIGS / png), Inches(x), Inches(y), width=Inches(w))
    pic.name = name
    pic._element.nvPicPr.cNvPr.set("descr", descr)
    return pic


# --------------------------------------------------------------------------- chart helpers
def style_text(font, size, color=MUTED, bold=False):
    font.name = FONT
    font.size = Pt(size)
    font.bold = bold
    font.italic = False
    font.color.rgb = rgb(color)


def chart_frame(chart, title, size=14):
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    style_text(chart.chart_title.text_frame.paragraphs[0].runs[0].font, size, NAVY, True)
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    style_text(chart.legend.font, 12, INK)
    style_text(chart.font, 12)
    va, ca = chart.value_axis, chart.category_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E2E8F0")
    va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    ca.format.line.color.rgb = rgb(LINE)
    style_text(va.tick_labels.font, 12)
    style_text(ca.tick_labels.font, 12, INK)
    va.tick_labels.number_format = "0.0"
    va.tick_labels.number_format_is_linked = False


def ablation_chart(slide, x, y, w, h):
    abl = json.loads((ROOT / "docs/results/ablations.json").read_text())
    rows = [("Image only", "image_only"), ("Text only", "text_only"), ("Concat", "concat_fusion"), ("Gated", "gated_fusion"),
            ("Attention", "attention (main config)")]
    cd = CategoryChartData()
    cd.categories = [r[0] for r in rows]
    cd.add_series("clean text", [round(abl[k]["conditions"]["clean"]["threat_f1"], 2) for _, k in rows])
    cd.add_series("real Tesseract OCR", [round(abl[k]["conditions"]["real_ocr"]["threat_f1"], 2) for _, k in rows])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    gf.name = "Chart ablation"
    ch = gf.chart
    chart_frame(ch, "Ablation: threat macro-F1 (4k screens)")
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0, 1.0
    ch.value_axis.major_unit = 0.2
    plot = ch.plots[0]
    plot.gap_width, plot.overlap = 45, -8
    for s, col in zip(plot.series, (BLUE, ORANGE)):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(col)
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = "0.00", False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    style_text(dl.font, 10.5, INK)
    return gf


def noise_chart(slide, x, y, w, h):
    res = ROOT / "docs/results"
    files = [("0 %", "test_main.json"), ("5 %", "test_main_ocr_noise_5%.json"), ("10 %", "test_main_ocr_noise_10%.json"),
             ("20 %", "test_main_ocr_noise_20%.json")]
    act, thr = [], []
    for _, f in files:
        m = json.loads((res / f).read_text())["model"]
        act.append(round(m["activity"]["accuracy"], 3))
        thr.append(round(m["threat"]["macro_f1"], 3))
    cd = CategoryChartData()
    cd.categories = ["clean"] + [f[0] for f in files[1:]]
    cd.add_series("activity accuracy", act)
    cd.add_series("threat macro-F1", thr)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    gf.name = "Chart OCR noise"
    ch = gf.chart
    chart_frame(ch, "Main model vs. OCR character errors")
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0.8, 1.02
    ch.value_axis.major_unit = 0.05
    ch.value_axis.tick_labels.number_format = "0.00"
    plot = ch.plots[0]
    for s, col, pos in zip(plot.series, (GREEN, RED), (XL_LABEL_POSITION.ABOVE, XL_LABEL_POSITION.BELOW)):
        s.smooth = False
        s.format.line.color.rgb = rgb(col)
        s.format.line.width = Pt(2.75)
        s.marker.style = XL_MARKER_STYLE.CIRCLE
        s.marker.size = 9
        s.marker.format.fill.solid()
        s.marker.format.fill.fore_color.rgb = rgb(col)
        s.marker.format.line.color.rgb = rgb(col)
        for i in range(len(files)):
            lab = s.points[i].data_label
            lab.position = pos
            lab.has_text_frame = False
            dLbl = lab._dLbl
            for tag in ("c:showLegendKey", "c:showVal", "c:showCatName", "c:showSerName", "c:showPercent", "c:showBubbleSize"):
                el = dLbl.find(qn(tag))
                if el is not None:
                    el.set("val", "1" if tag == "c:showVal" else "0")
            numFmt = dLbl.find(qn("c:numFmt"))
            if numFmt is None:
                numFmt = etree.Element(qn("c:numFmt"))
                dLbl.insert(1, numFmt)
            numFmt.set("formatCode", "0.00")
            numFmt.set("sourceLinked", "0")
            style_text(lab.font, 10.5, col, True)
    return gf


# --------------------------------------------------------------------------- build
def edit_header(slide):
    shapes = {s.shape_id: s for s in slide.shapes}
    # title: two centred lines, template font and colour kept
    t = shapes[1041459088]
    p = t.text_frame.paragraphs[0]
    run = p.runs[0]
    run.text = TITLE[0]
    run.font.size = Pt(50)
    br = etree.SubElement(p._p, qn("a:br"))
    br.append(copy.deepcopy(run._r.find(qn("a:rPr"))))
    r2 = copy.deepcopy(run._r)
    r2.find(qn("a:t")).text = TITLE[1]
    p._p.append(r2)
    end = p._p.find(qn("a:endParaRPr"))
    if end is not None:  # keep endParaRPr last
        p._p.remove(end)
        p._p.append(end)
    p._p.find(qn("a:pPr")).find(qn("a:lnSpc")).find(qn("a:spcPts")).set("val", "5700")
    t.top = Inches(3.56)
    # student / class / department / supervisor, as in the template, on two lines
    n = shapes[143567942]
    paras = n.text_frame.paragraphs
    for extra in paras[2:]:
        n.text_frame._txBody.remove(extra._p)
    students = PERSON.get("poster_students") or [PERSON["leader"]]
    lines = [f"{', '.join(students)}   |   {PERSON['course']}   |   Department of {PERSON['department']}",
             f"Supervisor: {PERSON['professor']}   |   {PERSON['category']}"]
    for p, line in zip(n.text_frame.paragraphs, lines):
        first = p.runs[0]
        for r in p.runs[1:]:
            p._p.remove(r._r)
        first.text = line
        first.font.size = Pt(26)
        end = p._p.find(qn("a:endParaRPr"))
        if end is not None:
            end.set("sz", "2600")
    n.top = Inches(5.28)


def build(template: Path):
    prs = Presentation(str(template))
    slide = prs.slides[0]
    edit_header(slide)
    # remove the red instruction placeholders inside the six cards
    for s in list(slide.shapes):
        if s.shape_id in (1966145483, 67382122, 1140325533, 144558003, 1691460614, 663585207):
            s._element.getparent().remove(s._element)

    B = dict(size=20, bullet=True, after=13, spacing=1.0)

    # ---- Background / Research Problem --------------------------------------------------------
    textbox(slide, 1.98, 7.84, 8.52, 3.4, [
        dict(B, text="**Threats arrive through the screen:** fake log-in pages, prize scams, malicious downloads and "
                     "“paste this command” lures."),
        dict(B, text="**Today’s tools see one signal:** blocklists miss new look-alike domains, NSFW filters ignore text, "
                     "and antivirus scans files, not screens."),
        dict(B, text="**Cloud analysis is a privacy risk:** screens show passwords and chats."),
        dict(B, text="**Research question:** can one small on-device network read pixels *and* screen text to flag "
                     "activity, explicit content and cyber threats in real time?"),
    ], "Background text")
    picture(slide, "poster_threats.png", 1.98, 11.28, 8.52, "Figure threat screens",
            "Four synthetic screens: phishing, scam, malware download and suspicious command")

    # ---- Objectives, Gap, and Contribution ----------------------------------------------------
    rows = [("icon_bullseye.png", "Objectives",
             "From one frame, recognise 9 screen activities, 3 explicit-content levels and 4 threat types; run in near "
             "real time on a CPU-only laptop or phone; explain every alert."),
            ("icon_puzzle.png", "Gap",
             "Screenshots, phishing text and NSFW images are studied separately, often with large cloud models. Few "
             "systems fuse vision with OCR text, survive missing text, or run on-device."),
            ("icon_bulb.png", "Contribution",
             "**ScreenGuard:** a 1.28 M-parameter vision + text multi-task network, 45 explainable rules, a real-time "
             "engine, and a synthetic screen generator for privacy-safe training.")]
    for k, (png, head, body) in enumerate(rows):
        y = 7.86 + k * 1.74
        icon(slide, 12.55, y + 0.05, 0.78, png, f"{head}")
        textbox(slide, 13.55, y, 7.72, 1.7, [dict(text=f"**{head}**", size=22, after=1),
                                             dict(text=body, size=19, spacing=0.97)], f"{head} text")

    # ---- Methods / Approach -------------------------------------------------------------------
    picture(slide, "poster_architecture.png", 1.90, 15.60, 8.95, "Figure architecture",
            "Real-time pipeline and the ScreenThreatNet multimodal multi-task network")
    textbox(slide, 1.98, 20.42, 8.8, 0.62, [
        dict(text="**Training:** 16,000 synthetic screens · masked multi-task loss with learned uncertainty weights · "
                  "AdamW, 6 epochs (29 min on a 4-core CPU) · temperature calibration · ONNX int8 export.", size=15.5,
             spacing=0.95)], "Methods text")

    # ---- Results / Key Findings ---------------------------------------------------------------
    tiles = [("0.996", "threat F1, real OCR\n(model + rules)"), ("1.000", "activity accuracy,\nreal OCR"),
             ("≈ 7 ms", "model time per frame,\n1 CPU thread"), ("0.5 %", "false-positive rate,\nreal OCR")]
    tw, gap = 2.16, 0.13
    for k, (v, lab) in enumerate(tiles):
        tile(slide, 12.24 + k * (tw + gap), 15.58, tw, 1.2, v, lab, f"Result tile {k + 1}")
    ablation_chart(slide, 12.17, 16.9, 4.55, 3.2)
    noise_chart(slide, 16.78, 16.9, 4.55, 3.2)
    textbox(slide, 12.26, 20.2, 9.0, 0.85, [
        dict(text="**Fusion matters:** pixels alone miss threats (F1 0.33); text alone fails when OCR returns nothing. "
                  "OCR-aware fine-tuning lifts real-OCR threat F1 from 0.971 to 0.992.", size=15.5, spacing=0.95)],
        "Results text")

    # ---- Validation and Impact ----------------------------------------------------------------
    textbox(slide, 2.05, 23.42, 8.8, 0.62, [
        dict(text="**Live test:** a scripted 65-frame session (13 screens, 5 threats) with real Tesseract OCR and the "
                  "int8 ONNX model on a 4-core laptop CPU.", size=16.5, spacing=0.95)], "Validation text")
    picture(slide, "poster_timeline.png", 2.0, 24.08, 8.8, "Figure session timeline",
            "Smoothed threat risk over the scripted session; alerts fire on all five threat screens")
    vt = [("5 / 5", "threats alerted\nwithin 2 frames"), ("0", "false alerts on\n8 benign screens"),
          ("80 %", "frames skipped by\nchange detection"), ("392 ms", "per analysed frame\n(96 % is OCR)")]
    tw2 = (8.8 - 3 * 0.12) / 4
    for k, (v, lab) in enumerate(vt):
        tile(slide, 2.03 + k * (tw2 + 0.12), 26.92, tw2, 1.08, v, lab, f"Validation tile {k + 1}", vsize=28, lsize=12.5,
             vcolor=RED if k == 0 else NAVY)
    textbox(slide, 2.05, 28.13, 8.8, 0.4, [dict(text="**Impact**", size=21)], "Impact heading")
    impact = [("icon_lock.png", "**Private by design:** runs fully on-device; no frames saved, no screen text in logs, "
                                "personal data redacted."),
              ("icon_chip.png", "**Edge-ready:** a 5 MB model; ONNX fp32/int8 runs at 4–16 ms per frame on one CPU thread "
                                "across four backbones."),
              ("icon_users.png", "**Social impact:** shields students, older adults and children from scams and harmful "
                                 "content, for families, schools and small businesses.")]
    for k, (png, txt) in enumerate(impact):
        y = 28.6 + k * 0.84
        icon(slide, 2.05, y + 0.06, 0.56, png, f"Impact {k + 1}", fill=NAVY)
        textbox(slide, 2.78, y, 8.1, 0.8, [dict(text=txt, size=16, spacing=0.95)], f"Impact {k + 1} text")

    # ---- Conclusion and Future Work -----------------------------------------------------------
    textbox(slide, 12.4, 23.42, 8.95, 0.4, [dict(text="**Three fusion strategies compared** (threat macro-F1)", size=17)],
            "Fusion heading")
    picture(slide, "poster_fusion.png", 12.36, 23.86, 8.95, "Figure fusion strategies",
            "Concat, gated and attention fusion of visual and text tokens")
    C = dict(size=17, bullet=True, after=4, spacing=0.95)
    textbox(slide, 12.4, 26.25, 8.9, 4.4, [
        dict(text="**Conclusion**", size=21, after=2),
        dict(C, text="Reading pixels and words together works: the fused model reaches 0.99 threat F1, while pixels "
                     "alone reach only 0.33."),
        dict(C, text="A small rule layer gives every alert a human-readable reason and cuts false alarms; OCR-aware "
                     "fine-tuning closes the real-OCR gap."),
        dict(C, text="The network is not the bottleneck: OCR takes 96 % of the time per frame, and change detection "
                     "skips 80 % of frames.", after=10),
        dict(text="**Future work**", size=21, after=2),
        dict(C, text="Validate on real screenshots (Rico, Phishpedia); all results so far are synthetic."),
        dict(C, text="Train and audit the explicit-content head on an approved dataset (built, not yet trained)."),
        dict(C, text="Distil a CLIP/SigLIP teacher into the TinyCNN, run OCR only on changed regions, and add Korean "
                     "and Kinyarwanda scam text."),
    ], "Conclusion text")
    textbox(slide, 12.4, 30.55, 8.9, 0.62, [
        dict(text="Key references: Howard et al., MobileNetV3, ICCV 2019 · Dosovitskiy et al., ViT, ICLR 2021 · "
                  "Arevalo et al., Gated Multimodal Units, ICLR-W 2017 · Kendall et al., uncertainty-weighted multi-task "
                  "loss, CVPR 2018.", size=11.5, color=MUTED, italic=True, spacing=0.95)], "References")
    prs.save(str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        shutil.copyfile(sys.argv[1], TEMPLATE)
    if not TEMPLATE.exists():
        raise SystemExit("Pass the KDU poster template once: python report/poster/build_poster.py <template.pptx>")
    build(TEMPLATE)
