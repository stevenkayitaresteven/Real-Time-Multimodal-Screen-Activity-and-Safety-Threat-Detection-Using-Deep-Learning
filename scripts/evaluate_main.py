"""Evaluate the main checkpoint under text-degradation conditions and with real OCR.

    python scripts/evaluate_main.py --checkpoint runs/main/best.pt

Writes docs/results/test_main*.json and docs/results/main_summary.md.
The real-OCR comparison needs data/synth/test_hires/manifest_ocr.jsonl, produced by
    screenguard generate ... (1280x720) and screenguard ocr-manifest ...
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screenguard.evaluate import evaluate_checkpoint, rows_to_markdown, summary_row  # noqa: E402

CONDITIONS = {"clean": {}, "ocr_noise_5%": {"ocr_noise_p": 0.05}, "ocr_noise_10%": {"ocr_noise_p": 0.10},
              "ocr_noise_20%": {"ocr_noise_p": 0.20}, "no_text": {"drop_text": True}}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/main/best.pt"))
    ap.add_argument("--test", default=str(ROOT / "data/synth/test/manifest.jsonl"))
    ap.add_argument("--hires-ocr", default=str(ROOT / "data/synth/test_hires/manifest_ocr.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "docs/results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, kw in CONDITIONS.items():
        rep = evaluate_checkpoint(a.checkpoint, [a.test], **kw)
        (out / ("test_main.json" if name == "clean" else f"test_main_{name}.json")).write_text(json.dumps(rep, indent=2))
        rows.append(summary_row(f"synthetic test / {name}", rep))
        print(name, rows[-1], flush=True)

    ocr_stats = {}
    if Path(a.hires_ocr).exists():
        recs = [json.loads(l) for l in Path(a.hires_ocr).read_text().splitlines()]
        sims = [difflib.SequenceMatcher(None, norm(r["text"]), norm(r["ocr_text"]), autojunk=False).ratio() for r in recs]
        ocr_stats = {"n": len(recs), "char_similarity_mean": round(statistics.mean(sims), 4),
                     "ocr_ms_mean": round(statistics.mean(r["ocr_ms"] for r in recs), 1),
                     "ocr_ms_p95": round(sorted(r["ocr_ms"] for r in recs)[int(0.95 * len(recs))], 1),
                     "empty_ocr": sum(1 for r in recs if not r["ocr_text"].strip())}
        for field in ("text", "ocr_text"):
            rep = evaluate_checkpoint(a.checkpoint, [a.hires_ocr], text_field=field)
            (out / f"test_hires_{field}.json").write_text(json.dumps(rep, indent=2))
            rows.append(summary_row(f"720p test / {'ground-truth text' if field == 'text' else 'Tesseract OCR'}", rep))
            print(field, rows[-1], flush=True)
        (out / "ocr_stats.json").write_text(json.dumps(ocr_stats, indent=2))

    cols = ["run", "activity_acc", "activity_f1", "threat_acc", "threat_f1", "threat_auroc", "threat_tpr@1%fpr", "threat_ece",
            "model_threat_f1", "rules_threat_f1", "hybrid_threat_f1", "hybrid_fpr", "hybrid_recall"]
    md = rows_to_markdown(rows, cols)
    if ocr_stats:
        md += (f"\n\nTesseract on {ocr_stats['n']} 1280x720 frames: mean character similarity to ground truth "
               f"{ocr_stats['char_similarity_mean']:.3f}, latency mean {ocr_stats['ocr_ms_mean']} ms "
               f"(p95 {ocr_stats['ocr_ms_p95']} ms), empty outputs {ocr_stats['empty_ocr']}.\n")
    (out / "main_summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
