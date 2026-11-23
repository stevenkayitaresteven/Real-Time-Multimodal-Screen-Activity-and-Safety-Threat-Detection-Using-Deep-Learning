"""Render docs/results/ablations.json as markdown tables (printed to stdout)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
abl = json.loads((ROOT / "docs/results/ablations.json").read_text())


def f(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


print("| Variant | Params | Activity acc | Threat F1 | Threat AUROC | Activity acc (real OCR) | Threat F1 (real OCR) "
      "| Threat F1 (10 % OCR noise) | Activity acc (no text) | Threat F1 (no text) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, e in abl.items():
    c = e["conditions"]
    print(f"| {name} | {e['n_params'] / 1e6:.2f} M | {f(c['clean'].get('activity_acc'))} | {f(c['clean'].get('threat_f1'))} "
          f"| {f(c['clean'].get('threat_auroc'))} | {f(c['real_ocr'].get('activity_acc'))} | {f(c['real_ocr'].get('threat_f1'))} "
          f"| {f(c['ocr_noise_10%'].get('threat_f1'))} | {f(c['no_text'].get('activity_acc'))} | {f(c['no_text'].get('threat_f1'))} |")
if len(sys.argv) > 1:
    print()
    print("| Variant | Rules-only threat F1 | Hybrid threat F1 | Hybrid FPR |")
    print("|---|---|---|---|")
    for name, e in abl.items():
        c = e["conditions"]["clean"]
        print(f"| {name} | {f(c.get('rules_threat_f1'))} | {f(c.get('hybrid_threat_f1'))} | {f(c.get('hybrid_fpr'))} |")
