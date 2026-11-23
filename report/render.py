"""Build DOCX -> PDF with a correct table of contents (iterates until page numbers are stable)."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"
DOCX = REP / "KDU_Project_Report_ScreenGuard.docx"
PDF = REP / "KDU_Project_Report_ScreenGuard.pdf"
TOC = REP / "assets/toc_pages.json"
ASSETS = REP / "assets"


def prepare_assets():
    """Image sizes for layout, and the KDU logo taken from the official template (kept out of git)."""
    import sys

    from PIL import Image

    ASSETS.mkdir(exist_ok=True)
    logo = ASSETS / "tpl-000.png"
    if not logo.exists():
        tpl = next(iter(sys.argv[1:]), None)
        if not tpl:
            raise SystemExit("Pass the template PDF once to extract its logo: python report/render.py <template.pdf>")
        subprocess.run(["pdfimages", "-png", "-f", "1", "-l", "1", tpl, str(ASSETS / "tpl")], check=True)
    sizes = {p.name: Image.open(p).size for p in sorted((REP / "figures").glob("*.png"))}
    (ASSETS / "sizes.json").write_text(json.dumps(sizes, indent=1))


def build_and_convert():
    subprocess.run(["node", str(REP / "build_report.js")], check=True, cwd=ROOT)
    subprocess.run(["zip", "-dq", str(DOCX), "*/"], cwd=ROOT)  # drop folder entries (Word/cloud compatibility)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(REP), str(DOCX)], check=True,
                   capture_output=True, timeout=300)


def page_texts():
    txt = subprocess.run(["pdftotext", "-layout", str(PDF), "-"], capture_output=True, text=True, check=True).stdout
    return txt.split("\f")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def locate(pages, headings):
    toc_page = next(i for i, p in enumerate(pages) if any(norm(l) == "Table of Contents" for l in p.splitlines()))
    found, start = {}, 0
    for h in headings:
        if h == "Table of Contents":
            found[h] = toc_page + 1
            continue
        for i in range(start, len(pages)):
            if i == toc_page:
                continue
            if any(norm(l) == h for l in pages[i].splitlines()):
                found[h] = i + 1
                start = i
                break
        else:
            print("heading not found:", h)
    return found


prepare_assets()
prev = None
for it in range(4):
    build_and_convert()
    pages = page_texts()
    headings = json.loads((REP / "assets/headings.json").read_text())
    found = locate(pages, headings)
    TOC.write_text(json.dumps(found, indent=1))
    print(f"pass {it + 1}: {len(pages) - 1} pages, {len(found)}/{len(headings)} headings located")
    if found == prev:
        break
    prev = found
print(json.dumps(found, indent=1))
