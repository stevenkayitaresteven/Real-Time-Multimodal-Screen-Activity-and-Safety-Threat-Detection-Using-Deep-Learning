# Project report builder

Builds the KDU Global Research Center 2026 Showcase project report (DOCX and PDF) from the
experiment results in `docs/results/` and the figures in `report/figures/`.

```bash
python report/make_report_figures.py          # architecture diagrams + result figures (needs data/ and runs/)
python report/render.py <template.pdf>        # DOCX -> PDF with a computed table of contents
```

Requirements: Node.js with the `docx` package, LibreOffice (`soffice`), Poppler (`pdfimages`, `pdftotext`),
Python with Pillow and Matplotlib.

Personal details (names, student IDs, phone number, course, professor) are read from
`report/assets/personal.json`, which is git-ignored, as are the generated `.docx`/`.pdf` files and the
logo extracted from the official template. Without that file the report is built with placeholders:

```json
{"leader": "...", "leaderId": "...", "phone": "...", "member": "...", "memberId": "...",
 "department": "...", "course": "...", "professor": "...", "submission": "25 November 2026"}
```
