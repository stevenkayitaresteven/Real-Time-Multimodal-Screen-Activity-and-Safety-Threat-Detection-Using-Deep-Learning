// Builds the KDU 2026 Showcase project report (DOCX) following the official template.
// Run from the repository root:  node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, ShadingType, Header, Footer, PageNumber, HeadingLevel, TabStopType, LevelFormat, VerticalAlign, Tab, LineRuleType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIGDIR = path.join(ROOT, "report/figures");
const SIZES = JSON.parse(fs.readFileSync(path.join(ROOT, "report/assets/sizes.json")));
const TOC_FILE = path.join(ROOT, "report/assets/toc_pages.json");
const TOC_PAGES = fs.existsSync(TOC_FILE) ? JSON.parse(fs.readFileSync(TOC_FILE)) : {};

// Personal details live in report/assets/personal.json (git-ignored) so they are never committed.
const PERSON_FILE = path.join(ROOT, "report/assets/personal.json");
const PERSON = Object.assign({ leader: "Group Leader", leaderId: "0000000", phone: "+00 00-0000-0000", member: "Team Member",
  memberId: "0000000", department: "Department", course: "Course", professor: "Course Professor", submission: "Submission date" },
  fs.existsSync(PERSON_FILE) ? JSON.parse(fs.readFileSync(PERSON_FILE)) : {});

const FONT = "Calibri";
const MONO = "Consolas";
const C = { navy: "163A63", blue: "0B4F9C", red: "E4002B", label: "EAF2FA", grey: "D9D9D9", muted: "595959", border: "BFC9D6", alt: "F4F7FB" };
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1134, CONTENT = PAGE_W - 2 * MARGIN; // A4, 2 cm margins
const CONTENT_PX = 640;

// ------------------------------------------------------------------ references
const REFS = {
  ic3: "Federal Bureau of Investigation, Internet Crime Complaint Center, “Internet Crime Report 2023,” Washington, DC, USA, 2024. [Online]. Available: https://www.ic3.gov/AnnualReport/Reports/2023_IC3Report.pdf",
  apwg: "Anti-Phishing Working Group, “Phishing Activity Trends Reports.” [Online]. Available: https://apwg.org/trendsreports/",
  safebrowsing: "Google, “Safe Browsing.” [Online]. Available: https://safebrowsing.google.com/",
  apple_cs: "Apple Inc., “About Communication Safety on your child’s Apple device,” Apple Support. [Online]. Available: https://support.apple.com/",
  mnv2: "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 4510–4520.",
  mnv3: "A. Howard et al., “Searching for MobileNetV3,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 1314–1324.",
  transformer: "A. Vaswani et al., “Attention is all you need,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, 2017, pp. 5998–6008.",
  vit: "A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.",
  tesseract: "R. Smith, “An overview of the Tesseract OCR engine,” in Proc. 9th Int. Conf. Document Anal. Recognit. (ICDAR), 2007, pp. 629–633.",
  rico: "B. Deka et al., “Rico: A mobile app dataset for building data-driven design applications,” in Proc. 30th Annu. ACM Symp. User Interface Softw. Technol. (UIST), 2017, pp. 845–854.",
  screen2words: "B. Wang, G. Li, X. Zhou, Z. Chen, T. Grossman, and Y. Li, “Screen2Words: Automatic mobile UI summarization with multimodal learning,” in Proc. 34th Annu. ACM Symp. User Interface Softw. Technol. (UIST), 2021, pp. 498–510.",
  webui: "J. Wu, S. Wang, S. Shen, Y.-H. Peng, J. Nichols, and J. P. Bigham, “WebUI: A dataset for enhancing visual UI understanding with web semantics,” in Proc. CHI Conf. Human Factors Comput. Syst., 2023.",
  visualphishnet: "S. Abdelnabi, K. Krombholz, and M. Fritz, “VisualPhishNet: Zero-day phishing website detection by visual similarity,” in Proc. ACM SIGSAC Conf. Comput. Commun. Secur. (CCS), 2020, pp. 1681–1698.",
  phishpedia: "Y. Lin et al., “Phishpedia: A hybrid deep learning based approach to visually identify phishing webpages,” in Proc. 30th USENIX Security Symp., 2021, pp. 3793–3810.",
  phishintention: "R. Liu, Y. Lin, X. Yang, S. H. Ng, D. M. Divakaran, and J. S. Dong, “Inferring phishing intention via webpage appearance and dynamics: A deep vision based approach,” in Proc. 31st USENIX Security Symp., 2022.",
  smsspam: "T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami, “Contributions to the study of SMS spam filtering: New collection and results,” in Proc. 11th ACM Symp. Document Eng. (DocEng), 2011, pp. 259–262.",
  zhang_charcnn: "X. Zhang, J. Zhao, and Y. LeCun, “Character-level convolutional networks for text classification,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 28, 2015, pp. 649–657.",
  kim_cnn: "Y. Kim, “Convolutional neural networks for sentence classification,” in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP), 2014, pp. 1746–1751.",
  canine: "J. H. Clark, D. Garrette, I. Turc, and J. Wieting, “CANINE: Pre-training an efficient tokenization-free encoder for language representation,” Trans. Assoc. Comput. Linguistics, vol. 10, pp. 73–91, 2022.",
  damerau: "F. J. Damerau, “A technique for computer detection and correction of spelling errors,” Commun. ACM, vol. 7, no. 3, pp. 171–176, 1964.",
  attack: "B. E. Strom, A. Applebaum, D. P. Miller, K. C. Nickels, A. G. Pennington, and C. B. Thomas, “MITRE ATT&CK: Design and philosophy,” The MITRE Corporation, McLean, VA, USA, Tech. Rep., 2018.",
  sigma: "SigmaHQ, “Sigma: Generic signature format for SIEM systems,” GitHub repository. [Online]. Available: https://github.com/SigmaHQ/sigma",
  se: "J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 7132–7141.",
  efficientnet: "M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. 36th Int. Conf. Mach. Learn. (ICML), 2019, pp. 6105–6114.",
  gmu: "J. Arevalo, T. Solorio, M. Montes-y-Gómez, and F. A. González, “Gated multimodal units for information fusion,” in Proc. Int. Conf. Learn. Represent. (ICLR) Workshop Track, 2017.",
  wang_hard: "W. Wang, D. Tran, and M. Feiszli, “What makes training multi-modal classification networks hard?,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 12695–12705.",
  moddrop: "N. Neverova, C. Wolf, G. Taylor, and F. Nebout, “ModDrop: Adaptive multi-modal gesture recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 38, no. 8, pp. 1692–1706, 2016.",
  kendall: "A. Kendall, Y. Gal, and R. Cipolla, “Multi-task learning using uncertainty to weigh losses for scene geometry and semantics,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 7482–7491.",
  guo: "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. 34th Int. Conf. Mach. Learn. (ICML), 2017, pp. 1321–1330.",
  gradcam: "R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 618–626.",
  jacob: "B. Jacob et al., “Quantization and training of neural networks for efficient integer-arithmetic-only inference,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 2704–2713.",
  open_nsfw: "Yahoo, “open_nsfw: Not suitable for work (NSFW) classification using deep neural network Caffe models,” GitHub repository, 2016. [Online]. Available: https://github.com/yahoo/open_nsfw",
  pytorch: "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 32, 2019, pp. 8024–8035.",
  onnxruntime: "ONNX Runtime developers, “ONNX Runtime: Cross-platform, high performance ML inferencing and training accelerator.” [Online]. Available: https://onnxruntime.ai",
  adamw: "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.",
  labelsmooth: "C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the Inception architecture for computer vision,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 2818–2826.",
  dhash: "N. Krawetz, “Kind of like that,” The Hacker Factor Blog, Jan. 2013. [Online]. Available: https://www.hackerfactor.com/blog/",
  mlkit: "Google, “ML Kit: Text recognition v2,” Google for Developers. [Online]. Available: https://developers.google.com/ml-kit",
};
const citeOrder = [];
function citeNum(key) {
  if (!REFS[key]) throw new Error("unknown reference " + key);
  if (!citeOrder.includes(key)) citeOrder.push(key);
  return citeOrder.indexOf(key) + 1;
}
function citeText(tok) {
  const nums = tok.slice(1, -1).split(",").map((k) => citeNum(k.trim().replace(/^@/, ""))).sort((a, b) => a - b);
  const parts = [];
  for (let i = 0; i < nums.length; i++) {
    let j = i;
    while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
    if (j - i >= 2) { parts.push(`[${nums[i]}]–[${nums[j]}]`); i = j; } else parts.push(`[${nums[i]}]`);
  }
  return parts.join(", ");
}

// ------------------------------------------------------------------ figure / table registry
const FIG_ORDER = ["samples", "pipeline", "model", "vision", "text", "fusion", "decision", "console", "confusion", "ocrgap",
  "ablations", "robustness", "latency", "gradcam", "timeline", "training"];
const TAB_ORDER = ["existing", "data", "config", "requirements", "roles", "main", "finetune", "fusiondec", "ablation", "latency",
  "validation", "risks", "hyper", "rules", "perclass", "airecord"];
const F = (k) => { const i = FIG_ORDER.indexOf(k); if (i < 0) throw new Error("fig " + k); return `Figure ${i + 1}`; };
const TB = (k) => { const i = TAB_ORDER.indexOf(k); if (i < 0) throw new Error("tab " + k); return `Table ${i + 1}`; };

// ------------------------------------------------------------------ inline markup: **bold**, *italic*, «code», [@ref,@ref]
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|«[^»]+»|\[@[^\]]+\]|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  const mk = (t, extra = {}) => new TextRun({ text: t, font: FONT, ...base, ...extra });
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(mk(text.slice(last, m.index)));
    const t = m[0];
    if (t.startsWith("**")) out.push(mk(t.slice(2, -2), { bold: true }));
    else if (t.startsWith("«")) out.push(mk(t.slice(1, -1), { font: MONO, size: (base.size || 22) - 3 }));
    else if (t.startsWith("[@")) out.push(mk(citeText(t)));
    else if (t.startsWith("_{")) out.push(mk(t.slice(2, -1), { subScript: true }));
    else if (t.startsWith("^{")) out.push(mk(t.slice(2, -1), { superScript: true }));
    else out.push(mk(t.slice(1, -1), { italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(mk(text.slice(last)));
  return out;
}

// ------------------------------------------------------------------ block helpers
const headings = [];
function H1(text, opts = {}) {
  headings.push({ level: 1, text });
  return new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: opts.pageBreak !== false, children: [new TextRun({ text, font: FONT })] });
}
function H2(text) { headings.push({ level: 2, text }); return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: FONT })] }); }
function H3(text) { return new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text, font: FONT })] }); }
function P(text, o = {}) {
  return new Paragraph({
    alignment: o.align || AlignmentType.JUSTIFIED, keepNext: o.keepNext, keepLines: o.keepLines,
    spacing: { before: o.before || 0, after: o.after === undefined ? 120 : o.after, line: o.line || 276, lineRule: LineRuleType.AUTO },
    indent: o.indent, children: runs(text, o.run || {}),
  });
}
let listInstance = 0;
function bullets(items, o = {}) {
  return items.map((t, i) => new Paragraph({
    numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: i === items.length - 1 ? (o.after === undefined ? 120 : o.after) : 50, line: 264, lineRule: LineRuleType.AUTO },
    children: runs(t, o.run || {}),
  }));
}
function numbered(items) {
  listInstance += 1;
  const inst = listInstance;
  return items.map((t, i) => new Paragraph({
    numbering: { reference: "numbers", level: 0, instance: inst }, alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: i === items.length - 1 ? 120 : 50, line: 264, lineRule: LineRuleType.AUTO }, children: runs(t),
  }));
}
function EQ(text, n) {
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(CONTENT / 2) }, { type: TabStopType.RIGHT, position: CONTENT }],
    spacing: { before: 60, after: 120, line: 276, lineRule: LineRuleType.AUTO },
    children: [new TextRun({ children: [new Tab()], font: FONT }), ...runs(text, { italics: true, font: "Cambria Math" }),
      new TextRun({ children: [new Tab(), `(${n})`], font: FONT })],
  });
}
function FIG(key, file, caption, frac = 1.0) {
  const [w, h] = SIZES[file];
  const width = Math.round(CONTENT_PX * frac);
  const height = Math.round(width * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60, line: 240, lineRule: LineRuleType.AUTO },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(FIGDIR, file)), transformation: { width, height },
        altText: { title: F(key), description: caption.replace(/[*«»]/g, ""), name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 200, line: 240, lineRule: LineRuleType.AUTO },
      children: [new TextRun({ text: `${F(key)}. `, bold: true, size: 18, font: FONT }), ...runs(caption, { size: 18 })] }),
  ];
}
const thin = { style: BorderStyle.SINGLE, size: 4, color: C.border };
const BORDERS = { top: thin, bottom: thin, left: thin, right: thin, insideHorizontal: thin, insideVertical: thin };
function cellParas(txt, base, align) {
  return String(txt).split("\n").map((line) => new Paragraph({ alignment: align, spacing: { before: 0, after: 0, line: 240, lineRule: LineRuleType.AUTO }, children: runs(line, base) }));
}
function TABLE(key, caption, headers, rows, widths, o = {}) {
  const size = o.size || 18;
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, i, head, r) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: C.navy, type: ShadingType.CLEAR, color: "auto" } : (r % 2 === 1 ? { fill: C.alt, type: ShadingType.CLEAR, color: "auto" } : undefined),
    margins: { top: 45, bottom: 45, left: 85, right: 85 }, verticalAlign: VerticalAlign.CENTER,
    children: cellParas(txt, head ? { size, bold: true, color: "FFFFFF" } : { size },
      head ? AlignmentType.CENTER : ((o.align && o.align[i]) || AlignmentType.LEFT)),
  });
  const table = new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER, borders: BORDERS,
    rows: [new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((h, i) => cell(h, i, true, 0)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, ri)) }))],
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80, line: 240, lineRule: LineRuleType.AUTO },
      children: [new TextRun({ text: `${TB(key)}. `, bold: true, size: 18, font: FONT }), ...runs(caption, { size: 18 })] }),
    table,
    new Paragraph({ spacing: { before: 0, after: 120 }, children: [] }),
  ];
}
const CENTER = AlignmentType.CENTER, RIGHT = AlignmentType.RIGHT;

// ------------------------------------------------------------------ cover page
function coverPage() {
  const logo = fs.readFileSync(path.join(ROOT, "report/assets/tpl-000.png"));
  const lw = 440, lh = Math.round(440 * 364 / 1082);
  const labelCell = (t, fill = C.label) => new TableCell({
    width: { size: 2270, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR, color: "auto" },
    margins: { top: 90, bottom: 90, left: 110, right: 110 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: C.navy, size: 22, font: FONT })] })],
  });
  const valueCell = (children, fill) => new TableCell({
    width: { size: 7368, type: WidthType.DXA }, shading: fill ? { fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 90, bottom: 90, left: 110, right: 110 }, verticalAlign: VerticalAlign.CENTER, children,
  });
  const v = (t) => [new Paragraph({ children: [new TextRun({ text: t, size: 22, font: FONT })] })];
  const blk = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
  const category = [
    new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: "☒  ", font: "Segoe UI Symbol", size: 22 }),
      new TextRun({ text: "Innovative Solutions:", bold: true, underline: {}, size: 22, font: FONT }),
      new TextRun({ text: " Business, Industry and Social Impact", size: 22, font: FONT })] }),
    new Paragraph({ children: [new TextRun({ text: "☐  ", font: "Segoe UI Symbol", size: 22, color: "8A8F98" }),
      new TextRun({ text: "Research and Discovery:", bold: true, underline: {}, size: 22, color: "8A8F98", font: FONT }),
      new TextRun({ text: " Evidence, Analysis and New Knowledge", size: 22, color: "8A8F98", font: FONT })] }),
  ];
  const rows = [
    new TableRow({ children: [labelCell("Project category"), valueCell(category)] }),
    new TableRow({ children: [labelCell("Department"), valueCell(v(PERSON.department))] }),
    new TableRow({ children: [labelCell("Course and section"), valueCell(v(PERSON.course))] }),
    new TableRow({ children: [labelCell("Course professor"), valueCell(v(PERSON.professor))] }),
    new TableRow({ children: [labelCell("Group leader"), valueCell(v(`${PERSON.leader} (Student ID: ${PERSON.leaderId})`))] }),
    new TableRow({ children: [labelCell("Group leader\nMobile No."), valueCell(v(PERSON.phone))] }),
    new TableRow({ children: [labelCell("Submission date", C.grey), valueCell(v(PERSON.submission), C.grey)] }),
  ];
  // multi-line label support
  rows[5] = new TableRow({ children: [new TableCell({ width: { size: 2270, type: WidthType.DXA }, shading: { fill: C.label, type: ShadingType.CLEAR, color: "auto" },
    margins: { top: 90, bottom: 90, left: 110, right: 110 }, verticalAlign: VerticalAlign.CENTER,
    children: ["Group leader", "Mobile No."].map((t) => new Paragraph({ children: [new TextRun({ text: t, bold: true, color: C.navy, size: 22, font: FONT })] })) }),
  valueCell(v(PERSON.phone))] });
  return [
    new Paragraph({ alignment: CENTER, spacing: { before: 400, after: 200, line: 240, lineRule: LineRuleType.AUTO }, children: [new ImageRun({ type: "png", data: logo, transformation: { width: lw, height: lh },
      altText: { title: "KDU Global Research Center", description: "Kyungdong University Global Research Center logo", name: "logo" } })] }),
    new Paragraph({ alignment: CENTER, spacing: { after: 160 }, children: [new TextRun({ text: "KDU Global Research Center", bold: true, size: 32, color: C.blue, font: FONT })] }),
    new Paragraph({ alignment: CENTER, spacing: { after: 560 }, children: [new TextRun({ text: "2026 Research and Innovation Showcase - 9 December 2026", bold: true, size: 25, color: "7F7F7F", font: FONT })] }),
    new Paragraph({ alignment: CENTER, spacing: { after: 120 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "5B9BD5", space: 8 } },
      children: [new TextRun({ text: "PROJECT REPORT", bold: true, size: 56, font: FONT })] }),
    new Paragraph({ alignment: CENTER, spacing: { before: 200, after: 520, line: 300, lineRule: LineRuleType.AUTO }, children: [new TextRun({ text: "Real-Time Multimodal Screen Activity and Safety Threat Detection Using Deep Learning", bold: true, size: 34, color: C.red, font: FONT })] }),
    new Table({ width: { size: 9638, type: WidthType.DXA }, columnWidths: [2270, 7368], alignment: CENTER, rows,
      borders: { top: blk, bottom: blk, left: blk, right: blk, insideHorizontal: blk, insideVertical: blk } }),
    new Paragraph({ alignment: CENTER, spacing: { before: 40 }, children: [new TextRun({ text: "Kyungdong University Global  |  Republic of Korea", size: 16, color: "7F7F7F", font: FONT })] }),
  ];
}

// ------------------------------------------------------------------ declaration page
function declarationPage() {
  const lab = (t) => new TableCell({ width: { size: 2800, type: WidthType.DXA }, shading: { fill: C.label, type: ShadingType.CLEAR, color: "auto" },
    margins: { top: 100, bottom: 100, left: 110, right: 110 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, size: 19, color: C.navy, font: FONT })] })] });
  const val = (t) => new TableCell({ width: { size: 6838, type: WidthType.DXA }, margins: { top: 100, bottom: 100, left: 110, right: 110 },
    verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ children: [new TextRun({ text: t, size: 21, font: FONT })] })] });
  const lb = { style: BorderStyle.SINGLE, size: 4, color: "C9D3E0" };
  const confirm = new Table({ width: { size: 9638, type: WidthType.DXA }, columnWidths: [2800, 6838], borders: { top: lb, bottom: lb, left: lb, right: lb, insideHorizontal: lb, insideVertical: lb },
    rows: [["Name", PERSON.leader], ["Student ID", PERSON.leaderId], ["Signature", ""], ["Date", PERSON.submission]].map(([a, b]) => new TableRow({ height: { value: 460, rule: "atLeast" }, children: [lab(a), val(b)] })) });
  const tw = [760, 4900, 1900, 2078];
  const hc = (t, i) => new TableCell({ width: { size: tw[i], type: WidthType.DXA }, shading: { fill: C.navy, type: ShadingType.CLEAR, color: "auto" },
    margins: { top: 70, bottom: 70, left: 90, right: 90 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: CENTER, children: [new TextRun({ text: t, bold: true, color: "FFFFFF", size: 19, font: FONT })] })] });
  const bc = (t, i) => new TableCell({ width: { size: tw[i], type: WidthType.DXA }, margins: { top: 110, bottom: 110, left: 90, right: 90 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: i === 1 ? CENTER : (i === 0 ? CENTER : CENTER), children: [new TextRun({ text: t, size: 21, bold: i === 0, font: FONT })] })] });
  const members = new Table({ width: { size: 9638, type: WidthType.DXA }, columnWidths: tw, borders: { top: lb, bottom: lb, left: lb, right: lb, insideHorizontal: lb, insideVertical: lb },
    rows: [new TableRow({ tableHeader: true, children: ["No.", "Name", "Student ID", "Department"].map(hc) }),
      new TableRow({ children: ["1", `${PERSON.leader} (Group Leader)`, PERSON.leaderId, PERSON.department].map(bc) }),
      new TableRow({ children: ["2", PERSON.member, PERSON.memberId, PERSON.department].map(bc) })] });
  return [
    H1("Authorship and Academic Integrity Declaration"),
    P("By signing below, the Group Leader confirms on behalf of all members that this report is original, properly referenced, ethically conducted, and approved by the course professor. The team accepts responsibility for the accuracy of all evidence, data, analysis, code, visuals, and claims."),
    ...bullets(["All sources, quotations, datasets, images, code, and external contributions are acknowledged.",
      "AI tools were used only as permitted by the course professor and are disclosed below.",
      "No data, findings, citations, references, or participant information were fabricated or misrepresented.",
      "Any required consent, privacy, safety, and partner approvals were obtained before the work was conducted."]),
    H2("Responsible AI Disclosure"),
    P("**Tool:** Claude (Anthropic), a general-purpose AI assistant, used through Claude Code. **Purpose:** assistance with implementing parts of the ScreenGuard software (data generator, model code, training and evaluation scripts, real-time pipeline), running the experiments, producing figures, and drafting and editing this report. **Sections affected:** Sections 1–6, all figures and tables, and the appendices. **Verification:** every quantitative result in this report is produced by scripts in the project repository and stored in machine-readable result files («docs/results/»); the code is covered by 65 automated tests; exported models were checked numerically against the original PyTorch model; and the team reviewed the code and text and checked each reported number against the stored result files before submission. A detailed record is given in Appendix C."),
    H2("Group Leader Confirmation"),
    confirm,
    H2("Team Members"),
    members,
  ];
}

// ------------------------------------------------------------------ abstract + TOC
function abstractPage() {
  return [
    H1("Abstract"),
    P("Most digital risks now reach people through a screen: a fake login page, a virus warning with a “support” phone number, a cracked installer, or a command that a stranger asks them to paste. Existing protections such as browser blocklists, mail filters and parental controls work in silos, and none looks at the screen as a whole. This project asks whether a compact deep learning model can read both the pixels and the text of a screen frame and decide, in near real time on an ordinary CPU, what the user is doing and whether anything on screen is unsafe. We built ScreenGuard, which combines a lightweight convolutional vision encoder, a byte-level Transformer over OCR text and attention-based fusion with three output heads: screen activity (nine classes), explicit content (three classes) and cyber threats (five classes). Because no public dataset labels these tasks jointly, we also developed a generator of labelled desktop and mobile screens. On held-out data, the 1.28-million-parameter model reached 100% activity accuracy and a threat macro-F1 of 0.994. With real Tesseract OCR the threat F1 was 0.971, rising to 0.992 after OCR-aware fine-tuning. Model inference takes about 7 ms per frame on one CPU thread, and change detection skipped 80% of frames in an end-to-end session. Ablations show that multimodal fusion, a shift-invariant byte encoder and modality dropout each make a measurable difference. The explicit-content head still needs an ethically approved dataset, and validation on real user screens is the main next step."),
    new Paragraph({ spacing: { before: 120, after: 240 }, children: [new TextRun({ text: "Keywords: ", bold: true, font: FONT }),
      new TextRun({ text: "multimodal deep learning, screen activity recognition, phishing and scam detection, optical character recognition, edge AI, real-time inference", font: FONT })] }),
    H1("Acknowledgements", { pageBreak: false }),
    P("We thank " + PERSON.professor + " for his teaching and guidance in the Deep Learning course, and the KDU Global Research Center for organising the 2026 Research and Innovation Showcase. This prototype is built on open-source software, and we gratefully acknowledge the developers of PyTorch, ONNX Runtime, Tesseract OCR, Pillow and Matplotlib."),
  ];
}
function tocPage() {
  const entries = [{ level: 1, text: "Authorship and Academic Integrity Declaration" }, { level: 1, text: "Abstract" }, { level: 1, text: "Acknowledgements" },
    ...headings.filter((h) => !["Authorship and Academic Integrity Declaration", "Abstract", "Acknowledgements", "Table of Contents", "Responsible AI Disclosure", "Group Leader Confirmation", "Team Members"].includes(h.text))];
  const out = [H1("Table of Contents")];
  for (const e of entries) {
    const pg = TOC_PAGES[e.text] !== undefined ? String(TOC_PAGES[e.text]) : "0";
    const top = e.level === 1;
    out.push(new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: CONTENT }], indent: top ? undefined : { left: 360 },
      spacing: { before: top ? 110 : 20, after: top ? 30 : 20, line: 240, lineRule: LineRuleType.AUTO },
      children: [new TextRun({ text: e.text, bold: top, size: top ? 21 : 20, color: top ? C.navy : "404040", font: FONT }),
        new TextRun({ children: [new Tab(), pg], bold: top, size: top ? 21 : 20, color: top ? C.navy : "404040", font: FONT })],
    }));
  }
  return out;
}

// ------------------------------------------------------------------ body
const W = (arr) => { const s = arr.reduce((a, b) => a + b, 0); return arr.map((x) => Math.round(x * CONTENT / s)); };

function section1() {
  return [
    H1("1 Project Overview"),
    H2("1.1 Background and Context"),
    P("Laptops and smartphones have become the place where people study, work, bank, shop and keep in touch. That convenience comes with a growing amount of risk that is delivered visually. In 2023 alone, the FBI’s Internet Crime Complaint Center received 880,418 complaints, with reported losses above US$12.5 billion; investment fraud was the costliest category at US$4.57 billion [@ic3]. Phishing campaigns tracked by the Anti-Phishing Working Group remain at persistently high volumes year after year [@apwg]. Behind each of these numbers there is usually a moment when the victim was looking at something on a screen that should have raised suspicion."),
    P("What has changed in recent years is *where* that moment happens. A parcel-delivery scam arrives by SMS, an “account suspended” notice arrives by e-mail, a crypto “double your coins” offer appears in a social-media feed or a fake livestream, and a fake CAPTCHA page may instruct the user to press Win + R and paste a command into Windows. Malware is often installed by the user themselves after a convincing “update your video codec” prompt. These threats cross applications, devices and formats, but all of them become visible on the screen before any damage is done."),
    P("Current protections are effective but fragmented. Browsers check URLs against reputation lists such as Google Safe Browsing [@safebrowsing], mail providers filter messages, antivirus engines scan files, and parental-control tools block categories of websites. Apple’s Communication Safety feature shows that sensitive imagery can be detected entirely on the device [@apple_cs], yet it is limited to specific apps. Each tool sees one channel, and most rely on signatures or blocklists that a new lookalike domain can escape. None of them interprets *what the user is actually seeing* across applications."),
    P("At the same time, deep learning has made this kind of interpretation affordable. Efficient convolutional networks such as MobileNetV2 and MobileNetV3 run comfortably on phone CPUs [@mnv2,@mnv3], Transformers [@transformer] and Vision Transformers [@vit] model long-range relations in text and images, and mature OCR engines [@tesseract] can turn screen content into text. This project explores how these components can be combined into a single on-device system that watches the screen like a careful friend would, and speaks up only when something looks wrong."),
    ...FIG("samples", "synthetic_samples.png", "Examples of the screens the system must understand, rendered by our data generator (§3.1). Each frame is labelled with an activity (left) and a threat class (right); threats include phishing logins, scam pop-ups, malicious downloads and dangerous commands across browsers, chat, e-mail and terminals."),
    H2("1.2 Problem Statement and Research Question"),
    P("The people this project addresses are ordinary laptop and phone users, particularly those with less security experience such as students, older adults and children, together with the families, schools and small organisations that try to protect them without a dedicated security team. Their unmet need is a protection layer that understands screen content in *any* application and warns them about phishing, scams, malicious downloads, dangerous commands and unwanted explicit content at the moment of decision, without uploading their screen to a remote server. Without it, the consequences are financial loss, account takeover, malware infection and harmful exposure for minors, all of which are documented at scale [@ic3]."),
    P("The project is driven by one research question:"),
    P("*Can a compact multimodal deep learning model, using only the pixels of a screen frame and the text extracted from it by OCR, recognise the user’s activity and detect safety threats accurately enough and fast enough to run continuously on a CPU-only personal device?*", { indent: { left: 567, right: 567 }, align: AlignmentType.LEFT }),
    P("We break this question into three testable hypotheses:"),
    ...bullets([
      "**H1 (complementarity).** Fusing visual and textual evidence detects threats and activities better than either modality alone, especially when one modality is degraded.",
      "**H2 (robustness).** A byte-level text encoder trained with simulated OCR noise degrades gracefully under real OCR errors, and the remaining gap can be closed by adapting to the OCR engine’s output.",
      "**H3 (feasibility).** With change-aware frame skipping, the full pipeline analyses a changed frame in well under one second on a laptop CPU, with model inference in the order of milliseconds.",
    ]),
    H2("1.3 Aim and Objectives"),
    P("**Aim:** to design, build and evaluate a privacy-preserving prototype that analyses screen frames in near real time and identifies both user activity and safety threats using multimodal deep learning."),
    ...bullets([
      "**Objective 1 (data):** create a reproducible, labelled data source covering 9 activities and 5 threat categories on desktop and mobile layouts, with at least 10,000 frames and ground-truth screen text.",
      "**Objective 2 (model):** design and train a multimodal, multi-task network with fewer than 2 million parameters and heads for activity, explicit content and threats.",
      "**Objective 3 (accuracy):** reach at least 95% activity accuracy and a threat macro-F1 of at least 0.90 on held-out data, and quantify the effect of real OCR and missing text.",
      "**Objective 4 (real time):** keep model inference under 20 ms per frame on one CPU thread and the analysis of a changed frame under 1 s, with an export path to ONNX for edge devices.",
      "**Objective 5 (trust):** give every threat alert human-readable reasons, provide visual explanations, and store no screen content by default.",
    ]),
    H2("1.4 Scope and Boundaries"),
    P("The prototype works on still frames captured from desktop screens (landscape) and phone screens (portrait), at any resolution, and on English-language text. It covers the nine activities and five threat categories listed in Appendix A, runs on a CPU, and uses Tesseract for OCR. Training and evaluation data are synthetic screens produced by our own generator, part of which were passed through real OCR to measure the effect of recognition errors."),
    P("Several things are deliberately outside the scope of this work. We did not collect recordings of real users’ screens, because that requires informed consent and ethics approval that we did not have during the course. The explicit-content head is fully implemented and testable, but we did not train it on real explicit imagery, which needs an ethically approved and legally reviewed dataset. We did not package the system as a mobile app, did not handle non-English text, audio or network traffic, and did not implement blocking: ScreenGuard warns the user and leaves the decision with them."),
    H2("1.5 Significance and Intended Impact"),
    P("The first beneficiaries are individual users, who get a warning at the moment it is useful, when the fake login page is open and before the password is typed. Parents, schools and small businesses gain a single safety layer that works across applications and needs no security staff. Because all processing happens on the device and the model is only 5 MB, the approach respects privacy and costs nothing to run at scale: there are no servers, no per-user cloud fees, and no screenshots leaving the machine."),
    P("Academically, the project contributes a reproducible benchmark protocol for joint screen understanding and safety detection, a quantified analysis of the OCR bottleneck, and ablation evidence on fusion strategies, backbones and text encoders under a tight compute budget. For real-world impact we propose three measurable indicators: detection rate at a fixed false-alarm budget (for example, recall at 1% false positives), time from threat appearance to alert, and average CPU cost per hour of use. These indicators connect directly to the financial and social harm described in §1.1."),
  ];
}

function section2() {
  return [
    H1("2 Evidence and Prior Work"),
    H2("2.1 Review Approach"),
    P("We searched Google Scholar, IEEE Xplore, the ACM Digital Library and arXiv for work published between 2014 and 2026, using combinations of the terms *screen understanding*, *UI dataset*, *visual phishing detection*, *scam detection*, *character-level text classification*, *multimodal fusion*, *efficient CNN* and *on-device inference*. We gave priority to peer-reviewed papers at major venues (CVPR, ICCV, NeurIPS, ICLR, ICML, USENIX Security, ACM CCS, UIST and CHI) and to widely adopted methods. Official documentation for PyTorch [@pytorch], ONNX Runtime [@onnxruntime] and Tesseract supported implementation, public FBI and APWG reports gave context on the threat landscape, and we examined how existing browser-protection and parental-control products approach the problem. A source was kept if it addressed one of our five themes, described its method clearly, and was relevant to CPU or on-device deployment."),
    H2("2.2 Relevant Knowledge and Existing Solutions"),
    P("**Screen and UI understanding.** Large datasets of user interfaces have driven progress in this area. Rico collected tens of thousands of Android screens together with their view hierarchies [@rico]; Screen2Words trained multimodal models to summarise mobile screens in natural language [@screen2words]; and WebUI gathered hundreds of thousands of rendered web pages with accessibility metadata [@webui]. A consistent lesson from this work is that pixels and text together describe a screen better than either alone. However, these systems target design and accessibility tasks, not safety, and they usually rely on structured metadata (view hierarchies, DOM trees) that a screen-level observer cannot access."),
    P("**Visual phishing detection.** VisualPhishNet compares the appearance of a page with a set of protected brands using a triplet network [@visualphishnet]. Phishpedia combines logo detection with Siamese matching to identify the impersonated brand and explain its decision [@phishpedia], and PhishIntention adds an analysis of whether the page actually intends to collect credentials [@phishintention]. These are strong, explainable approaches, but they assume a browser screenshot together with its URL or HTML, use object detectors that are heavy for continuous use on a laptop, and do not cover scams, malicious downloads or threats inside chat, e-mail and terminal windows."),
    P("**Text-based detection.** Text classifiers have long been used against SMS and e-mail spam, supported by public corpora such as the SMS Spam Collection [@smsspam]. Character-level convolutional networks [@zhang_charcnn] and CNN sentence classifiers [@kim_cnn] are known to tolerate spelling variation, and tokenization-free Transformers such as CANINE downsample character sequences so that attention remains affordable [@canine]. URL analysis typically combines lexical features with edit-distance measures for lookalike detection [@damerau]. For dangerous commands, the security community maintains public knowledge bases such as MITRE ATT&CK [@attack] and detection rules in the Sigma format [@sigma]."),
    P("**Efficient vision and multimodal learning.** MobileNetV2 introduced inverted residual blocks [@mnv2], MobileNetV3 added squeeze-and-excitation [@se] and the hard-swish activation [@mnv3], and EfficientNet scaled such networks systematically [@efficientnet]. Vision Transformers reach excellent accuracy but are known to need large datasets or pretraining [@vit]. For combining modalities, the Gated Multimodal Unit learns how much to trust each input [@gmu], while Transformer encoders [@transformer] allow tokens from different modalities to attend to each other. Multimodal networks often over-rely on one modality and under-train the other [@wang_hard], a problem that modality dropout helps to reduce [@moddrop]. Further building blocks used here are uncertainty-based multi-task weighting [@kendall], temperature scaling for calibration [@guo], Grad-CAM for visual explanations [@gradcam] and integer quantization for efficient inference [@jacob]."),
    P("**Explicit content and deployed products.** Open-source classifiers such as Yahoo’s open_nsfw model [@open_nsfw] and Apple’s on-device Communication Safety feature [@apple_cs] show that explicit-image detection is mature and can run locally. URL reputation services such as Safe Browsing [@safebrowsing] protect billions of users inside browsers. " + TB("existing") + " compares these approaches with the solution developed in this project."),
    ...TABLE("existing", "Comparison of representative existing solutions with this project.",
      ["Solution", "Input", "Coverage", "Activity-aware", "On-device", "Explanations"],
      [["Safe Browsing and browser warnings [@safebrowsing]", "URL reputation", "Web pages", "No", "Partly (local hash lists)", "Category only"],
        ["VisualPhishNet, Phishpedia, PhishIntention [@visualphishnet,@phishpedia,@phishintention]", "Page screenshot + URL/HTML", "Phishing websites", "No", "Research prototypes", "Target brand / intention"],
        ["SMS and e-mail filters (e.g. char-CNN [@zhang_charcnn])", "Message text", "One channel", "No", "Varies", "Rarely"],
        ["open_nsfw, Communication Safety [@open_nsfw,@apple_cs]", "Images", "Explicit content", "No", "Yes (Apple)", "No"],
        ["UI understanding models [@rico,@screen2words,@webui]", "Screenshot + metadata", "UI semantics", "Yes", "Not targeted", "Not targeted"],
        ["**ScreenGuard (this project)**", "Screen pixels + OCR text", "Activity, explicit content, 4 threat types, any app", "Yes", "Yes, CPU, 5 MB", "Rule and URL reasons, Grad-CAM, word saliency"]],
      W([2.6, 1.7, 1.9, 1.1, 1.4, 1.7])),
    H2("2.3 Gap or Opportunity"),
    P("The review points to five gaps. First, **coverage is fragmented**: each existing tool protects a single channel, while real attacks move between SMS, browser, e-mail, social media and the terminal. Second, **no public dataset labels screen activity, explicit content and cyber threats together**. Third, most detectors **assume structured inputs** such as URLs, HTML or view hierarchies, which an observer of the raw screen does not have. Fourth, **real-time cost on CPUs is rarely measured end to end**; in particular, the cost of OCR is usually left out, even though, as we show, it dominates the budget. Fifth, **trust is under-served**: alerts often come without explanations or calibrated confidence, which leads users to ignore them."),
    P("ScreenGuard responds to each gap: one model for any application, a reproducible generator of jointly labelled data, pixels and OCR text as the only inputs, latency reported for every stage including OCR, and calibrated, explained alerts."),
    H2("2.4 Conceptual and Design Basis"),
    P("Five design principles guided the work. (1) **Multimodal complementarity:** layout tells us *what* the user is doing, text tells us *whether* it is dangerous, so both are modelled jointly. (2) **Privacy by design:** all inference happens on the device, frames are never saved, and logs exclude screen text by default. (3) **Neuro-symbolic decision making:** a learned model provides generalisation, while transparent rules provide explanations and a safety net for novel attacks. (4) **Compute awareness:** the cheapest frame is the one that is not analysed, so change detection and asynchronous OCR come before model compression. (5) **Trustworthy alerting:** probabilities are calibrated, and alerts follow hysteresis so that they do not flicker."),
    P("The benchmark criteria follow from this setting: macro-F1, because classes are imbalanced; AUROC and recall at a 1% false-positive rate, because threats are rare and false alarms erode trust; expected calibration error (ECE); and per-stage latency, parameter count and model size."),
  ];
}

function section3() {
  return [
    H1("3 Methodology or Development Approach"),
    P("ScreenGuard is primarily an innovation project, but its claims rest on controlled experiments. We therefore use both sets of prompts from the template: §3.1 describes the research methodology behind the evidence, and §3.2 the development of the solution itself."),
    H2("3.1 Research Methodology"),
    H3("Research design"),
    P("We followed a quantitative, experimental design. One main configuration was trained and evaluated on held-out data under several conditions: clean text, simulated OCR noise, no text, and real OCR. An ablation study then changed one factor at a time (modalities, fusion type, image backbone, text encoder stem, modality dropout) under an identical, deliberately small training budget, so that differences can be attributed to the factor being varied. Every split comes from a different random seed of the generator, so that no test screen is a copy of a training screen."),
    H3("Dataset and materials"),
    P(`Because no public dataset provides joint labels for our three tasks, we wrote a procedural screen generator in Python with Pillow. It renders realistic desktop windows (Windows-, macOS- and Linux-style chrome, taskbars, distractor windows, notifications) and phone screens for all nine activities, with randomised themes, fonts, DPI scaling and layouts (${F("samples")}). Every frame comes with its labels and with the visible text in reading order, which is exactly what a perfect OCR engine would return. Text hidden behind pop-ups is removed, as it would be on a real screen. One frame takes about 20 ms to render at 640×360, so 8,000 frames take 45 seconds on four processes.`),
    P("To keep the task honest, the generator includes **hard negatives**: genuine security notifications from real brand domains, the official Rust installer that pipes «curl» into «sh», legitimate «.exe» downloads from vendor sites, hyphenated but harmless domains, and finance articles full of words such as “account” and “bank”. About a quarter of the phishing login pages carry no warning banner, so only the URL betrays them. All IP addresses come from the RFC 5737 documentation ranges, phone numbers are fictional, and no explicit imagery is generated at any point. Screen text has a median length of 545 bytes (90th percentile: 1,297 bytes)."),
    ...TABLE("data", "Data splits. Each split is generated from its own random seed.",
      ["Split", "Frames", "Resolution", "Seed", "Use"],
      [["Train (two sets of 8,000)", "16,000", "640×360 / 360×640", "101, 404", "Main model"],
        ["Train subset", "4,000", "640×360 / 360×640", "101", "Ablation budget"],
        ["Validation", "1,000", "640×360 / 360×640", "202", "Model selection, calibration"],
        ["Test", "1,500", "640×360 / 360×640", "303", "Held-out evaluation"],
        ["High-resolution test + Tesseract OCR", "300", "1280×720 / 720×1280", "505", "Real-OCR evaluation"],
        ["High-resolution train + Tesseract OCR", "2,500 + 500", "1280×720 / 720×1280", "606", "OCR-aware fine-tuning"]],
      W([3.2, 1.3, 2.1, 1.1, 2.4])),
    P("The label distribution is close to uniform across activities (about 11% each). Roughly 62% of frames are benign and 8–11% belong to each threat class, and about 13% of frames are phone screens. In real use threats are far rarer, which is why we also report metrics at a fixed false-positive rate."),
    H3("Instruments and procedures"),
    P(`All models were implemented in PyTorch 2.15 [@pytorch] and trained on a 4-core Intel Xeon CPU (2.3 GHz) without a GPU. ${TB("config")} lists the training configuration. Images are letterboxed (resized without distortion and padded) to 144×256 pixels and normalised. During training we applied light photometric augmentation and small translations, but no horizontal flips, since flipping would invert conventions such as chat-bubble sides. We also injected 3% simulated OCR noise: confusable characters (0/o, 1/l, rn/m), merged words and dropped lines.`),
    ...TABLE("config", "Training configuration of the main model.",
      ["Setting", "Value"],
      [["Optimiser", "AdamW [@adamw], learning rate 2×10⁻³, weight decay 0.05, gradient clipping 1.0"],
        ["Schedule", "5% linear warm-up, then cosine decay; 6 epochs; batch size 64"],
        ["Loss", "Masked multi-task cross-entropy, label smoothing 0.05 [@labelsmooth], learned uncertainty weights [@kendall]"],
        ["Regularisation", "Dropout 0.1, modality dropout 0.15, OCR-noise augmentation 3%"],
        ["Model selection", "Best mean validation macro-F1 over trained tasks; early-stopping patience 4"],
        ["Calibration", "One temperature per task fitted on validation NLL, clamped to [0.25, 10]"],
        ["Training time", "29 minutes for 6 epochs on 16,000 frames (4 CPU cores)"]],
      W([1.6, 6.4])),
    P("For the real-OCR experiments we used Tesseract 5.3.4 [@tesseract] in sparse-text mode (page segmentation mode 11). Before recognition, frames were converted to greyscale, dark-themed screens were inverted, and images were upscaled to about 1,080 pixels in height, because Tesseract is trained on dark text on a light background and prefers larger glyphs. Words below 30% confidence were discarded, and each OCR process was limited to one thread."),
    H3("Variables and measures"),
    P("The independent variables were the input modalities (image, text or both), the fusion strategy, the vision backbone, the text-encoder stem, modality dropout, and the text condition at test time. The dependent variables were accuracy and macro-F1 for activities and threats; for the binary “any threat” decision, AUROC, recall, false-positive rate and recall at a 1% false-positive rate; ECE; latency for each pipeline stage; parameter count; and model size."),
    H3("Analysis method and quality checks"),
    P("All metrics are computed by our own numpy implementation, so that the evaluation code is short and auditable. Several quality checks were built in. Splits use disjoint seeds. Every reported number is written by a script to a JSON file in the repository and can be regenerated with one command («scripts/reproduce.sh»). The code base is covered by 65 automated tests, including an end-to-end train, evaluate and export test. Exported ONNX models are compared numerically with the PyTorch model. Because each ablation variant was trained once, we treat differences below about 0.02 as noise."),
    H3("Ethics, consent, privacy and data protection"),
    P("The study involved no human participants and no personal data: all screens are synthetic, and real OCR was applied only to generated frames. We did not generate or collect explicit images. Because screen monitoring is a dual-use technology that could be abused for covert surveillance, the system was designed for transparent use: it runs locally, announces itself when started, never writes frames to disk, keeps screen text out of logs by default, and redacts e-mail addresses, phone numbers, card numbers and passwords if text logging is switched on."),
    H2("3.2 Innovation Development Approach"),
    H3("Target users, stakeholders and requirements"),
    P(`The primary users are individual laptop and phone owners. Secondary stakeholders are parents and schools responsible for minors, small organisations without security staff, and the security or parental-control vendors who might integrate the technology. From their needs we derived the requirements and acceptance criteria in ${TB("requirements")}, which are tested in §4.3.`),
    ...TABLE("requirements", "Requirements and acceptance criteria.",
      ["ID", "Requirement", "Acceptance criterion"],
      [["R1", "Recognise what the user is doing on screen", "Activity accuracy ≥ 95% on held-out data"],
        ["R2", "Detect phishing, scams, malicious downloads and dangerous commands", "Threat macro-F1 ≥ 0.90; recall ≥ 90% at ≤ 1% false positives"],
        ["R3", "Tolerate imperfect OCR and missing text", "Measured drop with real OCR; fine-tuning path that closes the gap"],
        ["R4", "Run continuously on a CPU-only device", "Model ≤ 20 ms per frame on 1 thread; changed frame < 1 s; model < 10 MB"],
        ["R5", "Explain every alert", "Rule or URL reasons for threat alerts; Grad-CAM and word saliency on request"],
        ["R6", "Stable, non-intrusive alerts", "Alert within a few frames of a threat appearing; no flicker; no false alerts on benign screens in the scripted session"],
        ["R7", "Privacy by default", "No frame storage, no screen text in logs, no network calls"]],
      W([0.6, 4.0, 4.4])),
    H3("Alternatives considered and the selected approach"),
    P("We considered four approaches. (A) **Sending screenshots to a cloud vision-language model** would be powerful but conflicts with privacy, adds cost and latency, and fails without connectivity. (B) **An image-only classifier** is cheap but cannot read a lookalike domain such as «paypa1-secure.net» at 256 pixels of width; our ablation confirms that it reaches a threat F1 of only 0.33. (C) **OCR plus hand-written rules** is transparent, but brittle: its F1 falls from 0.87 to 0.29 under 20% OCR noise (§4.2). (D) **A compact on-device multimodal network, combined with explainable rules and temporal logic**, keeps the strengths of (B) and (C) without their weaknesses. We selected (D)."),
    H3("System architecture"),
    P(`${F("pipeline")} shows the complete pipeline. A frame is first compared with the last analysed frame using a 256-bit difference hash [@dhash] and a coarse pixel difference. If nothing meaningful has changed, which is the case for about 80% of frames in practice, the cached verdict is reused and no OCR or model computation is spent. Changed frames are sent to OCR and pre-processing, then to the ScreenThreatNet model. The model’s threat probabilities are combined with evidence from rules and URL analysis, and the result passes through temporal smoothing and an alert state machine before an alert is shown with its reasons. A key design decision is that skipped frames still feed the temporal logic, so a dangerous page that stays on screen keeps accumulating evidence.`),
    ...FIG("pipeline", "fig_pipeline.png", "End-to-end real-time pipeline of ScreenGuard with measured latencies on a 4-core laptop-class CPU. Unchanged frames bypass OCR and the model but still update the temporal logic."),
    H3("ScreenThreatNet: the multimodal, multi-task model"),
    P(`${F("model")} shows the model. The screen frame and its OCR text are encoded separately into sequences of 128-dimensional tokens. A fusion Transformer lets them interact, and three classification heads produce the activity, explicit-content and threat probabilities. The whole model has 1.28 million parameters: 0.34 M in the vision encoder, 0.49 M in the text encoder, 0.40 M in the fusion module and 0.05 M in the heads.`),
    ...FIG("model", "fig_model.png", "Architecture of ScreenThreatNet. The vision and byte-level text encoders emit token sequences; the attention fusion module prepends one learnable query token per task; each task head reads its own token."),
    P(`**Vision encoders.** All vision encoders output a grid of tokens rather than a single vector, which allows the fusion module to attend to *where* things are on the screen and makes Grad-CAM available for every backbone (${F("vision")}). Our main encoder, TinyCNN, follows MobileNetV3 [@mnv3]: inverted residual blocks [@mnv2] with depthwise convolutions, squeeze-and-excitation [@se] and hard-swish activations. It downsamples early, so that most computation happens at 1/8 to 1/32 of the input resolution, and it produces a 5×8 grid of 40 tokens. Our first design downsampled late and was 3.7 times slower to train; moving the strides earlier fixed this. For comparison we also implemented a ViT-tiny with interpolated position embeddings, so that phones and laptops can use different input shapes, and wrappers for MobileNetV3-Small and EfficientNet-B0 [@efficientnet].`),
    ...FIG("vision", "fig_vision.png", "Vision encoders. (a) TinyCNN stages with channel counts and feature-map sizes for a 144×256 input. (b) The inverted residual block. (c) ViT-tiny. (d) The four backbones compared in this project, which all share the same 128-dimensional token interface."),
    P(`**Byte-level text encoder.** OCR output is noisy («paypa1», «rn» read as «m»), and phishing relies on character-level tricks such as homoglyphs, punycode and extra hyphens. Sub-word vocabularies tend to fragment such strings unpredictably, so we work directly on bytes, which also means there is no vocabulary file to ship. Long screens are truncated head-first (75%) and tail-last (25%), which keeps both the address bar and the bottom of the screen. ${F("text")} illustrates the most important design decision in this encoder. Our first version used a single strided convolution (kernel 16, stride 8), similar to patch embedding in ViTs. Such a layer must learn a keyword like “verify” separately for each of eight possible alignments within a window. Replacing it with a stride-1 n-gram convolution followed by max-pooling, in the spirit of character-level CNNs [@zhang_charcnn,@kim_cnn] and the downsampling of CANINE [@canine], makes detectors fire wherever an n-gram occurs. At equal cost, this change raised threat F1 from 0.578 to 0.912 in the ablation.`),
    P(`**Fusion.** We implemented the three fusion strategies shown in ${F("fusion")}. *Concat* pools each modality and concatenates the vectors (late fusion). The *Gated Multimodal Unit* [@gmu] learns a per-dimension gate that decides how much to trust each modality. *Attention fusion*, our main configuration, runs a two-layer Transformer [@transformer] over the sequence [task tokens | visual tokens | text tokens] with modality-type embeddings. Each task owns its own query token, so the threat head can attend to URL bytes while the activity head attends to layout. Missing text, which is common for video frames or OCR failures, is handled explicitly: attention fusion masks the text tokens, and the pooled fusions substitute a learned “missing” vector. During training, **modality dropout** [@moddrop] hides one modality for 15% of samples, so that neither pathway becomes lazy [@wang_hard].`),
    ...FIG("text", "fig_text_encoder.png", "Why the byte encoder uses an n-gram convolution with max-pooling. (a) A strided stem sees the same word at different offsets in each window. (b) A stride-1 detector followed by max-pooling responds to the word wherever it appears. (c) The complete encoder."),
    ...FIG("fusion", "fig_fusion.png", "The three fusion strategies compared. Parameter counts and single-thread latencies refer to the full model with each fusion variant."),
    P("**Loss and calibration.** Real training data are heterogeneous: an NSFW dataset labels only explicit content, and a phishing corpus labels only threats. We therefore mask the loss of each task to the samples that carry its label, and learn task weights through homoscedastic uncertainty [@kendall]:"),
    EQ("L = Σ_{t} [ exp(−s_{t}) · L_{t} + s_{t} ],   L_{t} = cross-entropy over the samples labelled for task t", 1),
    P("where s_{t} = log σ_{t}^{2} is a learnable scalar for each task. After training, a temperature T_{t} is fitted for each task on the validation set [@guo], and the calibrated probabilities are"),
    EQ("p_{t} = softmax(z_{t} / T_{t}),   0.25 ≤ T_{t} ≤ 10", 2),
    P("We introduced the bounds on T_{t} after observing that, on an almost perfectly classified validation set, the fitted temperature collapsed to 0.08. That would make the model extremely over-confident on real, unfamiliar screens."),
    H3("Decision layer and alerting"),
    P(`The threat decision combines the calibrated model with explainable evidence (${F("decision")}). URLs found in the OCR text are parsed into registered domains and checked for imitation of 16 commonly targeted brands, using homoglyph normalisation and the Damerau–Levenshtein distance [@damerau], as well as for raw IP hosts, punycode, frequently abused top-level domains, link shorteners and executable downloads. Forty-five signature rules, inspired by MITRE ATT&CK [@attack] and Sigma [@sigma], cover encoded PowerShell, «curl | sh» from raw IP addresses, abused system binaries, reverse shells, “ClickFix” paste-and-run lures, double file extensions, tech-support and crypto scams, and parcel smishing. Each matching rule i with weight w_{i} contributes evidence r_{c} = 1 − Π_{i} (1 − w_{i}) for its class c. This evidence and the model probabilities are then combined by a noisy-OR:`),
    EQ("q_{c} = 1 − (1 − p_{c})(1 − r_{c}),   q_{benign} = p_{benign} · Π_{c} (1 − r_{c})", 3),
    P("after which q is renormalised and the risk is defined as 1 − q_{benign}. Either source can raise the risk, but a confident benign model prediction can still outweigh a weak rule. This matters, for instance, for the official Rust installer, which triggers a pipe-to-shell rule with a weight of only 0.40. The risk is smoothed with an exponential moving average (α = 0.6), and an alert is raised only after it stays at or above 0.70 for two analysed frames. It is cleared only after it stays below 0.40 for three frames. Cooldowns apply per threat label, so a new kind of threat is reported immediately."),
    ...FIG("decision", "fig_decision.png", "Explainable decision layer. (a) URL and signature evidence; (b) noisy-OR fusion with the calibrated model; (c) the alert hysteresis state machine."),
    H3("Originality and new contribution"),
    P("We do not claim that the individual building blocks are new: efficient CNNs, Transformers, GMU fusion, uncertainty weighting and temperature scaling all come from the literature cited above. The contribution of this project lies in how they are combined and adapted to a problem that existing tools address only in fragments. Compared with visual phishing detectors [@visualphishnet,@phishpedia,@phishintention], ScreenGuard works from raw screen pixels and OCR text in any application, covers four threat types together with activity and explicit content, and runs on a CPU. Compared with message filters and URL blocklists, it adds visual context and generalises to lookalike domains it has never seen. Compared with UI-understanding models [@rico,@screen2words,@webui], it needs no view hierarchy and targets safety. Several specific improvements emerged during development: an n-gram/max-pool byte stem for OCR text, task-specific query tokens for multi-task fusion, the masked multi-task loss for heterogeneous datasets, bounded temperature scaling, and a frame-skipping pipeline whose temporal logic keeps running on cached predictions."),
    H3("Prototype implementation"),
    P("The prototype is a Python package of about 5,700 lines in 42 modules, with a command-line interface: «screenguard generate | train | evaluate | export | benchmark | analyze | monitor | ocr-manifest». Frames can come from the live screen (via the «mss» library), a video file, an image folder or a scripted synthetic session. Models are exported to ONNX [@onnxruntime] in fp32 and in int8, with dynamic quantization [@jacob] restricted to the matrix multiplications, and every export is checked automatically against PyTorch. The «analyze» command produces Grad-CAM heat-maps [@gradcam] and word-level saliency for a single screenshot. Appendix A lists the details."),
    H3("Testing and validation criteria"),
    P(`Validation combined four kinds of evidence: (i) held-out metrics against the acceptance criteria in ${TB("requirements")}; (ii) robustness tests with simulated OCR noise, missing text and real Tesseract OCR; (iii) latency benchmarks for every pipeline stage and backbone, across runtimes and thread counts; and (iv) an end-to-end scripted session of 13 screens (8 benign and 5 threats) shown for 5 frames each, processed by the real pipeline with real OCR. User feedback could not be collected within the course timeline, and a user study is planned as a next step (§6.3).`),
    H3("Resources, costs, risks, ethics, accessibility and sustainability"),
    P("The whole project ran on a 4-core CPU without a GPU, and all software used is open source, so the direct cost was zero. In total, the experiments used only a few CPU-hours: about 29 minutes for the main model, around 25 minutes for the full ablation grid, and a few minutes each for fine-tuning and OCR. The main risks (the synthetic-to-real gap, OCR dependence, adversarial evasion and misuse for surveillance) are analysed in §5.4. For accessibility, alerts are plain text with reasons, so they can be read aloud by screen readers, and the system runs on modest hardware. For sustainability, skipping unchanged frames avoids most of the computation, and CPU-only inference avoids the energy cost of a GPU or a data centre."),
    H2("3.3 Team Roles and Project Management"),
    P(`The project was organised as six work packages. ${TB("roles")} summarises responsibilities and outcomes.`),
    ...TABLE("roles", "Work packages, responsibilities and status.",
      ["Work package", "Lead", "Deliverable", "Status"],
      [["WP1 Literature review and requirements", PERSON.member, "Review (§2), requirements (Table 4)", "Completed"],
        ["WP2 Synthetic screen generator and threat corpus", PERSON.member, "Generator, 21,800 labelled frames", "Completed"],
        ["WP3 Model design and training", PERSON.leader, "ScreenThreatNet, training pipeline", "Completed"],
        ["WP4 Rules, URL analysis and decision layer", PERSON.member, "45 rules, lookalike detector, noisy-OR", "Completed"],
        ["WP5 Real-time pipeline, OCR and edge export", PERSON.leader, "Monitor, async OCR, ONNX fp32/int8", "Completed"],
        ["WP6 Evaluation, ablations and report", `Both (leader: ${PERSON.leader})`, "Experiments, figures, this report", "Completed"],
        ["Explicit-content training", "Both", "Trained NSFW head", "Deferred (needs approved dataset)"]],
      W([3.0, 2.0, 3.0, 1.6])),
    P("Several changes were made to the original plan, each prompted by evidence. (1) The strided byte stem was replaced by the n-gram stem after a pilot showed that it learned threats far too slowly (text-only threat F1 of 0.36 against 0.57 after three epochs on 2,000 screens). (2) OCR-aware fine-tuning was added after real OCR revealed a 5-point drop in activity accuracy. (3) The ablation budget was reduced from 8,000 to 4,000 training screens to fit the CPU budget, which also made differences between variants easier to see. (4) Training of the explicit-content head was postponed until an ethically approved dataset is available, and the pipeline reports this head as “unavailable” rather than guessing. (5) PyTorch’s eager int8 quantisation was replaced by ONNX Runtime quantisation, because the former breaks Transformer layers and is being deprecated."),
  ];
}

function section4() {
  return [
    H1("4 Results or Proposed Solution"),
    H2("4.1 Main Output"),
    P(`The main output is **ScreenGuard**, a working prototype that watches the screen, understands what is happening, and warns the user about safety threats with an explanation. ${F("console")} shows its actual console output during the scripted session. Each line reports the recognised activity, the threat class and risk, and the time spent. When a threat persists, an alert is raised with its reasons: for example, that «coinbase1-support.net» mentions Coinbase but is not an official Coinbase domain, or that a command “downloads and executes code in memory”. The alert is cleared once the user moves on.`),
    ...FIG("console", "fig_console.png", "Real output of «screenguard monitor» on the scripted session with Tesseract OCR. Lines marked (unchanged) were served from the cache in about 3 ms; alerts list the evidence behind them."),
    P("The deliverables include the trained model (5.07 MB as ONNX fp32 and 2.77 MB as int8), an OCR-aware fine-tuned version recommended for deployment, the data generator, and the evaluation, benchmark and reproduction scripts. A user can analyse a single screenshot with «screenguard analyze shot.png», or protect their own desktop with «screenguard monitor --source screen --async-ocr»."),
    H2("4.2 Evidence and Visuals"),
    H3("Recognition accuracy"),
    P(`${TB("main")} reports the main model on the 1,500-screen held-out test set under different text conditions, and on the 300 high-resolution frames read by real Tesseract OCR. With clean text, activity recognition is perfect and threat detection reaches a macro-F1 of 0.994 with well-calibrated confidence (ECE 0.003). Per-class threat F1 is 0.998 for benign, 0.991 for phishing, 0.994 for scam, 1.000 for malware download and 0.985 for suspicious command, and accuracy is the same on desktop and phone screens. The confusion matrices in ${F("confusion")} show that the few remaining errors mostly concern suspicious commands.`),
    ...TABLE("main", "Main model (1.28 M parameters) under different text conditions.",
      ["Test condition", "Activity acc.", "Threat acc.", "Threat macro-F1", "Any-threat AUROC", "Recall @ 1% FPR", "ECE (threat)"],
      [["Synthetic test, clean text", "**1.000**", "**0.996**", "**0.994**", "0.998", "0.993", "0.003"],
        ["5% simulated OCR noise", "0.998", "0.988", "0.985", "0.996", "0.977", "0.011"],
        ["10% simulated OCR noise", "0.989", "0.963", "0.952", "0.987", "0.926", "0.032"],
        ["20% simulated OCR noise", "0.953", "0.880", "0.832", "0.943", "0.713", "0.107"],
        ["No text (OCR failure)", "0.787", "0.626", "0.247", "0.673", "0.033", "0.164"],
        ["720p frames, ground-truth text", "1.000", "1.000", "1.000", "1.000", "1.000", "0.002"],
        ["720p frames, real Tesseract OCR", "0.950", "0.980", "0.971", "0.997", "0.974", "0.019"]],
      W([2.9, 1.1, 1.1, 1.2, 1.2, 1.2, 1.1]), { align: [undefined, CENTER, CENTER, CENTER, CENTER, CENTER, CENTER] }),
    ...FIG("confusion", "fig_confusion.png", "Row-normalised confusion matrices of the main model on the synthetic test set with clean text: activity (left) and threat (right)."),
    H3("Robustness to OCR and OCR-aware fine-tuning"),
    P(`Performance degrades gracefully with noisy text: at a 10% character error rate, threat F1 is still 0.952. Without any text, activity accuracy remains at 79% thanks to the visual pathway, but threat detection collapses (F1 0.247), which confirms that most threats are fundamentally textual. Real Tesseract OCR, with a mean character similarity of 0.854 to the ground truth, costs 5.0 points of activity accuracy and 2.9 points of threat F1 compared with ground-truth text on the same 720p frames. ${F("ocrgap")} shows where these losses occur. Terminal screens lose the most (recall 0.70), because Tesseract turns dark-theme shell prompts such as «user@host:~/project$» into fragments the model never saw during training. Fine-tuning on Tesseract’s own output for two short epochs closes the gap completely (${TB("finetune")}), without any regression on clean text.`),
    ...FIG("ocrgap", "fig_ocr_gap.png", "Per-class recall with real Tesseract OCR on the unseen 720p test set, before (blue) and after (orange) OCR-aware fine-tuning."),
    ...TABLE("finetune", "Effect of OCR-aware fine-tuning (720p test set with real Tesseract OCR unless noted).",
      ["Metric", "Main model", "+ OCR-aware fine-tune"],
      [["Activity accuracy", "0.950", "**1.000**"], ["Terminal recall / document-editing recall", "0.704 / 0.861", "**1.000 / 1.000**"],
        ["Threat macro-F1 (model)", "0.971", "**0.992**"], ["Suspicious-command recall", "0.889", "**1.000**"],
        ["Threat macro-F1 (hybrid decision)", "0.975", "**0.996**"], ["Any-threat false-positive rate (hybrid)", "1.07%", "**0.53%**"],
        ["Synthetic test, clean text: threat macro-F1", "0.994", "0.993"], ["Synthetic test, 20% OCR noise: threat macro-F1", "0.832", "0.857"]],
      W([4.6, 2.2, 2.6]), { align: [undefined, CENTER, CENTER] }),
    H3("Model, rules and hybrid decision"),
    P(`${TB("fusiondec")} compares the three decision modes. The hand-written rules alone reach an F1 of 0.867 on clean text, but fall to 0.289 at 20% OCR noise, because regular expressions break when a single character changes. The learned model is far more robust. The hybrid decision is best on clean text and on real OCR (0.995 and 0.975), with a false-positive rate of only 0.11% on clean text. Under heavy synthetic noise, false positives from the rules make the hybrid slightly worse than the model alone. The main value of the rules is therefore explanation, together with a safety net for novel attacks, and their weight should be tuned on real data.`),
    ...TABLE("fusiondec", "Threat macro-F1 of the three decision modes (main model).",
      ["Test condition", "Model only", "Rules only", "Hybrid (noisy-OR)", "Hybrid FPR", "Hybrid recall"],
      [["Clean text", "0.994", "0.867", "**0.995**", "0.11%", "99.8%"], ["5% OCR noise", "**0.985**", "0.715", "0.974", "1.7%", "97.7%"],
        ["10% OCR noise", "**0.952**", "0.524", "0.942", "2.6%", "93.5%"], ["20% OCR noise", "**0.832**", "0.289", "0.829", "6.0%", "80.6%"],
        ["Real Tesseract OCR (720p)", "0.971", "0.804", "**0.975**", "1.1%", "98.2%"]],
      W([2.8, 1.3, 1.3, 1.6, 1.3, 1.3]), { align: [undefined, CENTER, CENTER, CENTER, CENTER, CENTER] }),
    H3("Ablation study"),
    P(`${TB("ablation")} and ${F("ablations")} summarise the ablation study (4,000 training screens, 3 epochs, one factor changed at a time). Four results stand out. First, **neither modality is enough alone**: the image-only model cannot detect threats (F1 0.330), and the text-only model fails completely when OCR returns nothing (activity accuracy 0.109). Second, **gated fusion is the best small-data choice**, with a threat F1 of 0.976 on clean text and 0.947 with real OCR. Attention fusion is more data-hungry: it reaches only 0.912 at this budget, but 0.994 with 16,000 screens. Third, **CNNs beat the Vision Transformer** at this data scale (0.876 for ViT-tiny), consistent with the known need of ViTs for large data or pretraining [@vit]. Fourth, the **text stem** has the largest effect of any single design choice (0.578 for the strided stem against 0.912 for the n-gram stem).`),
    ...TABLE("ablation", "Ablation study (4,000 training screens, 3 epochs; synthetic test set unless noted).",
      ["Variant", "Params", "Activity acc.", "Threat F1", "Activity acc. (real OCR)", "Threat F1 (real OCR)", "Activity acc. (no text)", "Hybrid threat F1"],
      [["Attention fusion (main config.)", "1.28 M", "0.997", "0.912", "0.957", "0.881", "0.414", "0.947"],
        ["Image only", "0.79 M", "0.866", "0.330", "0.897", "0.335", "0.866", "0.834"],
        ["Text only", "0.94 M", "0.997", "0.969", "0.950", "0.911", "0.109", "0.980"],
        ["Concat fusion", "0.92 M", "0.994", "0.955", "0.963", "0.902", "0.547", "0.975"],
        ["Gated fusion (GMU)", "0.95 M", "**0.998**", "**0.976**", "0.957", "**0.947**", "**0.617**", "**0.987**"],
        ["ViT-tiny backbone", "2.92 M", "0.992", "0.876", "0.927", "0.801", "0.265", "0.930"],
        ["MobileNetV3-Small backbone", "1.94 M", "0.996", "0.954", "0.937", "0.922", "0.106", "0.976"],
        ["No modality dropout", "1.28 M", "0.995", "0.944", "0.950", "0.879", "0.186", "0.970"],
        ["Strided text stem", "1.50 M", "0.979", "0.578", "0.803", "0.495", "0.617", "0.842"]],
      W([2.7, 0.9, 1.0, 0.9, 1.15, 1.1, 1.15, 1.0]), { size: 17, align: [undefined, CENTER, CENTER, CENTER, CENTER, CENTER, CENTER, CENTER] }),
    ...FIG("ablations", "ablations.png", "Ablation results on clean text: threat macro-F1 (left) and activity accuracy (right)."),
    P(`${F("robustness")} looks at robustness in more detail. Every fusion model loses much of its activity accuracy when text disappears, a clear case of the “modality laziness” described by Wang et al. [@wang_hard]. Modality dropout more than doubles accuracy without text (from 0.19 to 0.41), and gated fusion retains the most (0.62). The image-only model is unaffected, but it is also unable to detect threats.`),
    ...FIG("robustness", "fig_robustness.png", "Robustness of ablation variants to text degradation: threat macro-F1 (top) and activity accuracy (bottom) with clean text, real Tesseract OCR and no text."),
    H3("Speed and size"),
    P(`${TB("latency")} and ${F("latency")} report single-frame latency on one CPU thread. The main model needs 7.3 ms with ONNX Runtime (6.0 ms with int8 weights), more than 100 frames per second on a single core, and gated fusion needs only 3.9 ms. ONNX Runtime is 1.2–2.2 times faster than eager PyTorch. Dynamic int8 quantisation shrinks the main model 1.8-fold and speeds inference up by 8–28%, with the largest gain for the Transformer-heavy ViT. Against these numbers, Tesseract takes 334 ms on a 720p frame, about 45 times the model. The remaining stages are cheap: change detection takes 3.4 ms, letterboxing 1.5 ms, tokenisation 0.02 ms, and rules with URL analysis 1.3 ms.`),
    ...TABLE("latency", "Mean single-frame latency (ms) on one CPU thread, batch size 1, 144×256 input and 1,024 text bytes.",
      ["Model (vision / fusion)", "Params", "ONNX size fp32 / int8", "PyTorch", "ONNX fp32", "ONNX int8"],
      [["TinyCNN / attention (main)", "1.28 M", "5.1 / 2.8 MB", "10.6", "7.3", "**6.0**"], ["TinyCNN / gated", "0.95 M", "3.7 / 2.3 MB", "7.8", "4.4", "**3.9**"],
        ["TinyCNN / concat", "0.92 M", "3.6 / 2.3 MB", "8.0", "4.6", "**3.9**"], ["MobileNetV3-Small / attention", "1.94 M", "7.6 / 5.3 MB", "13.7", "8.6", "**6.7**"],
        ["ViT-tiny / attention", "2.92 M", "11.4 / 4.1 MB", "19.9", "16.2", "**11.7**"], ["EfficientNet-B0 / attention", "5.11 M", "19.7 / 17.4 MB", "33.3", "15.4", "**14.2**"]],
      W([3.1, 1.0, 2.0, 1.1, 1.1, 1.1]), { align: [undefined, CENTER, CENTER, CENTER, CENTER, CENTER] }),
    ...FIG("latency", "latency.png", "Model latency by backbone, fusion and runtime (one CPU thread, symmetry-broken random weights whose file sizes match trained exports)."),
    H3("Explainability"),
    P(`${F("gradcam")} shows explanations produced by the «analyze» command. Word saliency picks out exactly the evidence a human analyst would point to: the lookalike address «chse.tk/billing/confirm» on a fake Chase login page, and the suspicious domain and wallet address on a giveaway scam. Grad-CAM shows that the activity head of the code-editor example focuses on the code area. For threats the image heat-maps are more diffuse, which is consistent with threat decisions being driven mainly by text, so the word-level explanation is the more useful one for users.`),
    ...FIG("gradcam", "gradcam_examples.png", "Grad-CAM heat-maps [@gradcam] with gradient×input word saliency for a phishing login (left), a giveaway scam (middle) and the activity of a code editor (right)."),
    H2("4.3 Validation or Testing"),
    P(`**Automated testing.** The software is covered by 65 automated tests. They check the generator (determinism, all 45 activity/threat/device combinations), the tokenizer, every combination of vision backbone and fusion module including single-modality variants, the masked loss, the URL analyser and rules, change detection, smoothing and hysteresis, redaction of personal data, asynchronous OCR, and a complete train → evaluate → export run with an ONNX parity check. The exported fp32 model matches PyTorch to within 4×10⁻⁶ in its logits, and the int8 model gives the same predicted class on all probe inputs.`),
    P(`**End-to-end session.** ${F("timeline")} plots the smoothed threat risk during the scripted 65-frame session with real OCR. Of the 65 frames, 52 (80%) were skipped as unchanged and served in about 3 ms each. Analysed frames took 392 ms on average (p95 565 ms), of which OCR accounted for 378 ms, the model for 6.8 ms and the rules for 0.9 ms. All five threats (a phishing e-mail, malicious shell commands, a giveaway scam, a malware download and a phishing login) raised an alert two frames after they appeared, each alert cleared after the screen changed, and none of the eight benign screens produced an alert.`),
    ...FIG("timeline", "fig_timeline.png", "Smoothed threat risk during the scripted session (13 screens × 5 frames, real Tesseract OCR). Shaded red bands are threat screens; filled markers were analysed, hollow markers were served from the cache."),
    ...TABLE("validation", "Results against the acceptance criteria of Table 4.",
      ["ID", "Criterion", "Result", "Met?"],
      [["R1", "Activity accuracy ≥ 95%", "1.000 (clean); 0.950 → 1.000 with real OCR after fine-tuning", "Yes"],
        ["R2", "Threat macro-F1 ≥ 0.90; recall ≥ 90% at ≤ 1% FPR", "F1 0.994; recall 99.3% at 1% FPR (clean); 0.992 with real OCR after fine-tuning", "Yes"],
        ["R3", "Measured OCR drop and a way to close it", "−5.0 / −2.9 points (activity / threat) vs ground-truth text; closed by OCR-aware fine-tuning", "Yes"],
        ["R4", "≤ 20 ms model, < 1 s changed frame, < 10 MB", "7.3 ms (6.0 ms int8); 392 ms mean, 565 ms p95; 5.1 MB (2.8 MB int8)", "Yes"],
        ["R5", "Reasons for every threat alert", "Rule and URL reasons attached; Grad-CAM and word saliency on request", "Yes*"],
        ["R6", "Prompt, stable alerts, no false alerts", "5/5 threats alerted after 2 frames; 0/8 false alerts; no flicker", "Yes"],
        ["R7", "Privacy by default", "No frames stored, no text in logs, no network calls (verified by tests)", "Yes"]],
      W([0.6, 3.0, 4.8, 0.8]), { align: [undefined, undefined, undefined, CENTER] }),
    P("*One alert in the session, for a malware download, was raised by the model alone without a matching rule, so it carried no textual reason. This is consistent with the design, since the model can detect threats the rules do not cover, but it shows that the model’s own evidence, such as word saliency, should be attached to alerts by default."),
    P("**Failures and unexpected outcomes.** Development produced several instructive failures, all of which were fixed and are covered by tests. (1) The strided byte stem learned threats very slowly. (2) Temperature scaling collapsed on the saturated validation set. (3) An early pipeline skipped unchanged frames entirely, so a static phishing page never reached the two frames needed for an alert. (4) A global cooldown suppressed a scam alert that followed a phishing alert. (5) In asynchronous mode, a new screen was briefly analysed with the previous screen’s text. (6) ONNX Runtime’s default dynamic quantisation made the model 2–3 times *slower*, because quantised convolutions are slow on this CPU. (7) Freshly initialised Transformer layers have identical weights, which the ONNX exporter de-duplicates; this made early benchmarks look unrealistically fast until the weights were perturbed."),
  ];
}

function section5() {
  return [
    H1("5 Discussion and Evaluation"),
    H2("5.1 Interpretation"),
    P("The results support all three hypotheses. **H1 (complementarity)** is supported by the ablations. Images alone identify the activity but not the threat, while text alone identifies the threat but collapses when OCR fails. The fused models combine both strengths, and the best fused model (gated fusion) outperforms text alone with real OCR (threat F1 0.947 against 0.911) as well as without text (activity accuracy 0.617 against 0.109). This matches the lesson from UI-understanding research that screens are inherently multimodal [@screen2words,@webui]."),
    P("**H2 (robustness)** is supported as well. The byte-level encoder loses only 4 points of threat F1 at a 10% character error rate, while regular expressions lose 34 points. With real OCR, the remaining errors are concentrated in a few classes, and OCR-aware fine-tuning removes them. This reflects a broader principle: a model should be trained on the inputs it will actually receive in deployment. The robustness of character-level models to spelling variation is consistent with earlier work [@zhang_charcnn,@canine]; our ablation adds the observation that *how* the byte sequence is downsampled matters even more than the byte representation itself."),
    P("**H3 (feasibility)** is clearly supported. The model needs milliseconds, and a changed frame needs under 0.6 s at the 95th percentile, with OCR responsible for over 95% of that time. The practical implication is that engineering effort is best spent on *not* running OCR (change detection, asynchronous OCR, OCR of changed regions only) rather than on shrinking a model that is already cheap. The ablations also reproduced two findings from the literature: ViTs need more data than CNNs [@vit], and multimodal networks under-use their weaker modality [@wang_hard], which modality dropout mitigates [@moddrop]."),
    P("Finally, these results must be read in light of the data. Train and test screens come from the same generator, and the rules were written by the same team as the text templates. The near-perfect clean-text scores should therefore be treated as an upper bound and as evidence that the system works end to end, not as an estimate of real-world accuracy. The more informative results are the *relative* ones: the ablations, the robustness curves and the latency breakdown."),
    H2("5.2 Originality and Added Value"),
    P("Compared with existing tools, the added value of ScreenGuard lies in its scope and its practicality. It analyses one stream (the screen) instead of separate channels, covers activity, explicit content and four kinds of cyber threat in a single 5 MB model, works without access to URLs, HTML or view hierarchies, and runs on a CPU with an explanation attached to each alert. The project also provides new evidence on several engineering questions that are rarely reported: how much OCR costs relative to the model, how quickly rules break under OCR noise compared with learned models, which fusion strategy works best at small data scales, and why the downsampling stem of a byte-level encoder matters. Finally, the synthetic generator with hard negatives and the masked multi-task loss give other researchers a reproducible starting point for building joint datasets from heterogeneous sources."),
    H2("5.3 Feasibility and Impact"),
    P("**Implementation.** On desktops the prototype already runs as it is: screen capture through «mss», OCR through Tesseract and inference through ONNX Runtime, which is available for Windows, macOS and Linux. On Android, screen capture is possible with the MediaProjection API after explicit user consent, and the on-device ML Kit text recogniser [@mlkit] can replace Tesseract; iOS offers ReplayKit broadcast extensions and the Vision framework for OCR. Because only the OCR text enters the model, swapping the OCR engine requires only a short OCR-aware fine-tuning run (§4.2), not a new architecture."),
    P("**Cost and scalability.** All inference happens on the user’s device, so the marginal cost per user is close to zero and there is no server infrastructure to scale. The model fits in 2.8–5.1 MB. To give an idea of energy use: if a screen is captured twice per second and 20% of frames change, about 0.4 frames per second need OCR and inference, which corresponds to roughly 15% of one CPU core while the screen is busy, and much less while the user is reading. Reducing OCR to changed regions would lower this further."),
    P("**Demand and sustainability.** Demand is evident from the losses reported in §1.1 and from the existing markets for parental-control, family-safety and endpoint-security products. Realistic routes to sustainability include a freemium consumer app (free warnings, paid family features), a licence for security or parental-control vendors who want a cross-application detection component, and a low-cost add-on for schools and small organisations. Impact can be measured with the indicators proposed in §1.5: recall at a fixed false-alarm budget, time to alert, CPU cost, and, in a deployment, the number of credentials or payments users chose *not* to submit after a warning."),
    H2("5.4 Limitations and Risks"),
    P(`${TB("risks")} lists the main limitations and risks, their effects, and how they can be reduced.`),
    ...TABLE("risks", "Limitations, risks and mitigations.",
      ["Limitation or risk", "Effect", "Mitigation"],
      [["Synthetic data only; templates and rules from the same authors", "Scores are an upper bound; rules-only results are optimistic", "Evaluate on Rico, WebUI and Phishpedia data and on consented real screenshots; independent rule review"],
        ["Single training seed per ablation variant", "Small differences (< 0.02) may be noise", "Repeat with several seeds and report confidence intervals"],
        ["English-only text; OCR dependence", "Lower recall on other languages, small fonts and low contrast", "Multilingual byte-level pretraining; platform OCR engines; OCR-aware fine-tuning"],
        ["Adversarial evasion (text inside images, new homoglyphs, threats split across screens)", "Missed detections", "Visual text recognition inside media, extended homoglyph tables, temporal models over screen sequences"],
        ["Explicit-content head untrained; potential demographic bias", "No explicit-content alerts yet; risk of unfair errors", "Ethically approved dataset, bias audits by subgroup; hash-matching services (not classifiers) for illegal material"],
        ["Misuse for covert surveillance", "Privacy violation, legal harm", "Local-only processing, visible start-up notice, no frame storage, usage policy for transparent use"],
        ["False alarms and alert fatigue", "Users ignore warnings", "Calibrated thresholds at a fixed FPR, hysteresis, per-label cooldowns, user feedback"],
        ["Platform restrictions on mobile capture", "Limited deployment on iOS", "Broadcast extensions with user consent; integration with existing safety features"]],
      W([3.0, 2.6, 4.0])),
  ];
}

function section6() {
  return [
    H1("6 Conclusion and Next Steps"),
    H2("6.1 Conclusion"),
    P("This project set out to answer whether a compact multimodal deep learning model can recognise screen activity and detect safety threats accurately and quickly enough to run continuously on a CPU-only device. Within the limits of synthetic evaluation, the answer is yes. ScreenGuard’s 1.28-million-parameter network reads both the pixels and the OCR text of a frame, recognises nine activities, and detects phishing, scams, malicious downloads and dangerous commands with a threat macro-F1 of 0.994 on clean text and 0.992 with real OCR after OCR-aware fine-tuning. It needs about 7 ms per frame. Together with change detection, explainable rules and calibrated temporal alerting, it forms a complete prototype that raised all five threat alerts in an end-to-end session, without false alerts, while skipping 80% of frames. Its main contributions are the integrated, privacy-preserving design, the evidence from the ablation study, and a reproducible data-generation and evaluation protocol."),
    H2("6.2 Recommendations"),
    ...numbered([
      "**Fine-tune on the deployed OCR engine’s output.** This single step removed the entire real-OCR gap (activity accuracy from 0.950 to 1.000, threat F1 from 0.971 to 0.992) at a cost of a few minutes of CPU time.",
      "**Choose the fusion strategy by data size.** Use gated fusion when fewer than about 5,000 labelled frames are available, and attention fusion when more data are available.",
      "**Keep the hybrid decision layer, but tune it on real data.** Rules provide explanations and a safety net, but they are brittle under OCR noise, so their weight should be set on real validation data.",
      "**Set alert thresholds at a fixed false-alarm budget**, such as 1% false positives, using calibrated probabilities, and keep hysteresis and per-label cooldowns to prevent alert fatigue.",
      "**Invest in avoiding OCR rather than in compressing the model**: change detection, asynchronous OCR and OCR of changed regions only. Deploy with ONNX Runtime, and evaluate int8 models on ARM devices where integer kernels give larger gains.",
      "**Attach the model’s own evidence**, such as word saliency, to alerts that are raised without a matching rule, so that every alert is explainable.",
    ]),
    H2("6.3 Next Steps"),
    P("**Immediate (next three months).** (1) Validate on real data: public datasets (Rico, WebUI, Phishpedia and PhishIntention screenshots) and a small set of screenshots collected with informed consent and ethics approval. (2) Train and audit the explicit-content head on an approved dataset, with per-group bias checks. (3) Build a desktop tray application around the existing pipeline and run a usability study of at least 15 users, measuring alert usefulness, false-alarm tolerance and CPU impact. (4) Repeat the ablations with several seeds and report confidence intervals."),
    P("**Longer term.** (1) Extend the system to more languages, including Korean, Kinyarwanda and French, through multilingual byte-level pretraining. (2) Distil a larger vision-language model into the compact student, to improve generalisation to unseen layouts. (3) Model sequences of screens, so that multi-step scams spanning several apps can be detected. (4) Explore federated or on-device learning from user feedback (“this was not a scam”) without uploading screens. (5) Seek partnerships with the university’s IT services, local banks or mobile-money operators, and security vendors, together with funding for a pilot deployment."),
  ];
}

function referencesSection() {
  const out = [H1("References")];
  citeOrder.forEach((k, i) => out.push(new Paragraph({
    tabStops: [{ type: TabStopType.LEFT, position: 567 }], indent: { left: 567, hanging: 567 }, alignment: AlignmentType.LEFT,
    spacing: { before: 0, after: 80, line: 252, lineRule: LineRuleType.AUTO },
    children: [new TextRun({ text: `[${i + 1}]`, size: 20, font: FONT }), new TextRun({ children: [new Tab()], size: 20, font: FONT }),
      new TextRun({ text: REFS[k], size: 20, font: FONT })],
  })));
  return out;
}

function appendices() {
  return [
    H1("Appendices"),
    P("The appendices contain supporting material referred to in the report: technical details (Appendix A), additional results (Appendix B) and the record of AI use (Appendix C). Source code, configurations and raw result files are available in the project repository."),
    H2("Appendix A Research Instruments and Technical Details"),
    P("**Label taxonomy.** Activities: web browsing, video streaming, messaging, e-mail, document editing, code editor, terminal, social media, file download. Explicit content: safe, suggestive, explicit. Threats: benign, phishing, scam, malware download, suspicious command."),
    ...TABLE("hyper", "Architecture details of the main model.",
      ["Component", "Configuration", "Parameters"],
      [["Input", "RGB 144×256 (letterboxed), up to 1,024 UTF-8 bytes (head 75% + tail 25%)", "–"],
        ["TinyCNN vision encoder", "Stem 3×3/2 (16) → IR-SE/2 (16) → IR/2 e4 (24) → IR (24) → IR-SE/2 k5 (40) → IR-SE k5 (40, 64) → IR-SE/2 k5 (96) → IR-SE k5 (96) → 1×1 (128); 5×8 = 40 tokens", "0.34 M"],
        ["Byte text encoder", "Embedding 260×64 → Conv1d k7 (128) → GELU → max-pool 8 → positions (130×128) → 2 pre-norm Transformer layers (4 heads, FFN 512)", "0.49 M"],
        ["Attention fusion", "3 task tokens + modality-type embeddings; 2 Transformer layers (4 heads, FFN 512); key-padding mask for absent text", "0.40 M"],
        ["Task heads", "LayerNorm → Linear 128 → GELU → Dropout → Linear (9 / 3 / 5 classes)", "0.05 M"],
        ["Total", "Masked multi-task loss with learned log-variances", "1.28 M"]],
      W([2.0, 6.2, 1.1]), { align: [undefined, undefined, CENTER] }),
    ...TABLE("rules", "Signature-rule coverage (45 rules) with examples.",
      ["Category", "Rules", "Examples of indicators"],
      [["Suspicious command", "23", "Encoded PowerShell; «curl | sh» from a raw IP; certutil/mshta/regsvr32 downloads; reverse shells; shadow-copy deletion; disabling antivirus; ClickFix “Win + R, Ctrl + V” lures; persistence via Run keys, scheduled tasks or cron"],
        ["Scam", "8", "Virus warning with a phone number; “send 0.5 BTC, receive 1 BTC”; gift-card payment demands; guaranteed returns; inheritance and lottery fees; “Mum, I lost my phone”"],
        ["Malware download", "7", "Double extensions («invoice.pdf.exe»); crack/keygen executables; fake codec or player updates; “enable macros”; risky attachment types; modded APKs"],
        ["Phishing", "7", "Account suspension threats; “verify your account”; artificial deadlines; parcel customs fees; copyright-appeal forms; requests for recovery phrases or card numbers"]],
      W([1.8, 0.8, 7.0]), { align: [undefined, CENTER, undefined] }),
    P("**Software and reproduction.** Python 3.13, PyTorch 2.15, ONNX Runtime 1.29, Tesseract 5.3.4 and Pillow. Main commands: «screenguard generate», «screenguard train --config configs/default.yaml», «screenguard evaluate», «screenguard export», «screenguard benchmark», «screenguard monitor --source screen --async-ocr» and «screenguard analyze shot.png --explain cam.png». The script «scripts/reproduce.sh» regenerates every number in this report with the exact seeds used."),
    H2("Appendix B Additional Results and Testing Evidence"),
    ...FIG("training", "fig_training.png", "Training curves of the main model: training loss (left) and validation metrics (right). The best checkpoint (epoch 6) reached a validation score of 0.9995."),
    ...TABLE("perclass", "Per-class threat results of the main model on the synthetic test set (clean text).",
      ["Class", "Precision", "Recall", "F1", "Support"],
      Object.entries(JSON.parse(fs.readFileSync(path.join(ROOT, "docs/results/test_main.json"))).model.threat.per_class).map(([c, v]) =>
        [c.replace("_", " ").replace(/^./, (x) => x.toUpperCase()), v.precision.toFixed(3), v.recall.toFixed(3), v.f1.toFixed(3), String(v.support)]),
      W([3.0, 1.5, 1.5, 1.5, 1.5]), { align: [undefined, CENTER, CENTER, CENTER, CENTER] }),
    P("**Latency with four threads.** With four threads, ONNX Runtime needs 4.5 ms (fp32) and 4.1 ms (int8) for the main model, and 2.8 ms for gated fusion. **Export parity.** The fp32 ONNX model differs from PyTorch by at most 4×10⁻⁶ in its logits; the int8 model by at most 0.056, with identical predicted classes on all probe inputs."),
    H2("Appendix C Partner Approval and AI Use Record"),
    P("**Partner approval.** No external or internal partner was involved, so no partner approval was required."),
    ...TABLE("airecord", "Record of AI tool use.",
      ["Tool", "Purpose", "Sections and outputs affected", "Verification performed"],
      [["Claude (Anthropic), via Claude Code", "Implementation assistance for the ScreenGuard code base (generator, models, training, evaluation, real-time pipeline, export)", "Software prototype; Sections 3–4", "65 automated tests; ONNX parity checks; end-to-end runs; review of the code by the team"],
        ["Claude (Anthropic)", "Running experiments, producing figures and tables", "Sections 4–5, Appendix B, all figures", "All numbers generated by scripts and stored in «docs/results/»; reproducible with «scripts/reproduce.sh»"],
        ["Claude (Anthropic)", "Drafting and editing report text", "All sections", "Reviewed by both team members; every reference and number checked against sources and result files"]],
      W([1.6, 2.8, 2.2, 3.0])),
  ];
}

// ------------------------------------------------------------------ assemble
const declaration = declarationPage();
const abstract = abstractPage();
const tocHeadingIndex = headings.length;
const body = [...section1(), ...section2(), ...section3(), ...section4(), ...section5(), ...section6()];
const refs = referencesSection();
const apps = appendices();
const toc = tocPage();

const header = new Header({ children: [new Paragraph({ alignment: RIGHT, spacing: { after: 0 },
  children: [new TextRun({ text: "2026 Research and Innovation Showcase |  Project Report", bold: true, size: 15, color: C.blue, font: FONT })] })] });
const footer = new Footer({ children: [new Paragraph({ alignment: RIGHT,
  children: [new TextRun({ text: "|  ", size: 16, color: "A6A6A6", font: FONT }), new TextRun({ children: [PageNumber.CURRENT], size: 22, font: FONT })] })] });

const doc = new Document({
  creator: `${PERSON.leader}, ${PERSON.member}`,
  title: "Real-Time Multimodal Screen Activity and Safety Threat Detection Using Deep Learning",
  description: "KDU Global Research Center 2026 Research and Innovation Showcase - Project Report",
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 }, paragraph: { spacing: { line: 276, lineRule: LineRuleType.AUTO } } },
      heading1: { run: { font: FONT, size: 36, bold: true, color: "000000" }, paragraph: { spacing: { before: 0, after: 200 }, keepNext: true } },
      heading2: { run: { font: FONT, size: 27, bold: true, color: "000000" }, paragraph: { spacing: { before: 260, after: 110 }, keepNext: true } },
      heading3: { run: { font: FONT, size: 23, bold: true, color: C.navy }, paragraph: { spacing: { before: 180, after: 70 }, keepNext: true } },
    },
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN, header: 567, footer: 567 } }, titlePage: true },
    headers: { default: header, first: new Header({ children: [new Paragraph({ children: [] })] }) },
    footers: { default: footer, first: new Footer({ children: [new Paragraph({ children: [] })] }) },
    children: [...coverPage(), ...declaration, ...abstract, ...toc, ...body, ...refs, ...apps],
  }],
});

fs.writeFileSync(path.join(ROOT, "report/assets/headings.json"), JSON.stringify(headings.map((h) => h.text), null, 1));
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "report/KDU_Project_Report_ScreenGuard.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, "references:", citeOrder.length, "headings:", headings.length);
});
