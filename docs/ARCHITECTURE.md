# Architecture and design decisions

This document explains *why* the system is built the way it is. The README covers *how* to use it.

## 1. Problem decomposition

One screen frame answers three questions at once:

| Task | Question | Classes | Main evidence |
|---|---|---|---|
| **Activity** | What is the user doing? | 9 (web browsing, video, messaging, email, document editing, code editor, terminal, social media, file download) | Visual layout |
| **NSFW** | Is explicit content visible? | safe / suggestive / explicit | Visual content of media regions |
| **Threat** | Is there a cyber-safety risk on screen? | benign, phishing, scam, malware download, suspicious command | Text: URLs, wording, file names, commands |

The tasks need different modalities: activity is mostly visual, NSFW is purely visual, and threats are mostly textual.
That is why the system is **multimodal**: a single CNN cannot read a lookalike domain at 256 px width, and a text model
cannot tell a video player from a document.

## 2. Pipeline

```
 capture (mss / video / folder)          ~5-15 ms
   │
   ▼
 change detector (dHash + pixel diff)    <1 ms ── unchanged ──► reuse last result
   │ changed
   ├──► OCR (Tesseract, sync or async)   200-1500 ms (the dominant cost; runs on change only)
   │      └► byte tokenizer
   ├──► letterbox 144x256 + normalise     ~2 ms
   ▼
 ScreenThreatNet (vision + text + fusion + 3 heads)   ~5-20 ms on 1 CPU thread
   │
   ├──► temperature scaling (calibrated probabilities)
   ├──► rules + URL analysis ──► decision fusion (noisy-OR)        <2 ms
   ▼
 EMA smoothing ─► alert hysteresis (on/off thresholds, min frames, cooldown) ─► console / JSONL / UI
```

### Real-time strategy
On a laptop CPU the neural model is cheap (single-digit milliseconds), so the real bottleneck is **OCR**. The design
attacks this in three ways:

1. **Skip unchanged frames.** Most of the time the screen is static. A 256-bit difference hash plus a 64×36 mean
   absolute difference detects meaningful change in under a millisecond. Unchanged frames reuse the previous result
   with no OCR and no model call.
2. **Asynchronous OCR** (`--async-ocr`). OCR runs on a background thread with "latest frame wins" semantics. The model
   always uses the most recent finished OCR result, so the visual path stays responsive at the capture frame rate even
   when OCR takes a second.
3. **Byte-level text input.** There is no tokenizer vocabulary to load, and the text encoder downsamples bytes 8× with
   a strided convolution, so even 1 KB of screen text costs only 128 Transformer tokens.

## 3. Model: ScreenThreatNet

### Vision encoders (`screenguard/models/vision.py`)
All encoders return a **token grid** `[B, N, D]` instead of only a pooled vector. Attention fusion can then use spatial
information (where a dialog box is), and Grad-CAM works for every backbone.

| Encoder | Params (encoder) | Notes |
|---|---|---|
| `tiny_cnn` | 0.34 M | MobileNetV3-style inverted residuals + squeeze-excite + hard-swish; downsamples early; trains from scratch |
| `mobilenet_v3_small` | ~0.9 M | torchvision; can load ImageNet weights (`model.pretrained=true`) |
| `vit_tiny` | ~1.8 M | patch-16 ViT with interpolated position embeddings, so phones and laptops can use different input sizes |
| `efficientnet_b0`, `resnet18`, `convnext_tiny` | larger | for accuracy-oriented experiments |
| `timm:<name>` | any | e.g. `timm:mobilevitv2_050`, `timm:efficientvit_b0` |

Input frames are **letterboxed** (aspect-preserving resize plus padding) to 144×256. Squashing a 9:19.5 phone screen into
16:9 would distort exactly the layout cues the activity head relies on.

### Text encoder (`screenguard/models/text.py`)
Byte embedding → strided Conv1d (kernel 16, stride 8) → learned positions → 2-layer pre-norm Transformer.

Why bytes and not WordPiece:
* OCR errors (`paypa1`, `rn`→`m`) fragment sub-word tokens unpredictably; byte models degrade gracefully.
* Phishing relies on character-level tricks (homoglyphs, punycode, extra hyphens).
* There is no vocabulary file to ship, so the tokenizer is about 30 lines of code and trivially portable to mobile.

Long screens are truncated **head + tail** (75 % / 25 %), so the browser chrome (URL bar) and the bottom of the screen
(download bar, chat input, latest message) both survive truncation.

### Fusion (`screenguard/models/fusion.py`)
* `concat`: late fusion of pooled vectors, followed by an MLP. This is the baseline.
* `gated`: Gated Multimodal Unit (Arevalo et al., 2017). A sigmoid gate weighs each modality per feature dimension.
* `attention` (default): a single-stream Transformer over `[task tokens | visual tokens | text tokens]` with
  modality-type embeddings. **Each task owns a query token**, so the threat head can attend to URL bytes while the
  activity head attends to layout tokens. This is a lightweight form of task-specific attention.

**Missing modalities are first-class.** Video frames often have no text, and OCR can fail. The fusion modules mask
absent text tokens (attention fusion) or substitute a learned "missing" embedding (pooled fusions). During training,
**modality dropout** hides one modality per sample with probability 0.15, which forces both pathways to be useful on
their own.

### Heads and loss
One MLP head per task. The loss is a **masked multi-task cross-entropy**. Real data is heterogeneous: a screenshot
corpus has activity labels, an NSFW corpus has only NSFW labels, and a phishing corpus has only threat labels. Each
sample therefore carries `-100` for tasks it isn't annotated for. Task weights are **learned** with homoscedastic
uncertainty (Kendall, Gal & Cipolla, CVPR 2018): `L = Σ exp(-s_t)·L_t + s_t`.

After training, a **temperature** per task is fitted on the validation set (Guo et al., 2017). Alert thresholds such as
"risk ≥ 0.7" are only meaningful on calibrated probabilities.

## 4. Neuro-symbolic decision layer (`screenguard/analysis/`)

The neural model is combined with explainable, hand-crafted evidence:

* **URL analysis**: extraction from OCR text, registered-domain parsing, brand lookalike detection (homoglyph
  normalisation plus Damerau-Levenshtein distance against a protected-brand list, in the spirit of reference-based
  detectors like Phishpedia), raw-IP hosts, punycode, abused TLDs, shorteners, executable downloads.
* **Signature rules** (Sigma-style regexes) for encoded PowerShell, `curl | sh` from raw IPs, LOLBins (`certutil`,
  `mshta`, `regsvr32`), reverse shells, shadow-copy deletion, ClickFix "press Win+R and paste" lures, double file
  extensions, cracked-software executables, tech-support and crypto-giveaway scams, and parcel smishing.

For each threat class `c`, model probability `p_c` and rule evidence `r_c` are combined with a noisy-OR:
`q_c = 1 − (1 − p_c)(1 − r_c)` and `q_benign = p_benign · Π(1 − r_c)`, then renormalised.

Why bother when the model already learns these cues?
* Rules generalise to **novel** lookalike domains and commands the model never saw.
* The model can **override weak rule false positives**. For example, the official `rustup` installer pipes `curl`
  into `sh`. It gets only weak rule evidence (0.40), which a confident benign model prediction outweighs.
* Every alert comes with **human-readable reasons** ("domain 'paypa1-secure.net' imitates PayPal").

## 5. Temporal logic (`screenguard/realtime/smoothing.py`)
Class probabilities are smoothed with an EMA. Alerts follow a hysteresis state machine:
* raise only when risk ≥ `on` for `min_on` consecutive analysed frames;
* clear only when risk < `off` (< `on`) for `min_off` frames;
* apply a cooldown between repeated alerts of the same kind.

This prevents alert flicker while scrolling, the main usability failure of per-frame classifiers.

## 6. Edge deployment
* `screenguard export` produces ONNX fp32 plus **dynamic int8** models and verifies both numerically against PyTorch.
* ONNX Runtime runs on Windows, macOS, Linux, Android (Java/Kotlin API), iOS (Objective-C/Swift API) and browsers
  (onnxruntime-web).
* OCR on mobile: use the platform engines (Android ML Kit text recognition, Apple Vision `VNRecognizeTextRequest`).
  They are faster and more accurate than Tesseract on phones. Only their text output enters the model, so they can be
  swapped in without retraining (retraining or fine-tuning on their output style is still recommended).
* Screen capture on mobile needs platform APIs (Android `MediaProjection`, iOS ReplayKit broadcast extension) and
  explicit user consent.

## 7. Synthetic data generator (`screenguard/data/synthetic.py`)
Joint activity/threat/NSFW screen datasets do not exist publicly, and collecting real screen recordings raises privacy
issues. The generator procedurally renders desktop (Windows/macOS/Linux-style chrome) and mobile screens for all nine
activities with randomised themes, fonts, DPI scale, window placement, distractor windows and notifications. Each frame
comes with the visible text in reading order; text hidden behind overlays is removed, mimicking what OCR would see.

Design choices that keep the task non-trivial:
* **Hard negatives**: legitimate security notifications, official `curl | sh` installers, signed `.exe` downloads from
  vendor sites, benign hyphenated domains and finance articles.
* **Visually identical threats**: about 25 % of phishing login pages have no warning banner, so only the URL betrays
  them.
* **OCR noise augmentation** (`ocr_noise_p`) during training: confusable characters, merged words, dropped lines.
* **Screen-context compositing** for image datasets (`ImageFolderComposite`): external images, such as an NSFW
  dataset, are pasted into the media slot of a rendered video player, article or social feed. This closes the domain
  gap between curated photos and how content actually appears on screen.

Synthetic accuracy is an **upper bound**. The generator is a development and pre-training tool, and real-world
evaluation needs real screens (see [DATASETS.md](DATASETS.md)).
