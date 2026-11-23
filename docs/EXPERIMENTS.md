# Experiments

All numbers in this file were produced by the scripts in this repository on a **4-core CPU without a GPU**.
[`scripts/reproduce.sh`](../scripts/reproduce.sh) regenerates all of them with the exact seeds used. Raw JSON results live in [`docs/results/`](results/) and figures in
[`docs/figures/`](figures/).

> **Read these numbers as a sanity check of the system, not as real-world performance.** All train and test screens
> come from the same synthetic generator, and the signature rules were written by the same author as the synthetic
> text templates. Both the model scores and especially the rules-only scores are therefore optimistic. The protocol
> below is what should be re-run on real data (see [DATASETS.md](DATASETS.md)).

## Setup
| | |
|---|---|
| Data | synthetic, 640×360 desktop / 360×640 mobile (≈13 % mobile). Train 16 000 (main model) or 4 000 (ablations), val 1 000, test 1 500, each split from a different seed |
| Text | ground-truth visible text in reading order. During training, 3 % simulated OCR character noise |
| Input | letterboxed 144×256 RGB, byte text up to 1 024 bytes (head + tail truncation) |
| Optimiser | AdamW, lr 2e-3, weight decay 0.05, 5 % warm-up then cosine, batch 64, label smoothing 0.05, grad-clip 1.0 |
| Loss | masked multi-task CE with learned uncertainty weights |
| Calibration | per-task temperature fitted on val |
| Metrics | accuracy, macro-F1, expected calibration error (ECE). For threats also binary any-threat AUROC, recall/FPR at 0.5, and TPR at 1 % FPR |


## 1. Main model
`configs/default.yaml`: TinyCNN + byte n-gram CNN/Transformer + attention fusion, **1.28 M parameters**, trained on
16 000 screens for 6 epochs (29 min on the CPU).

```bash
screenguard train --config configs/default.yaml
python scripts/evaluate_main.py --checkpoint runs/main/best.pt
```

### 1.1 Held-out synthetic test set (1 500 screens)
| Condition | Activity acc | Threat acc | Threat macro-F1 | Any-threat AUROC | TPR @ 1 % FPR | Threat ECE |
|---|---|---|---|---|---|---|
| clean text | **1.000** | **0.996** | **0.994** | 0.998 | 0.993 | 0.003 |
| 5 % OCR noise | 0.998 | 0.988 | 0.985 | 0.996 | 0.977 | 0.011 |
| 10 % OCR noise | 0.989 | 0.963 | 0.952 | 0.987 | 0.926 | 0.032 |
| 20 % OCR noise | 0.953 | 0.880 | 0.832 | 0.943 | 0.713 | 0.107 |
| no text (OCR failure) | 0.787 | 0.626 | 0.247 | 0.673 | 0.033 | 0.164 |

Per-class F1 on clean text: phishing 0.991, scam 0.994, malware download 1.000, suspicious command 0.985, benign
0.998. Desktop (1 299 screens) and mobile (201) performance is the same.

Observations:
* **Graceful degradation under OCR noise.** At a 10 % character error rate, threat F1 drops by only 4 points. This
  comes from the byte-level encoder plus OCR-noise augmentation during training.
* **Threat detection is fundamentally textual.** Without text, threat F1 collapses to 0.25, and only visually
  distinctive threats such as full-screen tech-support warnings remain detectable. Activity recognition keeps 79 %
  accuracy from pixels alone. Reliable OCR is therefore a hard requirement for the threat task.

### 1.2 Decision fusion: model vs rules vs hybrid (threat macro-F1)
| Condition | Model only | Rules only | Hybrid (noisy-OR) | Hybrid FPR | Hybrid recall |
|---|---|---|---|---|---|
| clean text | 0.994 | 0.867 | **0.995** | 0.11 % | 99.8 % |
| 5 % OCR noise | **0.985** | 0.715 | 0.974 | 1.7 % | 97.7 % |
| 10 % OCR noise | **0.952** | 0.524 | 0.942 | 2.6 % | 93.5 % |
| 20 % OCR noise | **0.832** | 0.289 | 0.829 | 6.0 % | 80.6 % |
| real Tesseract OCR (720p) | 0.971 | 0.804 | **0.975** | 1.1 % | 98.2 % |

* Regex rules are **brittle under OCR noise**: they fall from 0.87 to 0.29 at 20 % noise, while the learned model
  keeps 0.83.
* The hybrid wins on clean and real-OCR text, and its main value is **explanations**: every alert lists the rule and
  URL reasons behind it. Under heavy synthetic noise, rule false positives make the hybrid slightly worse than the
  model alone. A deployment should therefore tune `rule_scale` (in `fuse_threat`) on its own OCR engine's output.
* Rules-only scores on synthetic data are optimistic, because the rules and the text templates share an author.

### 1.3 Real OCR instead of ground-truth text (300 frames at 1280×720)
Tesseract 5.3 (`--psm 11`, dark-theme inversion, upscaling) reaches a mean character similarity of **0.854** to the
ground-truth text, at **374 ms** per frame on average (p95 625 ms) with 4 OCR processes in parallel.

| Text source | Activity acc | Threat macro-F1 | Any-threat AUROC |
|---|---|---|---|
| ground-truth text | 1.000 | 1.000 | 1.000 |
| Tesseract OCR | 0.950 | 0.971 | 0.997 |

The OCR gap is concentrated in specific classes: **terminal** activity recall falls to 0.70, document editing to
0.86, and suspicious-command recall to 0.89. Tesseract turns dark-theme shell prompts such as `user@host:~/project$`
into fragments (`admir`, `rminal`) that the model never saw in training. Section 2 closes this gap with OCR-aware
fine-tuning.

![Confusion matrices](figures/confusion_activity.png)
![Threat confusion matrix](figures/confusion_threat.png)

## 2. OCR-aware fine-tuning (closing the real-OCR gap)
The main model was trained on ground-truth screen text. To adapt it to the deployed OCR engine, 3 000 additional
1280×720 screens were rendered (new seed) and run through Tesseract. The model was then fine-tuned from
`runs/main/best.pt` for 2 epochs at lr 5e-4, on 2 500 of them (Tesseract text) mixed with 8 000 original screens
(ground-truth text). The remaining 500 joined the validation set. Evaluation uses the separate, unseen 300-frame 720p
test set from §1.3.

```bash
# data + OCR steps: see scripts/reproduce.sh (seed 606, 1280x720, Tesseract, 2500/500 split)
screenguard train --config configs/finetune_ocr.yaml
python scripts/evaluate_main.py --checkpoint runs/ocr_finetune/best.pt --out docs/results/ocr_finetune
```

| 720p test with **real Tesseract OCR** | Main model | + OCR-aware fine-tune |
|---|---|---|
| Activity accuracy | 0.950 | **1.000** |
| — terminal recall | 0.704 | **1.000** |
| — document-editing recall | 0.861 | **1.000** |
| Threat macro-F1 (model) | 0.971 | **0.992** |
| — suspicious-command recall | 0.889 | **1.000** |
| Threat macro-F1 (hybrid) | 0.975 | **0.996** |
| Any-threat false-positive rate (hybrid) | 1.07 % | **0.53 %** |
| Synthetic test, clean text: threat macro-F1 | 0.994 | 0.993 |
| Synthetic test, 20 % OCR noise: threat macro-F1 | 0.832 | 0.857 |

Fine-tuning on the deployed OCR engine's output **closes the OCR gap with no regression on clean text**, at a cost
of only ~7 minutes of CPU training. This is the single most important step when moving to a new OCR engine, for
example Android ML Kit or Apple Vision on phones. `exports/screenguard_ocrft.onnx` is the recommended deployment
model. The caveat is that the fine-tuning and test screens come from the same generator, so on real screens the same
recipe must be applied with real OCR'd frames.



<!-- ABLATIONS -->

## 4. Real-time performance
```bash
screenguard benchmark --variants tiny_cnn/attention tiny_cnn/gated tiny_cnn/concat mobilenet_v3_small/attention vit_tiny/attention efficientnet_b0/attention --threads 1 4
```

<!-- BENCHMARK -->

### End-to-end session (`screenguard monitor --source synthetic:5 --ocr tesseract`)
This scripted session has 13 screens, each shown for 5 frames: 8 benign screens and 5 threats. Full output is in
[`results/monitor_demo.txt`](results/monitor_demo.txt).

* **52 of 65 frames (80 %) were skipped** as unchanged: no OCR and no model call, ~3 ms each.
* Changed frames cost 392 ms on average. **OCR accounts for 378 ms of that, the model for 6.8 ms**, preprocessing for
  3 ms, and rules for 0.9 ms.
* **All 5 threats raised an alert** (phishing email, malicious shell commands, giveaway scam, malware download,
  phishing login), each 2 frames after it appeared, as required by the hysteresis `min_on=2`. Each alert came with
  reasons, e.g. *"coinbase1-support.net: mentions 'Coinbase' but is not an official Coinbase domain"*. There were
  **0 false alerts** on the 8 benign screens, and every alert cleared 2 frames after its screen went away.
* With `--async-ocr`, the visual pathway runs at the capture rate regardless of OCR, and text-based verdicts follow
  one OCR cycle (~0.4 s) later.

### Export
| Artifact | Size | Max abs. logit diff vs PyTorch | Arg-max agreement |
|---|---|---|---|
| ONNX fp32 | 5.07 MB | 4 × 10⁻⁶ | 100 % |
| ONNX int8 (MatMul/Gemm weights) | 2.77 MB | 0.056 | 100 % |

## 5. Engineering findings worth knowing
1. **Byte-text stem.** The first version of the text encoder used one strided convolution (kernel 16, stride 8). It
   must learn every keyword separately for each of 8 alignments. Replacing it with a stride-1 n-gram convolution
   plus max-pooling, which is shift-invariant, raised text-only threat F1 after 3 epochs on 2 000 screens from 0.36 to
   0.57, and activity accuracy from 88 % to 99 %, at the same cost. Section 3 shows the effect at full scale.
2. **Temperature scaling on a saturated validation set is ill-posed.** With ~100 % validation accuracy, NLL keeps
   falling as T → 0 (the fit returned T = 0.08). That would make the model extremely over-confident on real,
   out-of-distribution screens, so temperatures are clamped to [0.25, 10].
3. **Frame skipping must still feed the temporal logic.** An early version skipped unchanged frames entirely, so a
   static phishing page contributed only one observation and never satisfied the "2 consecutive frames" alert rule.
   Unchanged frames now re-feed the cached prediction into the smoother and alert state machine.
4. **Alert cooldowns must be per label.** A global cooldown suppressed alerts for a malicious command and a scam page because a
   phishing alert had fired shortly before.
5. **int8 is not automatically faster.** ONNX Runtime's default dynamic quantisation also converts convolutions, and
   on this x86 CPU `ConvInteger` made the model **2–3× slower** than fp32. Quantising only MatMul/Gemm keeps
   accuracy (100 % arg-max agreement) and halves the file size, but gives no meaningful speed-up for this small,
   convolution-heavy model. Larger gains need static QDQ quantisation, or ARM CPUs with int8 dot-product
   instructions.
6. **PyTorch eager dynamic quantisation breaks `nn.TransformerEncoder`.** The quantised `Linear.weight` becomes a
   method, which trips the encoder's fast-path checks, and the API is being deprecated. ONNX Runtime is the
   recommended quantised path.
7. **Tesseract OCR should run single-threaded** (`OMP_THREAD_LIMIT=1`) next to the model or in parallel workers.
   Otherwise OpenMP oversubscription made individual calls exceed 20 s under load.

