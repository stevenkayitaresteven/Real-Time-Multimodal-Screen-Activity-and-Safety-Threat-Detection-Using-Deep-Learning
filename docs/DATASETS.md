# Data: synthetic, real, and how to combine them

## Manifest format
Every dataset is a JSONL manifest, one record per screenshot:

```json
{"image": "images/000001.jpg", "text": "visible screen text", "activity": "email", "threat": "phishing", "nsfw": null}
```

* `image` is relative to the manifest's folder (absolute paths also work).
* `text` is the screen text. Use ground truth when you have it. Add `ocr_text` produced by
  `screenguard ocr-manifest` to train or evaluate on real OCR output (`data.text_field=ocr_text`).
* **Any label may be `null`.** The masked multi-task loss trains each head only on samples labelled for it, so
  datasets with different label sets can be mixed freely. List several manifests in `data.train_manifests`.

## Synthetic data (built in)
```bash
screenguard generate --out data/synth --train 8000 --val 1000 --test 1500 --width 640 --height 360
```
This renders labelled desktop and mobile screens for all activities and threats (see
[ARCHITECTURE.md §7](ARCHITECTURE.md#7-synthetic-data-generator-screenguarddatasyntheticpy)). It needs no downloads
and runs in about a minute on 4 cores.

## Real datasets to plug in
The public datasets below map onto this project's tasks. Check each dataset's licence and terms before use. Most are
for research use only.

| Task | Dataset | What it provides | How to use it here |
|---|---|---|---|
| Activity | **Rico** (Deka et al., UIST 2017) | ~70k Android app screenshots plus view hierarchies and app metadata | map app category → activity label; view-hierarchy text ≈ OCR text |
| Activity | **WebUI** (Wu et al., CHI 2023) | large-scale web page screenshots with accessibility text | `web_browsing` / `social_media` / `video_streaming` by domain |
| Activity | **Screen2Words** (Wang et al., UIST 2021) | screen summaries for Rico screens | weak labels for activity; captions as auxiliary text |
| Phishing | **Phishpedia** (Lin et al., USENIX Security 2021), **PhishIntention** (Liu et al., USENIX Security 2022), **VisualPhishNet** (Abdelnabi et al., CCS 2020) | phishing and benign webpage screenshots with URLs and brand labels | `threat=phishing/benign`, `activity=web_browsing`; also extend `analysis/brands.py` with their brand lists |
| Phishing URLs | PhishTank / OpenPhish feeds | live phishing URLs | screenshot **only inside an isolated sandbox VM**; or render URLs into the synthetic browser chrome |
| SMS / chat scams | **SMS Spam Collection** (UCI) and local smishing reports | labelled SMS text | render into the synthetic chat app (`text_corpus.py`) or train with `modalities=[text]` |
| Email phishing | Enron (ham) plus public phishing-email corpora | labelled email text | render into the synthetic email client |
| Commands | Sigma rules, MITRE ATT&CK procedure examples, Atomic Red Team | documented malicious command lines | add to `terminal_suspicious()` (keep placeholder IPs from RFC 5737) |
| NSFW | public research NSFW image datasets (e.g. LSPD) or an in-house, legally reviewed set | adult/suggestive/safe images | ImageFolder at `data/nsfw/{train,val}/{safe,suggestive,explicit}/`, then set `data.nsfw_root=data/nsfw` |

### NSFW training
```bash
screenguard train --set data.nsfw_root=data/nsfw data.nsfw_composite_p=0.7 train.out_dir=runs/with_nsfw
```
With probability `nsfw_composite_p`, each NSFW-dataset image is pasted into a synthetic screen (video player, article,
social feed, chat). The model then learns what such content looks like *on a screen*, surrounded by UI chrome and
scaled down, rather than only as a full-frame photo. Until a checkpoint has seen NSFW labels, the pipeline reports
`nsfw: {"available": false}` instead of emitting uncalibrated guesses.

Mandatory safeguards for NSFW data:
* Work only with adult content that is legal to possess in your jurisdiction, under your institution's ethics approval.
* A classifier is **not** an appropriate tool for child sexual abuse material (CSAM). Never collect or train on such
  material. Production systems detect known CSAM with hash-matching services operated by organisations such as NCMEC
  or the IWF, and report it through legal channels.
* Store the data encrypted, restrict access, and never commit it to git (the default `.gitignore` excludes `data/`).

### Collecting your own screen recordings
If you record real sessions (yours or volunteers'), get informed consent, record on dedicated accounts, avoid
capturing third parties' personal data, and keep raw frames encrypted. Label with the same taxonomy (`taxonomy.py`).
Run `screenguard ocr-manifest` to attach OCR text. A few thousand real labelled frames, mixed with synthetic data, are
far more valuable than synthetic data alone.

## Recommended training recipe for real deployments
1. Pre-train on synthetic data (all tasks except NSFW).
2. Add real activity screenshots (Rico/WebUI or your own), phishing screenshots, and NSFW data (composited).
3. Fine-tune with `data.text_field=ocr_text`, so the text pathway sees the output of the OCR engine you will deploy.
4. Calibrate on a real validation set. Choose alert thresholds for a target false-alarm rate, e.g. 1 % FPR, using the
   `tpr_at_1pct_fpr` metric. In real use threats are rare (well under 1 % of frames), so precision at low FPR matters
   more than accuracy.
