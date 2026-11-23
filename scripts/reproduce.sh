#!/usr/bin/env bash
# Reproduce every number in docs/EXPERIMENTS.md (exact seeds and paths).
# Runtime on a 4-core CPU: roughly 2.5-3 hours in total, mostly training.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Synthetic data (seeds fixed per split)
python - <<'EOF'
from screenguard.data.generate import generate_split as g
g("data/synth/train", 8000, 101)
g("data/synth/val", 1000, 202)
g("data/synth/test", 1500, 303)
g("data/synth/train2", 8000, 404)
g("data/synth/test_hires", 300, 505, width=1280, height=720)    # real-OCR test set
g("data/synth/train_hires", 3000, 606, width=1280, height=720)  # OCR-aware fine-tuning data
EOF
head -4000 data/synth/train/manifest.jsonl > data/synth/train/manifest_4k.jsonl   # ablation budget

# 2. Real OCR (Tesseract) for the high-resolution sets
python -m screenguard ocr-manifest --manifest data/synth/test_hires/manifest.jsonl --out data/synth/test_hires/manifest_ocr.jsonl
python -m screenguard ocr-manifest --manifest data/synth/train_hires/manifest.jsonl --out data/synth/train_hires/manifest_ocr.jsonl
head -2500 data/synth/train_hires/manifest_ocr.jsonl > data/synth/train_hires/ft_train.jsonl
tail -500 data/synth/train_hires/manifest_ocr.jsonl > data/synth/train_hires/ft_val.jsonl

# 3. Main model (§1) and OCR-aware fine-tuning (§2)
python -m screenguard train --config configs/default.yaml
python scripts/evaluate_main.py --checkpoint runs/main/best.pt
python -m screenguard train --config configs/finetune_ocr.yaml
python scripts/evaluate_main.py --checkpoint runs/ocr_finetune/best.pt --out docs/results/ocr_finetune

# 4. Ablations (§3)
python scripts/run_ablations.py --epochs 3

# 5. Export, latency benchmark and end-to-end session (§4) -- run on an otherwise idle machine
python -m screenguard export --checkpoint runs/ocr_finetune/best.pt --out exports/screenguard_ocrft.onnx
python -m screenguard export --checkpoint runs/main/best.pt --out exports/screenguard.onnx
python -m screenguard benchmark --variants tiny_cnn/attention tiny_cnn/gated tiny_cnn/concat \
    mobilenet_v3_small/attention vit_tiny/attention efficientnet_b0/attention --threads 1 4 --out docs/results/benchmark.json
python -m screenguard monitor --model exports/screenguard.onnx --source synthetic:5 --ocr tesseract --fps 0 \
    | tee docs/results/monitor_demo.txt

# 6. Figures
python scripts/make_figures.py
