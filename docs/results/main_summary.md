| run | activity_acc | activity_f1 | threat_acc | threat_f1 | threat_auroc | threat_tpr@1%fpr | threat_ece | model_threat_f1 | rules_threat_f1 | hybrid_threat_f1 | hybrid_fpr | hybrid_recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic test / clean | 1.0000 | 1.0000 | 0.9960 | 0.9936 | 0.9978 | 0.9930 | 0.0033 | 0.9936 | 0.8665 | 0.9951 | 0.0011 | 0.9982 |
| synthetic test / ocr_noise_5% | 0.9980 | 0.9979 | 0.9880 | 0.9845 | 0.9956 | 0.9772 | 0.0112 | 0.9845 | 0.7151 | 0.9741 | 0.0172 | 0.9772 |
| synthetic test / ocr_noise_10% | 0.9893 | 0.9897 | 0.9633 | 0.9524 | 0.9871 | 0.9264 | 0.0315 | 0.9524 | 0.5238 | 0.9417 | 0.0258 | 0.9352 |
| synthetic test / ocr_noise_20% | 0.9527 | 0.9553 | 0.8800 | 0.8320 | 0.9428 | 0.7128 | 0.1071 | 0.8320 | 0.2893 | 0.8288 | 0.0603 | 0.8056 |
| synthetic test / no_text | 0.7873 | 0.7722 | 0.6260 | 0.2467 | 0.6728 | 0.0333 | 0.1643 | 0.2467 | 0.1530 | 0.2467 | 0.0495 | 0.0981 |
| 720p test / ground-truth text | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0017 | 1.0000 | 0.8469 | 1.0000 | 0.0000 | 1.0000 |
| 720p test / Tesseract OCR | 0.9500 | 0.9432 | 0.9800 | 0.9711 | 0.9965 | 0.9735 | 0.0194 | 0.9711 | 0.8041 | 0.9750 | 0.0107 | 0.9823 |

Tesseract on 300 1280x720 frames: mean character similarity to ground truth 0.854, latency mean 373.8 ms (p95 624.7 ms), empty outputs 0.

