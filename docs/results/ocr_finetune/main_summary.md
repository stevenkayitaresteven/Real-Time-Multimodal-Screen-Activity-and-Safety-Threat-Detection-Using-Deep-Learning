| run | activity_acc | activity_f1 | threat_acc | threat_f1 | threat_auroc | threat_tpr@1%fpr | threat_ece | model_threat_f1 | rules_threat_f1 | hybrid_threat_f1 | hybrid_fpr | hybrid_recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic test / clean | 1.0000 | 1.0000 | 0.9953 | 0.9927 | 0.9972 | 0.9930 | 0.0029 | 0.9927 | 0.8665 | 0.9941 | 0.0022 | 0.9982 |
| synthetic test / ocr_noise_5% | 0.9993 | 0.9993 | 0.9867 | 0.9829 | 0.9960 | 0.9790 | 0.0098 | 0.9829 | 0.7151 | 0.9732 | 0.0172 | 0.9772 |
| synthetic test / ocr_noise_10% | 0.9900 | 0.9902 | 0.9627 | 0.9512 | 0.9859 | 0.9264 | 0.0302 | 0.9512 | 0.5238 | 0.9405 | 0.0269 | 0.9422 |
| synthetic test / ocr_noise_20% | 0.9520 | 0.9543 | 0.8967 | 0.8571 | 0.9476 | 0.7618 | 0.0838 | 0.8571 | 0.2893 | 0.8507 | 0.0431 | 0.8074 |
| synthetic test / no_text | 0.7867 | 0.7778 | 0.6093 | 0.2599 | 0.6357 | 0.0385 | 0.1567 | 0.2599 | 0.1530 | 0.2599 | 0.0936 | 0.1506 |
| 720p test / ground-truth text | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0016 | 1.0000 | 0.8469 | 1.0000 | 0.0000 | 1.0000 |
| 720p test / Tesseract OCR | 1.0000 | 1.0000 | 0.9933 | 0.9919 | 0.9967 | 0.9912 | 0.0027 | 0.9919 | 0.8041 | 0.9958 | 0.0053 | 1.0000 |

Tesseract on 300 1280x720 frames: mean character similarity to ground truth 0.854, latency mean 373.8 ms (p95 624.7 ms), empty outputs 0.

