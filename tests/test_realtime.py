import numpy as np
import pytest
from PIL import Image

from screenguard.privacy import redact
from screenguard.realtime.change_detector import ChangeDetector
from screenguard.realtime.smoothing import AlertStateMachine, EMASmoother


def test_change_detector_skips_identical_frames():
    cd = ChangeDetector(max_interval_s=100)
    a = Image.new("RGB", (320, 180), (240, 240, 240))
    b = a.copy()
    b.paste((10, 10, 10), (0, 0, 200, 120))
    assert cd.update(a, 0.0)[0] is True
    assert cd.update(a.copy(), 1.0)[0] is False
    assert cd.update(b, 2.0)[0] is True
    assert cd.update(b, 200.0)[0] is True  # forced refresh after max_interval_s


def test_ema_smoother():
    s = EMASmoother(0.5)
    assert np.allclose(s.update("x", np.array([1.0, 0.0])), [1.0, 0.0])
    assert np.allclose(s.update("x", np.array([0.0, 1.0])), [0.5, 0.5])
    with pytest.raises(ValueError):
        EMASmoother(0)


def test_alert_hysteresis_and_cooldown():
    sm = AlertStateMachine("threat", on=0.7, off=0.4, min_on=2, min_off=2, cooldown_s=10)
    assert sm.update(0.9, "phishing", 0) is None          # 1st frame above: not yet
    ev = sm.update(0.9, "phishing", 1)
    assert ev and ev.status == "raised"
    assert sm.update(0.5, "phishing", 2) is None          # between thresholds: stays active
    assert sm.update(0.1, "phishing", 3) is None
    ev = sm.update(0.1, "phishing", 4)
    assert ev and ev.status == "cleared"
    sm.update(0.9, "phishing", 5)
    assert sm.update(0.9, "phishing", 6) is None          # same label: still inside cooldown
    assert sm.update(0.9, "phishing", 12).status == "raised"
    assert sm.update(0.9, "scam", 13) is None             # label switch while active needs min_on frames
    ev = sm.update(0.9, "scam", 14)
    assert ev and ev.status == "raised" and ev.label == "scam"


def test_different_threat_type_ignores_cooldown():
    sm = AlertStateMachine("threat", on=0.7, off=0.4, min_on=1, min_off=1, cooldown_s=100)
    assert sm.update(0.9, "phishing", 0).status == "raised"
    assert sm.update(0.1, "phishing", 1).status == "cleared"
    assert sm.update(0.9, "phishing", 2) is None          # cooldown for the same label
    sm2 = AlertStateMachine("threat", on=0.7, off=0.4, min_on=1, min_off=1, cooldown_s=100)
    sm2.update(0.9, "phishing", 0)
    sm2.update(0.1, "phishing", 1)
    assert sm2.update(0.9, "scam", 2).status == "raised"  # a new kind of threat alerts immediately
    with pytest.raises(ValueError):
        AlertStateMachine("x", on=0.4, off=0.5)


def test_redaction():
    t = "mail bob.k@example.com call +1-888-555-0101 card 4111 1111 1111 1111 password: hunter2 code 123456"
    r = redact(t)
    for secret in ("bob.k@example.com", "555-0101", "4111", "hunter2", "123456"):
        assert secret not in r
