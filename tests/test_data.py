import random

import numpy as np
import pytest
from PIL import Image

from screenguard.data.ocr_noise import corrupt
from screenguard.data.preprocess import letterbox, preprocess
from screenguard.data.synthetic import MOBILE_ACTIVITIES, ScreenGenerator
from screenguard.taxonomy import ACTIVITY_CLASSES, THREAT_COMPATIBILITY, THREAT_CLASSES
from screenguard.text.tokenizer import CLS_ID, PAD_ID, SEP_ID, ByteTokenizer


def test_generator_is_deterministic():
    g = ScreenGenerator(320, 180, seed=3)
    a, b = g.generate(11), g.generate(11)
    assert a.text == b.text and a.activity == b.activity and a.threat == b.threat
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))
    assert g.generate(12).text != a.text


@pytest.mark.parametrize("activity", ACTIVITY_CLASSES)
def test_every_activity_threat_combination_renders(activity):
    g = ScreenGenerator(320, 180, seed=0)
    for threat in THREAT_COMPATIBILITY[activity]:
        devices = ["desktop", "mobile"] if activity in MOBILE_ACTIVITIES else ["desktop"]
        for dev in devices:
            s = g.sample(random.Random(1), activity=activity, threat=threat, device=dev)
            assert s.activity == activity and s.threat == threat and s.threat in THREAT_CLASSES
            assert s.image.size == ((320, 180) if dev == "desktop" else (180, 320))
            if activity not in ("video_streaming",):
                assert len(s.text) > 10, (activity, threat, dev)


def test_threat_text_contains_cues():
    g = ScreenGenerator(640, 360, seed=0)
    s = g.sample(random.Random(5), activity="terminal", threat="suspicious_command")
    assert "$" in s.text or "PS C:" in s.text
    s = g.sample(random.Random(5), activity="web_browsing", threat="phishing")
    assert s.url and s.url.split("//")[-1].split("/")[0] in s.text.replace("https://", "").replace("http://", "") + s.url


def test_content_image_is_composited():
    g = ScreenGenerator(640, 360, seed=0)
    red = Image.new("RGB", (200, 200), (255, 0, 0))
    s = g.sample(random.Random(2), activity="video_streaming", threat="benign", device="desktop", content_image=red)
    arr = np.asarray(s.image)
    red_px = ((arr[..., 0] > 240) & (arr[..., 1] < 20) & (arr[..., 2] < 20)).mean()
    assert red_px > 0.05


def test_letterbox_preserves_aspect():
    img = Image.new("RGB", (400, 100), (255, 255, 255))
    out = letterbox(img, (144, 256))
    arr = np.asarray(out)
    assert out.size == (256, 144)
    assert arr[0, 128].sum() == 0 and arr[72, 128].sum() == 765  # padded top, content in middle
    x = preprocess(img, (144, 256))
    assert x.shape == (3, 144, 256) and x.dtype == np.float32


def test_tokenizer_roundtrip_and_truncation():
    tok = ByteTokenizer(64, tail_frac=0.25)
    ids = tok.encode("hello\nworld")
    assert ids[0] == CLS_ID and SEP_ID in ids and ids[-1] == PAD_ID and len(ids) == 64
    assert tok.decode(ids) == "hello\nworld"
    long = "A" * 100 + "Z" * 100
    dec = tok.decode(tok.encode(long))
    assert dec.startswith("A") and dec.endswith("Z") and len(dec) == 63
    assert tok.has_content(tok.encode_batch(["", "x"])).tolist() == [False, True]
    assert tok.decode(tok.encode("paypa1.com ✓")) == "paypa1.com ✓"


def test_ocr_noise():
    text = "Verify your account at https://paypal.com/login now\nsecond line"
    assert corrupt(text, 0.0, random.Random(0)) == text
    a = corrupt(text, 0.2, random.Random(0))
    assert a != text and a == corrupt(text, 0.2, random.Random(0))
