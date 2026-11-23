"""Privacy helpers.

Screen monitoring sees everything the user sees, so the system follows
privacy-by-design defaults:

* all processing is on-device; nothing is uploaded;
* frames are never written to disk unless explicitly requested;
* event logs store labels, scores, rule IDs and domains — not screen text;
* if text logging is enabled, personal identifiers are redacted first.
"""
from __future__ import annotations

import re

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"\+?\d[\d\-\s().]{7,}\d"), "[PHONE]"),
    (re.compile(r"\b\d{6,}\b"), "[NUMBER]"),
    (re.compile(r"\b(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b"), "[WALLET]"),
    (re.compile(r"(?i)(password|passcode|pin)\s*[:=]\s*\S+"), r"\1: [REDACTED]"),
]


def redact(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text
