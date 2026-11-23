"""Byte-level tokenizer.

Why bytes instead of a sub-word vocabulary?

* OCR output is noisy (``paypa1``, ``rn`` vs ``m``); a fixed WordPiece
  vocabulary fragments such strings unpredictably, bytes degrade gracefully.
* Phishing hinges on *character-level* tricks in URLs (homoglyphs, punycode,
  extra hyphens) that sub-word tokenizers tend to normalise away.
* No vocabulary file needs to be downloaded or shipped to the device, and the
  tokenizer is trivially portable to mobile runtimes.

Long screens are truncated head+tail so that both the browser chrome (URL,
top of page) and the bottom of the screen (download bars, chat input,
footers) survive truncation.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

PAD_ID = 0
CLS_ID = 1
SEP_ID = 2  # used as the newline token
UNK_ID = 3
BYTE_OFFSET = 4
VOCAB_SIZE = 256 + BYTE_OFFSET


class ByteTokenizer:
    def __init__(self, max_len: int = 512, tail_frac: float = 0.25, lowercase: bool = False):
        if max_len < 8:
            raise ValueError("max_len must be >= 8")
        self.max_len = max_len
        self.tail_frac = tail_frac
        self.lowercase = lowercase
        self.vocab_size = VOCAB_SIZE

    def _bytes(self, text: str) -> List[int]:
        if self.lowercase:
            text = text.lower()
        out: List[int] = []
        for line_no, line in enumerate(text.split("\n")):
            if line_no:
                out.append(SEP_ID)
            out.extend(b + BYTE_OFFSET for b in line.encode("utf-8", errors="replace"))
        return out

    def encode(self, text: str) -> np.ndarray:
        """Return a fixed-length int64 array ``[CLS, bytes..., PAD...]``."""
        ids = self._bytes(text or "")
        budget = self.max_len - 1
        if len(ids) > budget:
            tail = int(budget * self.tail_frac)
            head = budget - tail
            ids = ids[:head] + (ids[-tail:] if tail else [])
        arr = np.full(self.max_len, PAD_ID, dtype=np.int64)
        arr[0] = CLS_ID
        arr[1:1 + len(ids)] = ids
        return arr

    def encode_batch(self, texts: Iterable[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts])

    def decode(self, ids: Iterable[int]) -> str:
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i == SEP_ID:
                buf.extend(b"\n")
            elif i >= BYTE_OFFSET:
                buf.append(i - BYTE_OFFSET)
        return buf.decode("utf-8", errors="replace")

    @staticmethod
    def has_content(ids: np.ndarray) -> np.ndarray:
        """True for rows that contain at least one real byte (not just CLS/PAD)."""
        return (ids >= BYTE_OFFSET).any(axis=-1)
