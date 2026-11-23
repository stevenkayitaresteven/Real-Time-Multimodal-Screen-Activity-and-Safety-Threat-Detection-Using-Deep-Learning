"""Label spaces for the three tasks the system solves.

The taxonomy is deliberately small and explicit so that it can be versioned
inside every checkpoint: a model trained with one label space is never
silently loaded with another.
"""
from __future__ import annotations

from typing import Dict, List, Optional

IGNORE_INDEX = -100  # label value for "this sample has no annotation for this task"

ACTIVITY_CLASSES: List[str] = [
    "web_browsing",
    "video_streaming",
    "messaging",
    "email",
    "document_editing",
    "code_editor",
    "terminal",
    "social_media",
    "file_download",
]

NSFW_CLASSES: List[str] = [
    "safe",
    "suggestive",
    "explicit",
]

THREAT_CLASSES: List[str] = [
    "benign",
    "phishing",
    "scam",
    "malware_download",
    "suspicious_command",
]

TASKS: Dict[str, List[str]] = {
    "activity": ACTIVITY_CLASSES,
    "nsfw": NSFW_CLASSES,
    "threat": THREAT_CLASSES,
}

TASK_ORDER: List[str] = list(TASKS.keys())

# Which threats can plausibly appear on which activity. Used by the synthetic
# generator and documented here so the label space and data are consistent.
THREAT_COMPATIBILITY: Dict[str, Dict[str, float]] = {
    # activity:          {threat: prior probability}
    "web_browsing":     {"benign": 0.50, "phishing": 0.14, "scam": 0.12, "malware_download": 0.12, "suspicious_command": 0.12},
    "video_streaming":  {"benign": 0.80, "scam": 0.20},
    "messaging":        {"benign": 0.50, "phishing": 0.18, "scam": 0.17, "malware_download": 0.15},
    "email":            {"benign": 0.50, "phishing": 0.20, "scam": 0.15, "malware_download": 0.15},
    "document_editing": {"benign": 1.00},
    "code_editor":      {"benign": 0.70, "suspicious_command": 0.30},
    "terminal":         {"benign": 0.50, "suspicious_command": 0.50},
    "social_media":     {"benign": 0.50, "scam": 0.30, "phishing": 0.20},
    "file_download":    {"benign": 0.50, "malware_download": 0.50},
}

# Activities whose layout has a large media slot (used when compositing
# external images, e.g. NSFW datasets, into realistic screen context).
MEDIA_ACTIVITIES: List[str] = ["web_browsing", "video_streaming", "social_media", "messaging"]


def label_to_index(task: str, label: Optional[str]) -> int:
    """Map a string label to its class index, or IGNORE_INDEX when missing."""
    if label is None or label == "":
        return IGNORE_INDEX
    classes = TASKS[task]
    try:
        return classes.index(label)
    except ValueError as exc:
        raise ValueError(f"Unknown label {label!r} for task {task!r}; expected one of {classes}") from exc


def index_to_label(task: str, index: int) -> Optional[str]:
    if index == IGNORE_INDEX:
        return None
    return TASKS[task][index]
