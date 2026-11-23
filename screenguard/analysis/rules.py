"""Signature rules over OCR text (Sigma-rule style indicators).

Each rule maps a regular expression to a threat class with a confidence
weight in (0, 1).  Rules are deliberately *not* the final decision: they are
combined with the neural model in :mod:`screenguard.analysis.decision`, which
lets the model veto rule false positives (e.g. the official ``rustup``
installer that pipes ``curl`` into ``sh``) and lets rules rescue novel threats
the model was never trained on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from .url_features import UrlReport, analyze_url, extract_urls


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    weight: float
    pattern: re.Pattern
    description: str


def _r(id_: str, category: str, weight: float, pattern: str, description: str) -> Rule:
    return Rule(id_, category, weight, re.compile(pattern, re.IGNORECASE | re.MULTILINE), description)


RULES: List[Rule] = [
    # ---- suspicious commands (MITRE ATT&CK T1059 and friends) -------------
    _r("CMD_PIPE_SHELL_REMOTE", "suspicious_command", 0.80, r"(curl|wget)\b[^\n|]{0,120}https?://(\d{1,3}\.){3}\d{1,3}[^\n|]*\|\s*(sudo\s+)?(ba|z)?sh\b",
       "downloads a script from a raw IP and pipes it into a shell"),
    _r("CMD_PIPE_SHELL", "suspicious_command", 0.40, r"(curl|wget|iwr|irm)\b[^\n|]{0,160}\|\s*(sudo\s+)?((ba|z)?sh|iex|cmd)\b",
       "pipes downloaded content directly into an interpreter"),
    _r("PS_ENCODED", "suspicious_command", 0.90, r"powershell(\.exe)?\b[^\n]{0,80}\s-(e|en|enc|encodedcommand)\s+[A-Za-z0-9+/=]{16,}",
       "PowerShell with a base64-encoded (hidden) command"),
    _r("PS_HIDDEN", "suspicious_command", 0.50, r"powershell(\.exe)?\b[^\n]{0,60}-w(indowstyle)?\s+hidden", "hidden PowerShell window"),
    _r("PS_IEX_DOWNLOAD", "suspicious_command", 0.85, r"(iex|invoke-expression)\b[^\n]{0,60}(downloadstring|net\.webclient|iwr|invoke-webrequest)|downloadstring\(",
       "downloads and executes code in memory"),
    _r("CERTUTIL_DOWNLOAD", "suspicious_command", 0.85, r"certutil(\.exe)?\b[^\n]{0,40}-urlcache", "abuses certutil to download a file (LOLBin)"),
    _r("BITSADMIN_DOWNLOAD", "suspicious_command", 0.60, r"bitsadmin\b[^\n]{0,40}/transfer", "abuses bitsadmin to download a file"),
    _r("MSHTA_REMOTE", "suspicious_command", 0.85, r"mshta(\.exe)?\s+https?://", "runs a remote HTA application"),
    _r("REGSVR32_REMOTE", "suspicious_command", 0.85, r"regsvr32\b[^\n]{0,60}/i:https?://", "regsvr32 remote scriptlet (Squiblydoo)"),
    _r("REVERSE_SHELL_DEVTCP", "suspicious_command", 0.90, r"/dev/tcp/\d{1,3}(\.\d{1,3}){3}/\d+", "bash reverse shell"),
    _r("REVERSE_SHELL_NC", "suspicious_command", 0.85, r"\b(nc|ncat|netcat)\b[^\n]{0,30}\s-e\s+/bin/(ba)?sh", "netcat reverse shell"),
    _r("REVERSE_SHELL_PY", "suspicious_command", 0.70, r"socket\.socket\(\)[^\n]{0,200}connect|os\.dup2\(s\.fileno\(\)", "socket-based reverse shell"),
    _r("RM_ROOT", "suspicious_command", 0.85, r"rm\s+-(rf|fr)\s+(--no-preserve-root\s+)?/(\s|$)", "recursive deletion of the root filesystem"),
    _r("CHMOD_ROOT", "suspicious_command", 0.60, r"chmod\s+-R\s+777\s+/(\s|$)", "makes the whole filesystem world-writable"),
    _r("SHADOW_DELETE", "suspicious_command", 0.90, r"vssadmin(\.exe)?\s+delete\s+shadows", "deletes backups (ransomware behaviour)"),
    _r("DEFENDER_OFF", "suspicious_command", 0.85, r"set-mppreference\b[^\n]{0,40}-disable(realtimemonitoring|ioavprotection)", "disables antivirus"),
    _r("B64_PIPE_SHELL", "suspicious_command", 0.80, r"base64\s+(-d|--decode)\s*\|\s*(ba|z)?sh", "decodes and runs a hidden script"),
    _r("B64_EXEC_PY", "suspicious_command", 0.60, r"b64decode\([^\n]{0,80}\)[^\n]{0,40}|os\.system\(p\.decode", "decodes and executes a hidden payload"),
    _r("PERSIST_RUNKEY", "suspicious_command", 0.50, r"reg(\.exe)?\s+add\s+[^\n]{0,80}\\currentversion\\run", "adds a Run-key persistence entry"),
    _r("PERSIST_SCHTASK", "suspicious_command", 0.40, r"schtasks(\.exe)?\s+/create", "creates a scheduled task"),
    _r("PERSIST_CRON", "suspicious_command", 0.55, r"crontab[^\n]{0,80}(curl|wget)", "cron job that downloads code"),
    _r("CLICKFIX", "suspicious_command", 0.75, r"(win(dows)?\s*\+\s*r)[\s\S]{0,200}(ctrl\s*\+\s*v|paste)", "page instructs you to paste a command into Run (ClickFix)"),
    _r("MACRO_AUTOOPEN_SHELL", "suspicious_command", 0.70, r"sub\s+auto_?open\(\)[\s\S]{0,120}shell", "Office macro that runs a shell on open"),
    # ---- malware downloads -------------------------------------------------
    _r("DOUBLE_EXTENSION", "malware_download", 0.85, r"\b[\w\-]+\.(pdf|docx?|xlsx?|jpe?g|png|txt|mp4)\.(exe|scr|bat|cmd|js|vbs|msi|ps1|jar|com|pif|hta)\b",
       "file name hides an executable behind a document extension"),
    _r("CRACK_KEYGEN", "malware_download", 0.70, r"\b[\w\-]*(crack|keygen|activator|patch|generator|hack|cheat)[\w\-]*\.(exe|msi|scr|apk|zip|rar)\b",
       "pirated software / generator executables commonly carry malware"),
    _r("FAKE_UPDATE", "malware_download", 0.55, r"(flash ?player|video codec|codec|media ?player|browser)[^\n]{0,60}(out of date|outdated|update required|missing|no longer secure)",
       "fake software-update lure"),
    _r("ENABLE_MACROS", "malware_download", 0.50, r"enable (macros|content|editing)", "asks you to enable macros/content in a document"),
    _r("RISKY_ATTACHMENT", "malware_download", 0.45, r"\b[\w\-]+\.(iso|img|lnk|hta|scr|xlsm)\b", "risky attachment/file type"),
    _r("SIDELOAD_APK", "malware_download", 0.45, r"\b(gold|pink|premium|mod|plus|free)[\w\-]*\.apk\b", "modded app (APK) from an unofficial source"),
    _r("UNKNOWN_PUBLISHER", "malware_download", 0.35, r"unknown publisher|this file may be dangerous", "unsigned / flagged executable"),
    # ---- scams --------------------------------------------------------------
    _r("TECH_SUPPORT", "scam", 0.80, r"(virus(es)?|infected|trojan|spyware|locked|blocked)[\s\S]{0,160}call[\s\S]{0,60}(\+?\d[\d\-\s().]{7,})",
       "fake virus warning with a support phone number"),
    _r("DONT_CLOSE", "scam", 0.40, r"do not (close|restart|shut down) (this window|your computer)", "pressure tactic to keep the page open"),
    _r("CRYPTO_DOUBLING", "scam", 0.85, r"(send|transfer)\s+[\d.]+\s*(btc|eth|usdt|sol|bitcoin)[\s\S]{0,60}(get|receive|back)", "crypto 'doubling' giveaway"),
    _r("GIVEAWAY", "scam", 0.55, r"(congratulations[\s\S]{0,60}(won|selected|winner))|claim (your )?(prize|reward)|lucky (visitor|draw)",
       "prize / giveaway lure"),
    _r("GIFT_CARD", "scam", 0.70, r"gift ?cards?[\s\S]{0,60}(codes?|pay|fee|send)|(pay|buy|purchase)[\s\S]{0,40}gift ?cards?", "payment demanded in gift cards"),
    _r("GUARANTEED_RETURNS", "scam", 0.70, r"guaranteed\s+(returns|profit)|\d{2,4}%\s+(daily\s+)?(profit|returns)", "unrealistic guaranteed investment returns"),
    _r("ADVANCE_FEE", "scam", 0.65, r"(inheritance|beneficiary|lottery|unclaimed)[\s\S]{0,120}(fee|bank account|claim)", "advance-fee fraud"),
    _r("FAMILY_EMERGENCY", "scam", 0.50, r"(lost my phone|new number)[\s\S]{0,80}(send|transfer)[\s\S]{0,20}money", "'hi mum' impersonation scam"),
    # ---- phishing -----------------------------------------------------------
    _r("VERIFY_ACCOUNT", "phishing", 0.35, r"(verify|confirm|validate|update)\s+(your\s+)?(account|identity|password|billing|payment|details)",
       "asks you to verify account details"),
    _r("ACCOUNT_THREAT", "phishing", 0.40, r"(account|access)\s+(has been |will be |was )?(temporarily )?(suspended|locked|disabled|limited|restricted|closed)",
       "threatens account suspension"),
    _r("UNUSUAL_ACTIVITY", "phishing", 0.25, r"(unusual|suspicious|unauthori[sz]ed)\s+(sign[- ]?in|login|activity)", "claims suspicious activity"),
    _r("DEADLINE", "phishing", 0.30, r"within\s+(12|24|48|72)\s+hours|expires? today|final notice", "artificial deadline"),
    _r("PARCEL_FEE", "phishing", 0.70, r"(parcel|package|shipment|delivery)[\s\S]{0,60}(unpaid|customs|fee|on hold)", "parcel-delivery smishing"),
    _r("COPYRIGHT_APPEAL", "phishing", 0.65, r"copyright (violation|infringement)[\s\S]{0,80}(appeal|form)", "fake copyright-violation appeal"),
    _r("SECRETS_FORM", "phishing", 0.60, r"recovery phrase|seed phrase|social security number|card number", "form asks for highly sensitive secrets"),
]


@dataclass
class RuleHit:
    rule_id: str
    category: str
    weight: float
    description: str
    match: str


@dataclass
class TextEvidence:
    hits: List[RuleHit] = field(default_factory=list)
    urls: List[UrlReport] = field(default_factory=list)
    class_scores: Dict[str, float] = field(default_factory=dict)

    def reasons(self, top: int = 5) -> List[str]:
        items = [(h.weight, h.description) for h in self.hits]
        for u in self.urls:
            for r in u.reasons:
                items.append((u.score, f"{u.host}: {r}"))
        items.sort(key=lambda t: -t[0])
        seen, out = set(), []
        for _, r in items:
            if r not in seen:
                seen.add(r)
                out.append(r)
            if len(out) >= top:
                break
        return out


def scan_text(text: str) -> List[RuleHit]:
    hits: List[RuleHit] = []
    for rule in RULES:
        m = rule.pattern.search(text or "")
        if m:
            hits.append(RuleHit(rule.id, rule.category, rule.weight, rule.description, m.group(0)[:120]))
    return hits


def analyze_text(text: str, url_weight: float = 0.85) -> TextEvidence:
    """Run rules and URL analysis; aggregate into per-class evidence in [0, 1).

    Evidence for a class is combined with a noisy-OR over its hits:
    ``r_c = 1 - prod(1 - w_i)``.
    """
    hits = scan_text(text)
    urls = [analyze_url(u) for u in extract_urls(text)]
    remaining: Dict[str, float] = {}

    def add(cat: str, w: float):
        remaining[cat] = remaining.get(cat, 1.0) * (1.0 - max(0.0, min(0.95, w)))

    for h in hits:
        add(h.category, h.weight)
    for u in urls:
        if u.score > 0.5:
            add("malware_download" if "risky_download" in u.features else "phishing", url_weight * u.score)
    return TextEvidence(hits=hits, urls=urls, class_scores={c: 1.0 - r for c, r in remaining.items()})
