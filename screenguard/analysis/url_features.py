"""URL extraction and lexical / reference-based phishing analysis.

These hand-crafted features are a fast, explainable *prior* that complements
the learned model: they generalise to brand-new lookalike domains the model
has never seen, and each one comes with a human-readable reason that is shown
in alerts.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from .brands import brand_labels, legit_domains

URL_RE = re.compile(
    r"""(?ix)
    (?:https?://|hxxps?://|www\.)[^\s<>"'`]+          # explicit scheme / www
    |
    \b(?:[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?\.)+      # bare domain labels
      (?:[a-z]{2,24}|xn--[a-z0-9-]{2,30})\b
      (?:/[^\s<>"'`]*)?
    """
)
FILE_EXTS = {"pdf", "exe", "py", "js", "txt", "doc", "docx", "jpg", "jpeg", "png", "gif", "zip", "sh", "md", "json",
             "csv", "html", "htm", "msi", "scr", "bat", "cmd", "apk", "iso", "lnk", "hta", "vbs", "ps1", "xlsx",
             "xlsm", "pptx", "mp3", "mp4", "pth", "yaml", "yml", "cfg", "ini", "log", "sql", "c", "h", "cpp", "ts",
             "tsx", "jsx", "css", "dll", "sys", "tar", "gz", "rar", "7z", "img", "jar", "b64", "sct"}
# Bare (scheme-less) domains are only accepted with a real TLD, so code such as
# ``torch.nn`` or ``os.system`` is not mistaken for a URL.
KNOWN_TLDS = set("""
com net org edu gov mil int io co ai app dev info biz me tv us uk de fr rw ke ng za in cn jp ru br au ca es it nl
se no pl ch be at cz dk fi gr hu ie il kr mx nz pt ro sg tr tw ua vn ar cl pe ug tz gh et eg ma sn cm ci bd pk
id my ph th hk sa ae qa ir iq lk np so gg ly sh to cc ws la st is lu lt lv ee sk si hr rs ba bg md by kz uz
xyz top tk ml ga cf gq zip mov click country kim work rest loan win bid stream download racing review party men
date gdn support live icu cyou buzz monster sbs quest lol online site store tech club shop vip fun space website
link page cloud email news blog today world life pro social video tube wiki one global agency digital network
""".split())
SUSPICIOUS_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "zip", "mov", "click", "country", "kim", "work",
                   "rest", "loan", "win", "bid", "stream", "download", "racing", "review", "party", "men", "date",
                   "gdn", "support", "live", "icu", "cyou", "buzz", "monster", "sbs", "quest", "lol", "info"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "cutt.ly", "rb.gy", "shorturl.at",
              "tiny.cc", "rebrand.ly", "s.id"}
MULTI_SUFFIXES = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "co.jp", "co.in", "com.br", "co.za",
                  "co.rw", "com.ng", "co.ke", "com.cn", "com.mx", "co.nz", "com.tr", "com.sg"}
SENSITIVE_WORDS = {"login", "signin", "sign-in", "verify", "verification", "secure", "account", "update", "banking",
                   "confirm", "password", "wallet", "unlock", "recover", "billing", "suspend", "webscr", "auth"}
RISKY_EXTS = (".exe", ".scr", ".bat", ".cmd", ".msi", ".vbs", ".js", ".jar", ".apk", ".ps1", ".hta", ".dll", ".iso",
              ".img", ".lnk", ".sct", ".xlsm")
HOMOGLYPHS = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s", "|": "l", "!": "i"}
MULTI_GLYPHS = [("rn", "m"), ("vv", "w"), ("cl", "d")]

_BRAND_LABELS = brand_labels()
_LEGIT = legit_domains()


@dataclass
class UrlReport:
    url: str
    host: str
    registered_domain: str
    score: float
    reasons: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    brand: Optional[str] = None


def extract_urls(text: str, limit: int = 20) -> List[str]:
    out: List[str] = []
    seen = set()
    for m in URL_RE.finditer(text or ""):
        u = m.group(0).rstrip(".,;:)]}!?'\"")
        low = u.lower()
        explicit = low.startswith(("http://", "https://", "hxxp", "www."))
        if not explicit:
            host_orig = u.split("/")[0]
            host = host_orig.lower()
            tld = host.rsplit(".", 1)[-1]
            if "." not in host or tld in FILE_EXTS or tld not in KNOWN_TLDS:
                continue
            if any(ch.isupper() for ch in host_orig.rsplit(".", 1)[-1]):  # e.g. Net.WebClient
                continue
            if re.fullmatch(r"\d+(\.\d+)+", host):  # version numbers like 3.13.1
                continue
        if low not in seen:
            seen.add(low)
            out.append(u)
        if len(out) >= limit:
            break
    return out


def registered_domain(host: str) -> str:
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def normalize_glyphs(s: str) -> str:
    s = "".join(HOMOGLYPHS.get(c, c) for c in s.lower())
    for a, b in MULTI_GLYPHS:
        s = s.replace(a, b)
    return s


def damerau_levenshtein(a: str, b: str, max_d: int = 3) -> int:
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _brand_match(host: str, reg: str) -> Tuple[Optional[str], str]:
    """Return (brand, kind) where kind is 'legit', 'embedded', 'lookalike' or ''."""
    if reg in _LEGIT:
        return _LEGIT[reg], "legit"
    sld = reg.split(".")[0]
    tokens = [t for t in re.split(r"[-.]", sld) if t]
    sub_labels = host[: -len(reg)].strip(".").split(".") if host != reg else []
    for label, brand in _BRAND_LABELS.items():
        if len(label) < 3:
            continue
        if label in sub_labels or any(label == t for t in tokens) or (len(label) >= 5 and label in sld):
            return brand, "embedded"
    candidates = tokens + [sld.replace("-", "")]
    for tok in candidates:
        nt = normalize_glyphs(tok)
        for label, brand in _BRAND_LABELS.items():
            if len(label) < 4:
                continue
            if nt == label and tok != label:
                return brand, "lookalike"
            d = damerau_levenshtein(nt, label, 2)
            if (len(label) >= 5 and d <= 1) or (len(label) >= 8 and d <= 2):
                return brand, "lookalike"
    return None, ""


def analyze_url(url: str) -> UrlReport:
    raw = url.strip()
    low = raw.lower().replace("hxxp", "http")
    if not low.startswith(("http://", "https://")):
        low = "http://" + low if not low.startswith("www.") else "https://" + low
        scheme_known = False
    else:
        scheme_known = True
    try:
        parts = urlsplit(low)
        host = (parts.hostname or "").strip(".")
        path = (parts.path or "") + ("?" + parts.query if parts.query else "")
    except ValueError:
        host, path, parts = "", "", None
    f: Dict[str, float] = {}
    reasons: List[str] = []
    is_ip = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))
    reg = host if is_ip else registered_domain(host)
    tld = "" if is_ip else host.rsplit(".", 1)[-1]
    sld = reg.split(".")[0] if reg else ""

    def add(name: str, weight: float, reason: Optional[str] = None):
        f[name] = weight
        if reason:
            reasons.append(reason)

    brand, kind = (None, "") if is_ip else _brand_match(host, reg)
    if kind == "legit":
        add("known_brand_domain", -3.0)
    elif kind == "embedded":
        add("brand_in_untrusted_domain", 3.0, f"mentions '{brand}' but is not an official {brand} domain")
    elif kind == "lookalike":
        add("brand_lookalike", 3.2, f"domain '{reg}' imitates {brand}")
    if is_ip:
        add("ip_host", 2.8, "uses a raw IP address instead of a domain name")
    if "xn--" in host:
        add("punycode", 2.0, "internationalised (punycode) domain, possible homograph attack")
    if tld in SUSPICIOUS_TLDS:
        add("suspicious_tld", 1.2, f"top-level domain '.{tld}' is frequently abused")
    if reg in SHORTENERS:
        add("shortener", 1.2, "link shortener hides the real destination")
    if "@" in (parts.netloc if parts else ""):
        add("at_symbol", 2.0, "'@' in URL can disguise the real host")
    if scheme_known and low.startswith("http://") and not is_ip:
        add("no_https", 0.6, "connection is not encrypted (http)")
    words = sum(1 for w in SENSITIVE_WORDS if w in path or w in host[: -len(reg)] or w in sld)
    if words and kind != "legit":
        add("sensitive_words", min(1.4, 0.5 * words), "login/verification keywords on an unknown site")
    n_sub = host[: -len(reg)].count(".") if host != reg else 0
    if n_sub >= 3:
        add("many_subdomains", 0.8, f"{n_sub} levels of subdomains")
    if sld.count("-") >= 2:
        add("hyphenated", 0.6, "many hyphens in domain name")
    if re.search(r"\d", sld) and not is_ip:
        add("digits_in_domain", 0.4)
    if len(raw) > 90:
        add("long_url", 0.4)
    if path.lower().split("?")[0].endswith(RISKY_EXTS):
        add("risky_download", 1.6, f"link downloads an executable file ({path.rsplit('.', 1)[-1]})")
    z = -3.0 + sum(f.values())
    score = 1.0 / (1.0 + math.exp(-z))
    return UrlReport(url=raw, host=host, registered_domain=reg, score=score, reasons=reasons, features=f, brand=brand)
