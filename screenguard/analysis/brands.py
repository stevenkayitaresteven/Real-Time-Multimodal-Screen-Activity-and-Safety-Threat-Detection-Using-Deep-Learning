"""Brands that are frequently impersonated in phishing, with their real domains.

This list is the "protected brand" reference used by the lookalike-domain
detector (the same idea used by reference-based phishing detectors such as
Phishpedia).  Extend it with brands relevant to your users, e.g. local banks
and mobile-money operators.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# brand name -> (legitimate registered domains, brand RGB colour)
BRANDS: Dict[str, Tuple[List[str], Tuple[int, int, int]]] = {
    "PayPal": (["paypal.com"], (0, 48, 135)),
    "Microsoft": (["microsoft.com", "live.com", "office.com", "microsoftonline.com", "outlook.com"], (0, 120, 212)),
    "Apple": (["apple.com", "icloud.com"], (30, 30, 30)),
    "Google": (["google.com", "gmail.com", "youtube.com"], (66, 133, 244)),
    "Amazon": (["amazon.com", "amazon.co.uk"], (255, 153, 0)),
    "Netflix": (["netflix.com"], (229, 9, 20)),
    "Facebook": (["facebook.com", "fb.com", "meta.com"], (24, 119, 242)),
    "Instagram": (["instagram.com"], (193, 53, 132)),
    "LinkedIn": (["linkedin.com"], (10, 102, 194)),
    "Dropbox": (["dropbox.com"], (0, 97, 255)),
    "DHL": (["dhl.com"], (212, 5, 17)),
    "Chase": (["chase.com"], (17, 122, 202)),
    "Coinbase": (["coinbase.com"], (0, 82, 255)),
    "WhatsApp": (["whatsapp.com"], (37, 211, 102)),
    "Steam": (["steampowered.com", "steamcommunity.com"], (23, 26, 33)),
    "DocuSign": (["docusign.com", "docusign.net"], (255, 204, 34)),
}


def brand_labels() -> Dict[str, str]:
    """Map the lowercase second-level label of each legit domain to its brand."""
    out: Dict[str, str] = {}
    for brand, (domains, _) in BRANDS.items():
        for d in domains:
            out[d.split(".")[0]] = brand
        out[brand.lower()] = brand
    return out


def legit_domains() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for brand, (domains, _) in BRANDS.items():
        for d in domains:
            out[d] = brand
    return out
