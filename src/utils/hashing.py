"""Stable content IDs (FC-94)."""

from __future__ import annotations

import hashlib


def hash_id(content: str, length: int = 16) -> str:
    """SHA256 hex digest truncated to ``length`` chars (default 16 = 64 bits).

    64 bits ≈ 1.8e19 IDs before a 50% collision chance (birthday bound), vs
    MD5[:12] = 48 bits ≈ 16M. Non-cryptographic use (content fingerprint /
    dedup key); SHA256 just avoids MD5's weaknesses for the same effort.
    """
    return hashlib.sha256(content.encode()).hexdigest()[:length]
