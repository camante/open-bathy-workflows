"""Hashing / fingerprinting helpers.

These are used for cache keys and deterministic filenames.
"""

from __future__ import annotations

import hashlib
from typing import Any


def hash_key(*parts: Any, n: int = 12) -> str:
    """Return a short deterministic hash for arbitrary parts.

    Notes:
      - Uses MD5 for filename/key compactness; not for security.
      - Be defensive: callers may pass floats/ints/Paths.
    """
    h = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:n]


def sha1_text(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]
