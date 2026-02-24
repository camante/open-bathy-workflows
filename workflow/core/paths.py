"""Filesystem/path helpers.

Keep these helpers tiny and side-effect free (except directory creation).
"""

from __future__ import annotations

from pathlib import Path


def ensure_dir(p: Path) -> Path:
    """Create directory (and parents) if needed; return the Path."""
    p.mkdir(parents=True, exist_ok=True)
    return p
