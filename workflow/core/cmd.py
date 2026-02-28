"""Command building helpers.

Lifted from bathy_main.py (Phase-2 refactor). No behavior changes intended.
"""

from __future__ import annotations

from typing import List


def build_cmd(*args) -> List[str]:
    out: List[str] = []
    for a in args:
        if a is None:
            continue
        out.append(str(a))
    return out
