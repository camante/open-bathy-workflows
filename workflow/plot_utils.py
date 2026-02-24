#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_utils.py – safe, lazy Matplotlib initialization.

Why this exists:
- Several modules imported Matplotlib at import-time, which can:
  * create noisy cache warnings in headless/readonly environments
  * slow down non-plotting runs
  * fail when Matplotlib isn't installed (even if plots are disabled)

Use:
    plt = lazy_pyplot()   # returns matplotlib.pyplot
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

def _ensure_mplconfigdir() -> None:
    # Avoid ~/.config/matplotlib permission issues by forcing a writable cache dir.
    # If user already set MPLCONFIGDIR, respect it.
    if os.environ.get("MPLCONFIGDIR"):
        return
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "mplconfig_open_bathy"
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(tmp)
    except Exception:
        # Best effort only
        return

def lazy_pyplot(backend: str = "Agg"):
    """Import matplotlib + pyplot lazily and safely."""
    _ensure_mplconfigdir()
    import matplotlib
    if backend:
        try:
            matplotlib.use(backend)
        except Exception:
            pass
    import matplotlib.pyplot as plt
    return plt
