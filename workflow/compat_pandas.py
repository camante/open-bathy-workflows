"""Compatibility shims for pandas/geopandas stacks.

This repo is frequently run in mixed environments (conda, system python, HPC) where
GeoPandas may lag behind pandas. GeoPandas versions prior to the pandas 2.0
transition reference pandas.Int64Index/UInt64Index, which were removed.

This module provides a safe, minimal shim so scripts can still write GeoPackages
via GeoPandas without forcing users to immediately reconcile package versions.

Importing this module has no effect on newer stacks where these symbols already
exist.
"""

from __future__ import annotations

import logging

log = logging.getLogger("compat_pandas")
try:
    import pandas as pd  # type: ignore

    if not hasattr(pd, "Int64Index"):
        pd.Int64Index = pd.Index  # type: ignore[attr-defined]
    if not hasattr(pd, "UInt64Index"):
        pd.UInt64Index = pd.Index  # type: ignore[attr-defined]
except Exception:
    # If pandas is not available, callers will fail elsewhere anyway.
    log.debug("Optional step failed; continuing.", exc_info=True)
