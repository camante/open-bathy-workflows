# -*- coding: utf-8 -*-
"""Lightweight dependency helpers.

Goal:
- Allow `--help` and lightweight introspection to run even if heavy geo/science deps
  (rasterio/geopandas/sklearn) are not installed.
- Provide clear, consistent error messages when a pipeline stage actually needs them.
"""

from __future__ import annotations

from typing import Any, Optional


class MissingDependency(RuntimeError):
    pass


def _missing(pkg: str, extra: str = "") -> MissingDependency:
    msg = f"Missing required dependency: {pkg}."
    if extra:
        msg += " " + extra.strip()
    msg += " Install it (e.g., via conda/pip) or run with stages that don't require it."
    return MissingDependency(msg)


def try_import(name: str) -> Optional[Any]:
    try:
        module = __import__(name)
        return module
    except Exception:
        return None


def require(module: Optional[Any], name: str, extra: str = "") -> Any:
    if module is None:
        raise _missing(name, extra)
    return module
