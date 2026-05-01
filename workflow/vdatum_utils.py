"""Vertical-datum conversion utilities.

The full VDatum conversion path is not part of the active river workflow in
this package.  Calls fail explicitly rather than silently returning an
unconverted raster.
"""

from __future__ import annotations


def convert_sdb_msl_to_navd88(*args, **kwargs):
    raise RuntimeError("convert_sdb_msl_to_navd88_unavailable_in_uploaded_phase0_package")


__all__ = ["convert_sdb_msl_to_navd88"]
