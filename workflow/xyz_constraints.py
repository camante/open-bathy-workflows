#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xyz_constraints.py

Utilities for enforcing authoritative XYZ as hard constraints.

Assumptions
- Z values are bed elevations in an orthometric datum (NAVD88 in your workflow).
- X/Y are in the same CRS as the template raster.

Supported inputs (conservative)
- .xyz/.txt/.dat: whitespace- or comma-delimited; first three numeric columns are x,y,z
- .csv: columns named (x,y,z) or (lon,lat,z) etc; falls back to first three numeric columns

We intentionally avoid heavy GIS dependencies here; vector formats (gpkg/shp) should be
converted to XYZ upstream in this workflow.
"""


import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import rasterio


@dataclass
class RasterizedConstraints:
    values: np.ndarray  # float32 template grid with nodata where unconstrained
    mask: np.ndarray    # bool mask of constrained pixels
    count: int          # number of input points used (after bounds filtering)


def _read_text_xyz(path: Path) -> np.ndarray:
    """Read an xyz-like text file to Nx3 float array."""
    # Try whitespace first, then commas
    arr = np.genfromtxt(path, dtype=float, comments='#', invalid_raise=False)
    if arr is None or (isinstance(arr, float) and np.isnan(arr)):
        return np.empty((0, 3), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((1, -1))
    if arr.shape[1] < 3:
        # maybe comma-delimited
        arr = np.genfromtxt(path, dtype=float, delimiter=',', comments='#', invalid_raise=False)
        if arr.ndim == 1:
            arr = arr.reshape((1, -1))
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    return arr[:, :3]


def _read_csv_xyz(path: Path) -> np.ndarray:
    import csv
    with path.open('r', newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return np.empty((0, 3), dtype=float)
    # Candidate names
    keys = {k.lower(): k for k in rows[0].keys()}
    def pick(*names):
        for n in names:
            if n in keys:
                return keys[n]
        return None
    kx = pick('x','easting','lon','longitude')
    ky = pick('y','northing','lat','latitude')
    kz = pick('z','bed','elev','elevation','depth')
    out = []
    for row in rows:
        try:
            if kx and ky and kz:
                x = float(row[kx]); y = float(row[ky]); z = float(row[kz])
                out.append((x,y,z)); continue
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        # fallback: first 3 numeric fields
        vals=[]
        for v in row.values():
            try:
                vals.append(float(v))
            except Exception:
                continue
        if len(vals) >= 3:
            out.append((vals[0], vals[1], vals[2]))
    if not out:
        return np.empty((0, 3), dtype=float)
    return np.asarray(out, dtype=float)


def read_xyz_points(paths: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        suf = p.suffix.lower()
        try:
            if suf in ('.csv',):
                arr = _read_csv_xyz(p)
            else:
                arr = _read_text_xyz(p)
        except Exception:
            continue
        if arr.size == 0:
            continue
        xs.extend(arr[:,0].tolist())
        ys.extend(arr[:,1].tolist())
        zs.extend(arr[:,2].tolist())
    if not xs:
        return (np.empty((0,), dtype=float), np.empty((0,), dtype=float), np.empty((0,), dtype=float))
    return (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(zs, dtype=float))


def rasterize_constraints_to_template(
    template_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    nodata: float = -9999.0,
    agg: str = 'median'
) -> RasterizedConstraints:
    """Rasterize point constraints onto the template grid.

    Aggregation is per-pixel over all points falling into that pixel.
    """
    if x.size == 0:
        with rasterio.open(template_path) as ds:
            arr = np.full((ds.height, ds.width), nodata, dtype=np.float32)
            m = np.zeros((ds.height, ds.width), dtype=bool)
        return RasterizedConstraints(arr, m, 0)

    with rasterio.open(template_path) as ds:
        h, w = ds.height, ds.width
        rows, cols = rasterio.transform.rowcol(ds.transform, x, y, op=int)
        rows = np.asarray(rows); cols = np.asarray(cols)
        inb = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w) & np.isfinite(z)
        if not np.any(inb):
            arr = np.full((h, w), nodata, dtype=np.float32)
            m = np.zeros((h, w), dtype=bool)
            return RasterizedConstraints(arr, m, 0)
        rows = rows[inb]; cols = cols[inb]; zv = z[inb].astype(float)
        idx = rows * w + cols
        order = np.argsort(idx)
        idx = idx[order]
        zv = zv[order]
        # group
        uniq, start = np.unique(idx, return_index=True)
        # compute aggregate
        outv = np.empty(uniq.shape[0], dtype=np.float32)
        for i, s in enumerate(start):
            e = start[i+1] if i+1 < start.size else zv.size
            chunk = zv[s:e]
            if agg == 'mean':
                outv[i] = float(np.mean(chunk))
            else:
                outv[i] = float(np.median(chunk))
        arr = np.full((h*w,), nodata, dtype=np.float32)
        arr[uniq] = outv
        arr = arr.reshape((h, w))
        m = arr != nodata
        return RasterizedConstraints(arr, m, int(zv.size))


def burn_constraints(raster: np.ndarray, constraints: RasterizedConstraints) -> np.ndarray:
    """Overwrite raster values wherever constraints.mask is True."""
    out = raster.copy()
    out[constraints.mask] = constraints.values[constraints.mask]
    return out
