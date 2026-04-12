"""Seam comparison metrics between adjacent bathy tiles.

Goals:
  - Deterministic: no filesystem discovery / globbing.
  - Explicit inputs: caller provides exact raster paths (or explicit manifests).
  - Seam-focused: evaluate a narrow strip along the shared boundary.

This module is dependency-light (numpy + rasterio).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import logging
import numpy as np

from nodata_utils import sanitize_array, valid_mask

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeamSpec:
    kind: str  # 'vertical' or 'horizontal'
    seam_coord: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def _nearly_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def infer_adjacency(
    bounds_a: Tuple[float, float, float, float],
    bounds_b: Tuple[float, float, float, float],
    tol: float,
) -> Optional[SeamSpec]:
    """Infer adjacency (shared edge) or overlap neighborhood."""
    axmin, aymin, axmax, aymax = bounds_a
    bxmin, bymin, bxmax, bymax = bounds_b

    # Area overlap
    oxmin = max(axmin, bxmin)
    oymin = max(aymin, bymin)
    oxmax = min(axmax, bxmax)
    oymax = min(aymax, bymax)
    ow = oxmax - oxmin
    oh = oymax - oymin
    if ow > tol and oh > tol:
        # Use smaller overlap dimension to define seam orientation
        if ow <= oh:
            seam_x = (oxmin + oxmax) / 2.0
            return SeamSpec("vertical", seam_x, oxmin, oymin, oxmax, oymax)
        seam_y = (oymin + oymax) / 2.0
        return SeamSpec("horizontal", seam_y, oxmin, oymin, oxmax, oymax)

    # Touching edges
    yint_min = max(aymin, bymin)
    yint_max = min(aymax, bymax)
    xint_min = max(axmin, bxmin)
    xint_max = min(axmax, bxmax)

    if _nearly_equal(axmax, bxmin, tol) and (yint_max - yint_min) > tol:
        return SeamSpec("vertical", axmax, axmax, yint_min, bxmin, yint_max)
    if _nearly_equal(axmin, bxmax, tol) and (yint_max - yint_min) > tol:
        return SeamSpec("vertical", axmin, bxmax, yint_min, axmin, yint_max)
    if _nearly_equal(aymax, bymin, tol) and (xint_max - xint_min) > tol:
        return SeamSpec("horizontal", aymax, xint_min, aymax, xint_max, bymin)
    if _nearly_equal(aymin, bymax, tol) and (xint_max - xint_min) > tol:
        return SeamSpec("horizontal", aymin, xint_min, bymax, xint_max, aymin)

    return None


def _read_strip(ds, seam: SeamSpec, strip_px: int, side: str) -> Tuple[np.ndarray, Optional[float]]:
    import rasterio
    from rasterio.windows import from_bounds

    nodata = ds.nodata
    if seam.kind == "vertical":
        px_w = abs(ds.transform.a)
        if side == "a":
            xmin, xmax = seam.seam_coord - strip_px * px_w, seam.seam_coord
        else:
            xmin, xmax = seam.seam_coord, seam.seam_coord + strip_px * px_w
        ymin, ymax = seam.ymin, seam.ymax
    else:
        px_h = abs(ds.transform.e)
        if side == "a":
            ymin, ymax = seam.seam_coord, seam.seam_coord + strip_px * px_h
        else:
            ymin, ymax = seam.seam_coord - strip_px * px_h, seam.seam_coord
        xmin, xmax = seam.xmin, seam.xmax

    win = from_bounds(xmin, ymin, xmax, ymax, transform=ds.transform)
    win = win.round_offsets().round_lengths()
    arr = sanitize_array(ds.read(1, window=win, masked=False), nodata, dtype=np.float32)
    return arr, nodata


def compute_seam_metrics(
    raster_a: str | Path,
    raster_b: str | Path,
    strip_px: int = 3,
    nodata: Optional[float] = None,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    import rasterio

    ra = Path(raster_a)
    rb = Path(raster_b)
    if not ra.exists():
        raise FileNotFoundError(str(ra))
    if not rb.exists():
        raise FileNotFoundError(str(rb))

    with rasterio.open(ra) as da, rasterio.open(rb) as db:
        if da.crs != db.crs:
            raise ValueError(f"CRS mismatch: {da.crs} vs {db.crs}")
        px = min(abs(da.transform.a), abs(db.transform.a), abs(da.transform.e), abs(db.transform.e))
        seam = infer_adjacency(da.bounds, db.bounds, tol=max(tol, px * 0.25))
        if seam is None:
            return {"status": "not_adjacent", "raster_a": str(ra), "raster_b": str(rb)}

        a_strip, a_nd = _read_strip(da, seam, strip_px, side="a")
        b_strip, b_nd = _read_strip(db, seam, strip_px, side="b")

        nd = nodata
        if nd is None:
            nd = a_nd if a_nd is not None else b_nd

        a = a_strip.astype("float64", copy=False)
        b = b_strip.astype("float64", copy=False)

        ny = min(a.shape[0], b.shape[0])
        nx = min(a.shape[1], b.shape[1])
        a = a[:ny, :nx]
        b = b[:ny, :nx]

        valid = np.isfinite(a) & np.isfinite(b)
        if nd is not None:
            valid &= (a != nd) & (b != nd)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return {"status": "no_valid", "raster_a": str(ra), "raster_b": str(rb), "seam_kind": seam.kind}

        dv = (a - b)[valid]
        absdv = np.abs(dv)
        out: Dict[str, Any] = {
            "status": "ok",
            "raster_a": str(ra),
            "raster_b": str(rb),
            "seam_kind": seam.kind,
            "strip_px": int(strip_px),
            "n_valid": n_valid,
            "mean_abs": float(absdv.mean()),
            "rmse": float(np.sqrt((dv * dv).mean())),
            "max_abs": float(absdv.max()),
            "p95_abs": float(np.quantile(absdv, 0.95)),
        }

        # Normal gradient mismatch (requires 2+ px strip)
        if strip_px >= 2:
            try:
                if seam.kind == "vertical" and a.shape[1] >= 2 and b.shape[1] >= 2:
                    ga = a[:, -1] - a[:, -2]
                    gb = b[:, 1] - b[:, 0]
                    gv = (ga - gb)
                    gv = gv[np.isfinite(gv)]
                    if gv.size:
                        out["mean_abs_normal_gradient_mismatch"] = float(np.mean(np.abs(gv)))
                elif seam.kind == "horizontal" and a.shape[0] >= 2 and b.shape[0] >= 2:
                    ga = a[0, :] - a[1, :]
                    gb = b[-2, :] - b[-1, :]
                    gv = (ga - gb)
                    gv = gv[np.isfinite(gv)]
                    if gv.size:
                        out["mean_abs_normal_gradient_mismatch"] = float(np.mean(np.abs(gv)))
            except Exception:
                log.debug("ignored", exc_info=True)  # gradient mismatch metric optional
        return out


def load_primary_raster_from_io_manifest(
    io_manifest_path: str | Path,
    prefer_keys: Tuple[str, ...] = ("combined", "final", "bathy"),
) -> Path:
    """Pick a primary raster from an explicit io_manifest.json outputs list.

    NOTE: This does not glob the filesystem; it only ranks explicit manifest outputs.
    The selected path is returned so callers can audit/override if desired.
    """
    import json

    p = Path(io_manifest_path)
    io = json.loads(p.read_text(errors="ignore"))
    outs = [Path(x) for x in (io.get("outputs") or []) if isinstance(x, str)]
    outs = [x for x in outs if x.suffix.lower() in (".tif", ".tiff")]
    if not outs:
        raise ValueError(f"No raster outputs in manifest: {p}")

    def score(path: Path) -> Tuple[int, int]:
        s = path.name.lower()
        pri = 0
        for k in prefer_keys:
            if k in s:
                pri += 10
        # prefer combined deliverables over method-specific internals
        if "/river/" in str(path).lower() or "river" in s:
            pri -= 1
        if "/sdb/" in str(path).lower() or "sdb" in s:
            pri -= 1
        return (-pri, len(s))

    return sorted(outs, key=score)[0]


def compute_mask_boundary_seam_metrics(
    *,
    river_raster: str,
    fused_raster: str,
    mask_raster: str,
    mask_threshold: float = 0.5,
    boundary_mode: str = "inner",
) -> dict:
    """Compute seam-like difference stats at the boundary of a mask.

    Intended use: quantify how the fused product diverges from the river-only raster
    right at the river-domain transition (a common artifact zone).

    Parameters
    ----------
    river_raster : str
        Raster of river-bottom (or river patch) elevations on the output grid.
    fused_raster : str
        Raster of fused/combined elevations on the same grid.
    mask_raster : str
        Raster mask defining river domain (non-zero = river).
    mask_threshold : float
        Threshold for mask_raster to be considered True.
    boundary_mode : str
        "inner" -> boundary pixels inside the mask only
        "both"  -> boundary pixels on both sides (inner + immediate exterior)

    Returns
    -------
    dict
        Summary statistics and counts. Empty stats if no valid pixels.
    """

    import numpy as np
    import rasterio

    def _read(path: str):
        with rasterio.open(path) as ds:
            a = ds.read(1)
            nodata = ds.nodata
        return a, nodata

    r, r_nodata = _read(river_raster)
    f, f_nodata = _read(fused_raster)
    m, m_nodata = _read(mask_raster)

    if (r.shape != f.shape) or (r.shape != m.shape):
        return {
            "ok": False,
            "reason": "shape_mismatch",
            "river_shape": list(r.shape),
            "fused_shape": list(f.shape),
            "mask_shape": list(m.shape),
        }

    m_valid = np.ones_like(m, dtype=bool)
    if m_nodata is not None:
        m_valid &= (m != m_nodata)
    mask = m_valid & (m.astype("float64") > float(mask_threshold))

    if not mask.any():
        return {"ok": False, "reason": "empty_mask", "n_mask": int(mask.sum())}

    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    boundary_inner = np.zeros_like(mask, dtype=bool)
    for dy, dx in shifts:
        rolled = np.roll(mask, shift=(dy, dx), axis=(0, 1))
        boundary_inner |= (mask & ~rolled)

    if boundary_mode not in ("inner", "both"):
        boundary_mode = "inner"

    boundary = boundary_inner
    if boundary_mode == "both":
        boundary_outer = np.zeros_like(mask, dtype=bool)
        for dy, dx in shifts:
            rolled = np.roll(mask, shift=(dy, dx), axis=(0, 1))
            boundary_outer |= (~mask & rolled)
        boundary |= boundary_outer

    valid = boundary
    if r_nodata is not None:
        valid &= (r != r_nodata)
    if f_nodata is not None:
        valid &= (f != f_nodata)

    if not valid.any():
        return {"ok": False, "reason": "no_valid_boundary_pixels", "n_valid": int(valid.sum())}

    diff = (f.astype("float64") - r.astype("float64"))[valid]
    ad = np.abs(diff)

    def _pct(x, p):
        return float(np.percentile(x, p)) if x.size else float("nan")

    return {
        "ok": True,
        "n_valid": int(diff.size),
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std(ddof=0)),
        "diff_rmse": float(np.sqrt((diff * diff).mean())),
        "abs_p50": _pct(ad, 50),
        "abs_p90": _pct(ad, 90),
        "abs_p95": _pct(ad, 95),
        "abs_p99": _pct(ad, 99),
        "boundary_mode": boundary_mode,
        "mask_threshold": float(mask_threshold),
    }



def compute_array_overlap_identity_metrics(a: np.ndarray, b: np.ndarray, *, nodata: Optional[float] = None) -> Dict[str, Any]:
    """Compute identity stats over common valid pixels for same-grid overlap checks."""
    arr_a = np.asarray(a, dtype='float64')
    arr_b = np.asarray(b, dtype='float64')
    ny = min(arr_a.shape[0], arr_b.shape[0])
    nx = min(arr_a.shape[1], arr_b.shape[1])
    arr_a = arr_a[:ny, :nx]
    arr_b = arr_b[:ny, :nx]
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    if nodata is not None:
        valid &= valid_mask(arr_a, nodata) & valid_mask(arr_b, nodata)
    n_valid = int(valid.sum())
    if n_valid <= 0:
        return {'status': 'no_valid', 'n_valid': 0}
    dv = (arr_a - arr_b)[valid]
    absdv = np.abs(dv)
    exact = int(np.count_nonzero(absdv == 0.0))
    return {
        'status': 'ok',
        'n_valid': n_valid,
        'mean_abs': float(absdv.mean()),
        'rmse': float(np.sqrt((dv * dv).mean())),
        'max_abs': float(absdv.max()),
        'p95_abs': float(np.quantile(absdv, 0.95)),
        'exact_identity_fraction': float(exact / max(n_valid, 1)),
    }


def compute_raster_overlap_identity_metrics(raster_a: str | Path, raster_b: str | Path) -> Dict[str, Any]:
    import rasterio
    ra = Path(raster_a)
    rb = Path(raster_b)
    if not ra.exists() or not rb.exists():
        raise FileNotFoundError(f'missing overlap raster(s): {ra}, {rb}')
    with rasterio.open(ra) as da, rasterio.open(rb) as db:
        if da.crs != db.crs or da.transform != db.transform or da.width != db.width or da.height != db.height:
            return {'status': 'not_aligned', 'raster_a': str(ra), 'raster_b': str(rb)}
        a = da.read(1)
        b = db.read(1)
        nd = da.nodata if da.nodata is not None else db.nodata
        out = compute_array_overlap_identity_metrics(a, b, nodata=nd)
        out.update({'raster_a': str(ra), 'raster_b': str(rb)})
        return out





def _read_overlap_arrays(ds_a, ds_b):
    import rasterio
    from rasterio.windows import from_bounds

    oxmin = max(ds_a.bounds.left, ds_b.bounds.left)
    oymin = max(ds_a.bounds.bottom, ds_b.bounds.bottom)
    oxmax = min(ds_a.bounds.right, ds_b.bounds.right)
    oymax = min(ds_a.bounds.top, ds_b.bounds.top)
    if oxmax <= oxmin or oymax <= oymin:
        return None
    wa = from_bounds(oxmin, oymin, oxmax, oymax, transform=ds_a.transform).round_offsets().round_lengths()
    wb = from_bounds(oxmin, oymin, oxmax, oymax, transform=ds_b.transform).round_offsets().round_lengths()
    arr_a = sanitize_array(ds_a.read(1, window=wa, masked=False), ds_a.nodata, dtype=np.float32)
    arr_b = sanitize_array(ds_b.read(1, window=wb, masked=False), ds_b.nodata, dtype=np.float32)
    ny = min(arr_a.shape[0], arr_b.shape[0])
    nx = min(arr_a.shape[1], arr_b.shape[1])
    if ny <= 0 or nx <= 0:
        return None
    return arr_a[:ny, :nx], arr_b[:ny, :nx], (oxmin, oymin, oxmax, oymax)


def compute_raster_trusted_interior_identity_metrics(
    raster_a: str | Path,
    raster_b: str | Path,
    trusted_a: str | Path,
    trusted_b: str | Path,
    *,
    nodata: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute raster overlap identity metrics restricted to common trusted interior."""
    import rasterio

    ra = Path(raster_a)
    rb = Path(raster_b)
    ta = Path(trusted_a)
    tb = Path(trusted_b)
    for path in (ra, rb, ta, tb):
        if not path.exists():
            raise FileNotFoundError(str(path))

    with rasterio.open(ra) as da, rasterio.open(rb) as db, rasterio.open(ta) as dta, rasterio.open(tb) as dtb:
        if da.crs != db.crs or da.crs != dta.crs or da.crs != dtb.crs:
            raise ValueError('CRS mismatch among raster/trusted inputs')
        pair = _read_overlap_arrays(da, db)
        mask_pair = _read_overlap_arrays(dta, dtb)
        if pair is None or mask_pair is None:
            return {'status': 'no_valid', 'reason': 'no_overlap'}
        a, b, _ = pair
        ma, mb, _ = mask_pair
        ny = min(a.shape[0], b.shape[0], ma.shape[0], mb.shape[0])
        nx = min(a.shape[1], b.shape[1], ma.shape[1], mb.shape[1])
        a = a[:ny, :nx].astype('float64', copy=False)
        b = b[:ny, :nx].astype('float64', copy=False)
        trusted = (ma[:ny, :nx] > 0) & (mb[:ny, :nx] > 0)
        nd = nodata
        if nd is None:
            nd = da.nodata if da.nodata is not None else db.nodata
        valid = trusted & np.isfinite(a) & np.isfinite(b)
        if nd is not None:
            valid &= (a != nd) & (b != nd)
        n_valid = int(valid.sum())
        if n_valid <= 0:
            return {'status': 'no_valid', 'n_valid': 0, 'trusted_overlap_pixels': int(trusted.sum()), 'reason': 'no_valid_trusted_overlap'}
        dv = (a - b)[valid]
        absdv = np.abs(dv)
        exact = absdv == 0.0
        return {
            'status': 'ok',
            'n_valid': n_valid,
            'trusted_overlap_pixels': int(trusted.sum()),
            'mean_abs': float(absdv.mean()),
            'rmse': float(np.sqrt((dv * dv).mean())),
            'max_abs': float(absdv.max()),
            'exact_identity_fraction': float(exact.mean()) if exact.size else None,
        }


def compute_vector_trusted_interior_identity_metrics(
    vector_a: str | Path,
    vector_b: str | Path,
    trusted_a: str | Path,
    trusted_b: str | Path,
    *,
    key_fields: tuple[str, ...] = ('component_id', 'station_m'),
    compare_fields: tuple[str, ...] | None = None,
    station_precision: int = 3,
) -> Dict[str, Any]:
    """Compute vector identity metrics restricted to rows inside both trusted interiors."""
    try:
        import geopandas as gpd
        import pandas as pd
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('geopandas, pandas, and rasterio are required for trusted vector identity metrics') from exc

    va = Path(vector_a); vb = Path(vector_b); ta = Path(trusted_a); tb = Path(trusted_b)
    for path in (va, vb, ta, tb):
        if not path.exists():
            raise FileNotFoundError(str(path))
    gdf_a = gpd.read_file(va)
    gdf_b = gpd.read_file(vb)
    if gdf_a.empty or gdf_b.empty:
        return {'status': 'no_valid', 'n_valid': 0, 'reason': 'empty_vector'}

    def _attach_trusted(gdf, raster_path):
        with rasterio.open(raster_path) as ds:
            if gdf.crs and ds.crs and str(gdf.crs) != str(ds.crs):
                gdf = gdf.to_crs(ds.crs)
            coords = [(geom.x, geom.y) if geom is not None and not geom.is_empty else (np.nan, np.nan) for geom in gdf.geometry]
            vals = []
            for x, y in coords:
                if not np.isfinite(x) or not np.isfinite(y):
                    vals.append(0)
                    continue
                try:
                    vals.append(int(next(ds.sample([(x, y)]))[0] > 0))
                except Exception:
                    vals.append(0)
            out = gdf.copy()
            out['__trusted'] = np.asarray(vals, dtype=np.uint8)
            return out

    gdf_a = _attach_trusted(gdf_a, ta)
    gdf_b = _attach_trusted(gdf_b, tb)
    frame_a = gdf_a.drop(columns='geometry', errors='ignore').copy()
    frame_b = gdf_b.drop(columns='geometry', errors='ignore').copy()
    join_keys = []
    for key in key_fields:
        if key not in frame_a.columns or key not in frame_b.columns:
            return {'status': 'missing_keys', 'missing_key': key, 'vector_a': str(va), 'vector_b': str(vb)}
        join_key = str(key)
        if key == 'station_m':
            join_key = '__station_key'
            frame_a[join_key] = pd.to_numeric(frame_a[key], errors='coerce').round(int(station_precision))
            frame_b[join_key] = pd.to_numeric(frame_b[key], errors='coerce').round(int(station_precision))
        else:
            frame_a[join_key] = frame_a[key]
            frame_b[join_key] = frame_b[key]
        join_keys.append(join_key)
    if compare_fields is None:
        preferred = ['graph_backbone_z_m','graph_hard_lock','graph_junction_constrained','graph_solution_mode','graph_solver_support_class','graph_uncertainty_class','graph_residual_to_candidate_z_m','graph_unsupported_span_m']
        compare_fields = tuple(f for f in preferred if f in frame_a.columns and f in frame_b.columns)
    else:
        compare_fields = tuple(f for f in compare_fields if f in frame_a.columns and f in frame_b.columns)
    if not compare_fields:
        return {'status': 'no_valid', 'n_valid': 0, 'reason': 'no_shared_compare_fields'}
    merged = frame_a[join_keys + ['__trusted'] + list(compare_fields)].merge(
        frame_b[join_keys + ['__trusted'] + list(compare_fields)], on=join_keys, how='inner', suffixes=('_a','_b'))
    if merged.empty:
        return {'status': 'no_valid', 'n_valid': 0, 'reason': 'no_overlap_rows'}
    trusted = (merged['__trusted_a'] > 0) & (merged['__trusted_b'] > 0)
    merged = merged.loc[trusted].copy()
    n_valid = int(len(merged))
    if n_valid <= 0:
        return {'status': 'no_valid', 'n_valid': 0, 'trusted_overlap_rows': 0, 'reason': 'no_trusted_overlap_rows'}
    field_metrics = {}
    overall_max_abs = 0.0
    exact_all = np.ones(n_valid, dtype=bool)
    for field in compare_fields:
        col_a = merged[f'{field}_a']
        col_b = merged[f'{field}_b']
        metric: Dict[str, Any] = {'type': 'categorical'}
        try:
            arr_a = pd.to_numeric(col_a, errors='coerce').to_numpy(dtype='float64')
            arr_b = pd.to_numeric(col_b, errors='coerce').to_numpy(dtype='float64')
            valid = np.isfinite(arr_a) & np.isfinite(arr_b)
        except Exception:
            valid = np.zeros(n_valid, dtype=bool)
            arr_a = arr_b = None
        if valid.any():
            dv = arr_a[valid] - arr_b[valid]
            absdv = np.abs(dv)
            exact = absdv == 0.0
            metric = {'type':'numeric','n_valid':int(valid.sum()),'max_abs':float(absdv.max()) if absdv.size else 0.0,'mean_abs':float(absdv.mean()) if absdv.size else 0.0,'rmse':float(np.sqrt((dv*dv).mean())) if absdv.size else 0.0,'exact_identity_fraction':float(exact.mean()) if exact.size else None}
            exact_field = np.zeros(n_valid, dtype=bool); exact_field[np.where(valid)[0]] = exact
            exact_all &= exact_field
            overall_max_abs = max(overall_max_abs, metric['max_abs'])
        else:
            sa = col_a.fillna('__nan__').astype(str)
            sb = col_b.fillna('__nan__').astype(str)
            exact = (sa == sb).to_numpy(dtype=bool)
            metric = {'type':'categorical','n_valid':n_valid,'max_abs':0.0 if bool(exact.all()) else 1.0,'exact_identity_fraction':float(exact.mean()) if exact.size else None}
            exact_all &= exact
            overall_max_abs = max(overall_max_abs, metric['max_abs'])
        field_metrics[field] = metric
    return {'status':'ok','n_valid':n_valid,'trusted_overlap_rows':n_valid,'field_metrics':field_metrics,'max_abs':float(overall_max_abs),'exact_identity_fraction':float(exact_all.mean()) if exact_all.size else None}
def compute_vector_overlap_identity_metrics(
    vector_a: str | Path,
    vector_b: str | Path,
    *,
    key_fields: tuple[str, ...] = ("component_id", "station_m"),
    compare_fields: tuple[str, ...] | None = None,
    station_precision: int = 3,
) -> Dict[str, Any]:
    """Compute overlap identity stats for vector artifacts on shared keys.

    Designed for graph-backed river diagnostics where the same stationed network
    should agree across nested/overlapping AOIs over the trusted interior.
    """
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("geopandas and pandas are required for vector overlap identity metrics") from exc

    va = Path(vector_a)
    vb = Path(vector_b)
    if not va.exists() or not vb.exists():
        raise FileNotFoundError(f'missing overlap vector(s): {va}, {vb}')

    gdf_a = gpd.read_file(va)
    gdf_b = gpd.read_file(vb)
    if gdf_a.empty or gdf_b.empty:
        return {'status': 'no_valid', 'n_valid': 0, 'reason': 'empty_vector'}

    join_keys = []
    frame_a = gdf_a.drop(columns='geometry', errors='ignore').copy()
    frame_b = gdf_b.drop(columns='geometry', errors='ignore').copy()
    for key in key_fields:
        if key not in frame_a.columns or key not in frame_b.columns:
            return {'status': 'missing_keys', 'missing_key': key, 'vector_a': str(va), 'vector_b': str(vb)}
        join_key = str(key)
        if key == 'station_m':
            join_key = '__station_key'
            frame_a[join_key] = pd.to_numeric(frame_a[key], errors='coerce').round(int(station_precision))
            frame_b[join_key] = pd.to_numeric(frame_b[key], errors='coerce').round(int(station_precision))
        else:
            frame_a[join_key] = frame_a[key]
            frame_b[join_key] = frame_b[key]
        join_keys.append(join_key)

    if compare_fields is None:
        preferred = [
            'graph_backbone_z_m',
            'graph_hard_lock',
            'graph_junction_constrained',
            'graph_solution_mode',
            'graph_solver_support_class',
            'graph_uncertainty_class',
            'graph_residual_to_candidate_z_m',
            'graph_unsupported_span_m',
        ]
        compare_fields = tuple(f for f in preferred if f in frame_a.columns and f in frame_b.columns)
    else:
        compare_fields = tuple(f for f in compare_fields if f in frame_a.columns and f in frame_b.columns)
    if not compare_fields:
        return {'status': 'no_valid', 'n_valid': 0, 'reason': 'no_shared_compare_fields'}

    merged = frame_a[join_keys + list(compare_fields)].merge(
        frame_b[join_keys + list(compare_fields)],
        on=join_keys,
        how='inner',
        suffixes=('_a', '_b'),
    )
    n_valid = int(len(merged))
    if n_valid <= 0:
        return {'status': 'no_valid', 'n_valid': 0, 'reason': 'no_overlap_rows'}

    field_metrics: Dict[str, Any] = {}
    overall_max_abs = 0.0
    exact_all = np.ones(n_valid, dtype=bool)
    for field in compare_fields:
        col_a = merged[f'{field}_a']
        col_b = merged[f'{field}_b']
        metric: Dict[str, Any] = {'type': 'categorical'}
        try:
            arr_a = pd.to_numeric(col_a, errors='coerce').to_numpy(dtype='float64')
            arr_b = pd.to_numeric(col_b, errors='coerce').to_numpy(dtype='float64')
            valid = np.isfinite(arr_a) & np.isfinite(arr_b)
        except Exception:
            valid = np.zeros(n_valid, dtype=bool)
            arr_a = arr_b = None
        if valid.any():
            dv = arr_a[valid] - arr_b[valid]
            absdv = np.abs(dv)
            exact = absdv == 0.0
            metric = {
                'type': 'numeric',
                'n_valid': int(valid.sum()),
                'max_abs': float(absdv.max()) if absdv.size else 0.0,
                'mean_abs': float(absdv.mean()) if absdv.size else 0.0,
                'rmse': float(np.sqrt((dv * dv).mean())) if absdv.size else 0.0,
                'exact_identity_fraction': float(exact.sum() / max(int(valid.sum()), 1)),
            }
            overall_max_abs = max(overall_max_abs, float(metric['max_abs']))
            all_exact_field = np.zeros(n_valid, dtype=bool)
            all_exact_field[valid] = exact
            exact_all &= all_exact_field
        else:
            arr_a_obj = col_a.fillna('__nan__').astype(str).to_numpy()
            arr_b_obj = col_b.fillna('__nan__').astype(str).to_numpy()
            exact = arr_a_obj == arr_b_obj
            metric = {
                'type': 'categorical',
                'n_valid': int(n_valid),
                'mismatch_count': int((~exact).sum()),
                'exact_identity_fraction': float(exact.sum() / max(n_valid, 1)),
                'max_abs': 0.0 if bool(np.all(exact)) else 1.0,
            }
            overall_max_abs = max(overall_max_abs, float(metric['max_abs']))
            exact_all &= exact
        field_metrics[field] = metric

    return {
        'status': 'ok',
        'vector_a': str(va),
        'vector_b': str(vb),
        'n_valid': n_valid,
        'key_fields': list(key_fields),
        'compare_fields': list(compare_fields),
        'station_precision': int(station_precision),
        'max_abs': float(overall_max_abs),
        'exact_identity_fraction': float(exact_all.sum() / max(n_valid, 1)),
        'field_metrics': field_metrics,
    }
