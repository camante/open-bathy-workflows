#!/usr/bin/env python
"""Regression metrics for OBW river outputs.

This script is intentionally lightweight and conservative:
- No workflow changes; it only reads outputs.
- Produces a per-run metrics.json and a cross-run metrics_summary.csv.

Usage examples:
  python regression_metrics.py --run-dir regression/a_typical/bed+hydro
  python regression_metrics.py --root regression --glob "**/" 

It attempts to compute:
  - Mask connected component count (if scipy is available)
  - Mask nonzero area/pixel count
  - Depth/bed raster stats (min/mean/median/p95/max)
  - Nodata fraction inside mask
  - Constant raster detection
  - Boundary gradient std proxy (edge-artifact score)

Optional (best-effort):
  - Skeleton longitudinal max |dz/ds| when a skeleton/flowline vector is present
    and geopandas is available.
"""

from __future__ import annotations

import logging
log = logging.getLogger(__name__)
import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


np = _try_import("numpy")
rasterio = _try_import("rasterio")


@dataclass
class RasterStats:
    n: int = 0
    nodata_frac: Optional[float] = None
    min: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    p95: Optional[float] = None
    max: Optional[float] = None
    is_constant: Optional[bool] = None


def _find_first(run_dir: Path, candidates: List[str]) -> Optional[Path]:
    for c in candidates:
        p = run_dir / c
        if p.exists():
            return p
    # fallback: try recursive match
    for c in candidates:
        matches = list(run_dir.rglob(c))
        if matches:
            return matches[0]
    return None


def _read_mask(mask_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """Return mask array (bool) and metadata."""
    if rasterio is None or np is None:
        raise RuntimeError("rasterio and numpy are required for raster metrics")
    with rasterio.open(mask_path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        if nodata is not None:
            valid = arr != nodata
        else:
            valid = np.ones(arr.shape, dtype=bool)
        # channel mask is assumed 0/1 or 0/nonzero
        m = (arr != 0) & valid
        meta = {
            "shape": [int(arr.shape[0]), int(arr.shape[1])],
            "crs": str(ds.crs) if ds.crs else None,
            "transform": tuple(ds.transform) if ds.transform else None,
            "nodata": nodata,
            "pixel_size": (abs(ds.transform.a), abs(ds.transform.e)) if ds.transform else None,
        }
        return m, meta


def _raster_stats(raster_path: Path, mask: Optional[Any] = None) -> RasterStats:
    if rasterio is None or np is None:
        raise RuntimeError("rasterio and numpy are required for raster metrics")
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1).astype("float64", copy=False)
        nodata = ds.nodata
        if nodata is not None:
            valid = arr != nodata
        else:
            valid = np.isfinite(arr)
        if mask is not None:
            valid = valid & mask
        vals = arr[valid]
        rs = RasterStats(n=int(vals.size))
        if vals.size == 0:
            rs.nodata_frac = 1.0
            rs.is_constant = None
            return rs
        # nodata fraction inside mask (or whole raster if mask is None)
        denom = int(mask.sum()) if mask is not None else int(arr.size)
        if denom > 0:
            rs.nodata_frac = float(1.0 - (vals.size / denom))
        rs.min = float(np.nanmin(vals))
        rs.max = float(np.nanmax(vals))
        rs.mean = float(np.nanmean(vals))
        rs.median = float(np.nanmedian(vals))
        rs.p95 = float(np.nanpercentile(vals, 95))
        rs.is_constant = bool(math.isfinite(rs.min) and math.isfinite(rs.max) and rs.min == rs.max)
        return rs


def _boundary_gradient_std(raster_path: Path, mask: Any) -> Optional[float]:
    """Proxy for edge artifacts: stddev of |grad| near mask boundary."""
    if rasterio is None or np is None:
        return None
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1).astype("float64", copy=False)
        nodata = ds.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid = valid & (arr != nodata)
        valid = valid & mask
        if valid.sum() < 10:
            return None
        # Identify boundary pixels: mask pixel with any 4-neighbor outside mask
        m = mask.astype(bool)
        up = np.zeros_like(m); up[1:] = m[:-1]
        dn = np.zeros_like(m); dn[:-1] = m[1:]
        lf = np.zeros_like(m); lf[:, 1:] = m[:, :-1]
        rt = np.zeros_like(m); rt[:, :-1] = m[:, 1:]
        interior = m & up & dn & lf & rt
        boundary = m & (~interior)
        boundary = boundary & valid
        if boundary.sum() < 10:
            return None
        # simple gradient magnitude using forward differences
        gx = np.zeros_like(arr)
        gy = np.zeros_like(arr)
        gx[:, :-1] = arr[:, 1:] - arr[:, :-1]
        gy[:-1, :] = arr[1:, :] - arr[:-1, :]
        g = np.hypot(gx, gy)
        vals = g[boundary]
        if vals.size < 10:
            return None
        return float(np.nanstd(vals))


def _connected_components(mask: Any) -> Optional[int]:
    """Return count of connected components for True mask (4-connected) if scipy available."""
    if np is None:
        return None
    scipy_nd = None
    try:
        from scipy import ndimage as scipy_nd  # type: ignore
    except Exception:
        return None
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labeled, n = scipy_nd.label(mask.astype(np.uint8), structure=struct)
    _ = labeled  # unused
    return int(n)


def _find_skeleton_vector(run_dir: Path) -> Optional[Path]:
    candidates = [
        "skeleton.gpkg",
        "river_skeleton.gpkg",
        "river_network.gpkg",
        "river.gpkg",
        "cross_sections.gpkg",
    ]
    return _find_first(run_dir, candidates)


def _skeleton_max_slope(
    line_path: Path,
    raster_path: Path,
    step_m: float = 25.0,
) -> Optional[float]:
    """Best-effort max |dz/ds| along line(s). Requires geopandas."""
    if rasterio is None or np is None:
        return None
    gpd = _try_import("geopandas")
    shapely = _try_import("shapely")
    if gpd is None or shapely is None:
        return None
    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import LineString  # type: ignore
    except Exception:
        return None

    try:
        gdf = gpd.read_file(line_path)
    except Exception:
        return None
    if gdf.empty or "geometry" not in gdf:
        return None
    lines = [geom for geom in gdf.geometry if geom is not None and geom.geom_type in ("LineString", "MultiLineString")]
    if not lines:
        return None

    with rasterio.open(raster_path) as ds:
        # Reproject lines to raster CRS if needed
        try:
            if gdf.crs and ds.crs and str(gdf.crs) != str(ds.crs):
                gdf2 = gdf.to_crs(ds.crs)
                lines = [geom for geom in gdf2.geometry if geom is not None and geom.geom_type in ("LineString", "MultiLineString")]
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        max_slope = 0.0
        for geom in lines:
            # explode multilines
            if geom.geom_type == "MultiLineString":
                parts = list(geom.geoms)
            else:
                parts = [geom]
            for ln in parts:
                if ln.length <= step_m * 2:
                    continue
                n = max(3, int(ln.length / step_m) + 1)
                dists = np.linspace(0.0, ln.length, n)
                pts = [ln.interpolate(float(d)) for d in dists]
                coords = [(p.x, p.y) for p in pts]
                zs = np.array([v[0] for v in ds.sample(coords)], dtype="float64")
                # drop nodata/nans
                if ds.nodata is not None:
                    ok = (zs != ds.nodata) & np.isfinite(zs)
                else:
                    ok = np.isfinite(zs)
                if ok.sum() < 3:
                    continue
                zs = zs[ok]
                ss = dists[ok]
                dz = np.diff(zs)
                ds_ = np.diff(ss)
                good = ds_ > 0
                if good.sum() == 0:
                    continue
                slopes = np.abs(dz[good] / ds_[good])
                local = float(np.nanmax(slopes)) if slopes.size else 0.0
                if local > max_slope:
                    max_slope = local
        return float(max_slope) if max_slope > 0 else None


def compute_run_metrics(run_dir: Path) -> Dict[str, Any]:
    run_dir = run_dir.resolve()

    mask_path = _find_first(
        run_dir,
        [
            "river_channel_mask.tif",
            "channel_mask.tif",
            "mask.tif",
            "river_domain_mask.tif",
        ],
    )
    depth_path = _find_first(
        run_dir,
        [
            "river_depth_terrain_patch.tif",
            "river_depth.tif",
            "depth.tif",
            "river_depth_final.tif",
        ],
    )
    bed_path = _find_first(
        run_dir,
        [
            "river_bed.tif",
            "bed.tif",
            "river_bed_final.tif",
        ],
    )

    metrics: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "files": {
            "mask": str(mask_path) if mask_path else None,
            "depth": str(depth_path) if depth_path else None,
            "bed": str(bed_path) if bed_path else None,
        },
    }

    if mask_path is None:
        mask_path = _mask_from_report(run_dir)

    if mask_path is None:
        metrics["error"] = "No channel mask found"
        return metrics
    if rasterio is None or np is None:
        metrics["error"] = "Missing dependencies: rasterio and numpy are required"
        return metrics

    mask, mask_meta = _read_mask(mask_path)
    metrics["mask"] = {
        "meta": mask_meta,
        "nonzero_pixels": int(mask.sum()),
    }
    # Connected components (optional)
    cc = _connected_components(mask)
    metrics["mask"]["connected_components"] = cc
    # Approx area
    px = mask_meta.get("pixel_size")
    if px and all(px):
        metrics["mask"]["area_m2"] = float(mask.sum()) * float(px[0]) * float(px[1])

    if depth_path and depth_path.exists():
        ds = _raster_stats(depth_path, mask=mask)
        metrics["depth"] = asdict(ds)
        metrics["depth"]["boundary_grad_std"] = _boundary_gradient_std(depth_path, mask)
    else:
        metrics["depth"] = None

    if bed_path and bed_path.exists():
        bs = _raster_stats(bed_path, mask=mask)
        metrics["bed"] = asdict(bs)
        metrics["bed"]["boundary_grad_std"] = _boundary_gradient_std(bed_path, mask)
    else:
        metrics["bed"] = None

    # Optional skeleton slope check
    skel_vec = _find_skeleton_vector(run_dir)
    if skel_vec and bed_path and bed_path.exists():
        max_slope = _skeleton_max_slope(skel_vec, bed_path, step_m=25.0)
        metrics["skeleton"] = {
            "vector": str(skel_vec),
            "bed_max_abs_slope_m_per_m": max_slope,
        }
    else:
        metrics["skeleton"] = None

    return metrics


def _collect_run_dirs(root: Path, pattern: str) -> List[Path]:
    # pattern is a directory glob relative to root
    dirs = [p for p in root.glob(pattern) if p.is_dir()]
    # filter out hidden and cache
    out = []
    for d in dirs:
        name = d.name
        if name.startswith(".") or name == "__pycache__":
            continue
        out.append(d)
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute regression metrics for OBW river runs")
    ap.add_argument("--run-dir", action="append", default=[], help="A single run directory (repeatable)")
    ap.add_argument("--root", default=None, help="Root folder containing runs")
    ap.add_argument("--glob", default="**/*", help="Glob under --root to find run dirs")
    ap.add_argument("--out", default=None, help="Output directory (defaults to run dir or root)")
    ap.add_argument("--write-csv", action="store_true", help="Write metrics_summary.csv")
    args = ap.parse_args()

    run_dirs: List[Path] = []
    for rd in args.run_dir:
        run_dirs.append(Path(rd))
    if args.root:
        run_dirs.extend(_collect_run_dirs(Path(args.root), args.glob))
    run_dirs = [d for d in run_dirs if d.exists() and d.is_dir()]
    if not run_dirs:
        log.info("No run directories found. Provide --run-dir or --root.")
        return 2

    results: List[Dict[str, Any]] = []
    for d in run_dirs:
        try:
            m = compute_run_metrics(d)
        except Exception as e:
            m = {"run_dir": str(d.resolve()), "error": f"{type(e).__name__}: {e}"}
        results.append(m)
        # per-run json
        out_dir = Path(args.out) if args.out else d
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "metrics.json"
        out_json.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        log.info(f"[OK] wrote {out_json}")

    if args.write_csv:
        import csv

        out_dir = Path(args.out) if args.out else (Path(args.root) if args.root else run_dirs[0])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "metrics_summary.csv"
        # Flatten a few key fields
        rows: List[Dict[str, Any]] = []
        for r in results:
            row = {
                "run_dir": r.get("run_dir"),
                "mask_nonzero_pixels": (r.get("mask") or {}).get("nonzero_pixels") if isinstance(r.get("mask"), dict) else None,
                "mask_components": (r.get("mask") or {}).get("connected_components") if isinstance(r.get("mask"), dict) else None,
                "depth_n": (r.get("depth") or {}).get("n") if isinstance(r.get("depth"), dict) else None,
                "depth_nodata_frac": (r.get("depth") or {}).get("nodata_frac") if isinstance(r.get("depth"), dict) else None,
                "depth_median": (r.get("depth") or {}).get("median") if isinstance(r.get("depth"), dict) else None,
                "depth_p95": (r.get("depth") or {}).get("p95") if isinstance(r.get("depth"), dict) else None,
                "depth_max": (r.get("depth") or {}).get("max") if isinstance(r.get("depth"), dict) else None,
                "depth_constant": (r.get("depth") or {}).get("is_constant") if isinstance(r.get("depth"), dict) else None,
                "depth_edge_grad_std": (r.get("depth") or {}).get("boundary_grad_std") if isinstance(r.get("depth"), dict) else None,
                "bed_median": (r.get("bed") or {}).get("median") if isinstance(r.get("bed"), dict) else None,
                "bed_max_slope": (r.get("skeleton") or {}).get("bed_max_abs_slope_m_per_m") if isinstance(r.get("skeleton"), dict) else None,
                "error": r.get("error"),
            }
            rows.append(row)
        fieldnames = list(rows[0].keys())
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        log.info(f"[OK] wrote {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
