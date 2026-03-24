from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pyproj import Transformer

from support_classes import SUPPORT_CLASS_CODE_TO_NAME
import logging

log = logging.getLogger(__name__)

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    log.debug("benchmark_workflow_stage: suppressed exception", exc_info=True)
    gpd = None


def _infer_col(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        hit = lower_map.get(str(cand).lower())
        if hit is not None:
            return str(hit)
    raise ValueError(f"Could not find {label} column. Tried {list(candidates)}. Columns={list(df.columns)}")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".gpkg", ".shp", ".geojson"}:
        if gpd is None:
            raise ValueError(f"Benchmark holdout format {suffix} requires geopandas: {path}")
        gdf = gpd.read_file(path)
        if gdf.empty:
            return pd.DataFrame()
        if gdf.geometry is not None:
            geom = gdf.geometry
            if getattr(geom, "x", None) is not None and getattr(geom, "y", None) is not None:
                gdf = gdf.copy()
                gdf["x"] = geom.x
                gdf["y"] = geom.y
        if gdf.crs is not None and gdf.crs.to_epsg() is not None and "crs" not in gdf.columns:
            gdf = gdf.copy()
            gdf["crs"] = f"EPSG:{gdf.crs.to_epsg()}"
        return pd.DataFrame(gdf.drop(columns=[c for c in ["geometry"] if c in gdf.columns]))
    raise ValueError(f"Unsupported benchmark holdout format: {path}")


def _sample_raster(raster_path: Path, x: np.ndarray, y: np.ndarray, src_epsg: int) -> np.ndarray:
    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        dst_epsg = ds.crs.to_epsg()
        if dst_epsg is None:
            raise ValueError(f"Raster CRS has no EPSG code: {raster_path} ({ds.crs})")
        if src_epsg != dst_epsg:
            tx = Transformer.from_crs(f"EPSG:{src_epsg}", ds.crs, always_xy=True)
            xs, ys = tx.transform(x, y)
        else:
            xs, ys = x, y
        vals = np.array([v[0] for v in ds.sample(list(zip(xs, ys)))], dtype="float64")
        nodata = ds.nodata
        if nodata is not None:
            vals[np.isclose(vals, nodata)] = np.nan
        return vals


def _sample_mask(raster_path: Optional[Path], x: np.ndarray, y: np.ndarray, src_epsg: int) -> Optional[np.ndarray]:
    if raster_path is None:
        return None
    vals = _sample_raster(raster_path, x, y, src_epsg=src_epsg)
    return np.isfinite(vals) & (vals != 0)




def _resolve_auto_holdout_candidate(*, args, report: dict, out_dir: Path) -> Optional[Path]:
    explicit = getattr(args, "benchmark_holdout", None)
    if explicit:
        p = Path(str(explicit)).resolve()
        return p if p.exists() else None
    auth = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base"), dict) else {}
    for sec_name in ("sdb_guidance", "river_guidance"):
        sec = auth.get(sec_name, {}) if isinstance(auth.get(sec_name), dict) else {}
        cand = sec.get("path")
        if cand:
            p = Path(str(cand))
            if not p.is_absolute():
                p = (out_dir / p).resolve()
            if p.exists():
                return p
    return None


def _infer_points_epsg_from_df(df: pd.DataFrame, x_name: str, y_name: str, *, baseline_raster: Path, final_raster: Path, explicit_epsg: Optional[int]) -> int:
    if explicit_epsg and int(explicit_epsg) > 0:
        return int(explicit_epsg)
    if "crs" in df.columns:
        vals = [str(v) for v in df["crs"].dropna().unique().tolist()]
        if len(vals) == 1 and vals[0].upper().startswith("EPSG:"):
            try:
                return int(vals[0].split(":", 1)[1])
            except Exception:
                log.debug("Failed to parse CRS from column value %s", vals[0], exc_info=True)
    x = pd.to_numeric(df[x_name], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(df[y_name], errors="coerce").to_numpy(dtype="float64")
    return _infer_points_epsg(x=x, y=y, x_name=x_name, y_name=y_name, baseline_raster=baseline_raster, final_raster=final_raster)


def _transform_xy_to_raster_crs(x: np.ndarray, y: np.ndarray, src_epsg: int, raster_path: Path) -> tuple[np.ndarray, np.ndarray, Any]:
    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        dst_crs = ds.crs
        dst_epsg = dst_crs.to_epsg()
        if dst_epsg is None:
            raise ValueError(f"Raster CRS has no EPSG code: {raster_path} ({ds.crs})")
        if src_epsg != dst_epsg:
            tx = Transformer.from_crs(f"EPSG:{src_epsg}", dst_crs, always_xy=True)
            xx, yy = tx.transform(x, y)
        else:
            xx, yy = x, y
    return np.asarray(xx, dtype="float64"), np.asarray(yy, dtype="float64"), dst_crs


def _build_auto_holdout(*, candidate_path: Path, bench_dir: Path, baseline_raster: Path, final_raster: Path, points_epsg_arg: Optional[int], x_col: Optional[str], y_col: Optional[str], z_col: Optional[str], holdout_frac: float, holdout_min_points: int, holdout_seed: int, logger) -> tuple[Path, int, str, str, str, dict[str, Any]]:
    df = _read_table(candidate_path)
    if df.empty:
        raise ValueError(f"Auto-holdout candidate pool is empty: {candidate_path}")
    x_name = x_col or _infer_col(df, ["x", "X", "easting", "lon", "longitude"], "x")
    y_name = y_col or _infer_col(df, ["y", "Y", "northing", "lat", "latitude"], "y")
    z_name = z_col or _infer_col(df, ["z", "Z", "depth_m", "z_m", "depth"], "z/depth")
    points_epsg = _infer_points_epsg_from_df(df, x_name, y_name, baseline_raster=baseline_raster, final_raster=final_raster, explicit_epsg=points_epsg_arg)

    x = pd.to_numeric(df[x_name], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(df[y_name], errors="coerce").to_numpy(dtype="float64")
    z = pd.to_numeric(df[z_name], errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    base = df.loc[finite].copy().reset_index(drop=True)
    if base.empty:
        raise ValueError(f"Auto-holdout candidate pool has no finite x/y/z rows: {candidate_path}")
    x = pd.to_numeric(base[x_name], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(base[y_name], errors="coerce").to_numpy(dtype="float64")
    xx, yy, dst_crs = _transform_xy_to_raster_crs(x, y, points_epsg, baseline_raster)

    with rasterio.open(baseline_raster) as ds:
        px = abs(ds.transform.a) or 1.0
        py = abs(ds.transform.e) or px
    block = max(float(max(px, py) * 25.0), 250.0)
    xmin = float(np.nanmin(xx))
    ymin = float(np.nanmin(yy))
    bx = np.floor((xx - xmin) / block).astype(int)
    by = np.floor((yy - ymin) / block).astype(int)
    block_ids = np.array([f"{a}_{b}" for a, b in zip(bx, by)], dtype=object)
    base["_bench_block_id"] = block_ids

    uniq, counts = np.unique(block_ids, return_counts=True)
    target = max(int(math.ceil(len(base) * float(holdout_frac))), int(holdout_min_points))
    target = min(target, len(base))
    rng = np.random.default_rng(int(holdout_seed))
    order = rng.permutation(len(uniq))
    chosen = []
    running = 0
    for idx in order:
        chosen.append(uniq[idx])
        running += int(counts[idx])
        if running >= target:
            break
    hold = base[base["_bench_block_id"].isin(chosen)].copy()
    hold["benchmark_holdout"] = 1
    hold["benchmark_holdout_method"] = "spatial_blocks"
    hold["benchmark_holdout_seed"] = int(holdout_seed)
    hold["benchmark_points_epsg"] = int(points_epsg)
    hold_path = bench_dir / "auto_holdout_points.csv"
    hold.to_csv(hold_path, index=False)
    receipt = {
        "candidate_path": str(candidate_path),
        "candidate_rows": int(len(df)),
        "finite_candidate_rows": int(len(base)),
        "points_epsg": int(points_epsg),
        "method": "spatial_blocks",
        "block_size_in_raster_crs_units": float(block),
        "block_crs": str(dst_crs),
        "holdout_frac_requested": float(holdout_frac),
        "holdout_min_points": int(holdout_min_points),
        "holdout_seed": int(holdout_seed),
        "selected_block_count": int(len(chosen)),
        "holdout_rows": int(len(hold)),
        "x_col": x_name,
        "y_col": y_name,
        "z_col": z_name,
        "path": str(hold_path),
    }
    receipt_path = bench_dir / "auto_holdout_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    logger.info("[BENCHMARK] Auto-holdout selected %s/%s rows from %s into %s", len(hold), len(base), candidate_path, hold_path)
    return hold_path, points_epsg, x_name, y_name, z_name, receipt


def _infer_points_epsg(*, x: np.ndarray, y: np.ndarray, x_name: str, y_name: str, baseline_raster: Path, final_raster: Path) -> int:
    x_lower = str(x_name).lower()
    y_lower = str(y_name).lower()
    looks_lonlat_name = x_lower in {"lon", "longitude"} or y_lower in {"lat", "latitude"}
    finite = np.isfinite(x) & np.isfinite(y)
    if np.any(finite):
        xf = x[finite]
        yf = y[finite]
        looks_lonlat_range = (np.nanmin(xf) >= -180.0 and np.nanmax(xf) <= 180.0 and np.nanmin(yf) >= -90.0 and np.nanmax(yf) <= 90.0)
        if looks_lonlat_name or looks_lonlat_range:
            return 4326

    for rp in (baseline_raster, final_raster):
        with rasterio.open(rp) as ds:
            if ds.crs is None:
                continue
            epsg = ds.crs.to_epsg()
            if epsg is None:
                continue
            if np.any(finite):
                xf = x[finite]
                yf = y[finite]
                left, bottom, right, top = ds.bounds
                if np.nanmin(xf) >= left and np.nanmax(xf) <= right and np.nanmin(yf) >= bottom and np.nanmax(yf) <= top:
                    return int(epsg)
    raise ValueError(
        "Could not infer benchmark point CRS automatically. Provide --benchmark-points-epsg, "
        f"or use lon/lat-like columns. x_col={x_name} y_col={y_name}"
    )


def _align_raster_to_template(src_path: Optional[Path], template_path: Path, *, dtype: str = "float64", nodata_value: float = np.nan, resampling: Resampling = Resampling.nearest) -> Optional[np.ndarray]:
    if src_path is None:
        return None
    with rasterio.open(template_path) as tds:
        out = np.full((tds.height, tds.width), nodata_value, dtype=dtype)
        with rasterio.open(src_path) as sds:
            src = sds.read(1)
            src_nodata = sds.nodata
            if src_nodata is not None and np.issubdtype(src.dtype, np.floating):
                src = src.astype("float64", copy=False)
                src[np.isclose(src, src_nodata)] = np.nan
            reproject(
                source=src,
                destination=out,
                src_transform=sds.transform,
                src_crs=sds.crs,
                dst_transform=tds.transform,
                dst_crs=tds.crs,
                src_nodata=src_nodata,
                dst_nodata=nodata_value,
                resampling=resampling,
            )
    return out


def _raster_diff_stats(diff: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool) & np.isfinite(diff)
    n = int(np.count_nonzero(m))
    if n == 0:
        return {
            "n": 0,
            "changed_pixels": 0,
            "changed_frac": np.nan,
            "mean_diff": np.nan,
            "mean_abs_diff": np.nan,
            "p95_abs_diff": np.nan,
            "max_abs_diff": np.nan,
        }
    vals = diff[m]
    abs_vals = np.abs(vals)
    changed = abs_vals > 1.0e-6
    return {
        "n": n,
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frac": float(np.count_nonzero(changed) / float(n)),
        "mean_diff": float(np.mean(vals)),
        "mean_abs_diff": float(np.mean(abs_vals)),
        "p95_abs_diff": float(np.percentile(abs_vals, 95.0)),
        "max_abs_diff": float(np.max(abs_vals)),
    }


def _compute_zone_diff_summary(*, baseline_raster: Path, final_raster: Path, support_class_raster: Optional[Path], river_mask_raster: Optional[Path], estuary_mask_raster: Optional[Path]) -> dict[str, Any]:
    final_arr = _align_raster_to_template(final_raster, final_raster, dtype="float64", nodata_value=np.nan, resampling=Resampling.bilinear)
    base_arr = _align_raster_to_template(baseline_raster, final_raster, dtype="float64", nodata_value=np.nan, resampling=Resampling.bilinear)
    if final_arr is None or base_arr is None:
        return {}
    diff = final_arr - base_arr
    finite = np.isfinite(final_arr) & np.isfinite(base_arr)
    zones: dict[str, dict[str, Any]] = {
        "overall": _raster_diff_stats(diff, finite),
    }
    support_arr = _align_raster_to_template(support_class_raster, final_raster, dtype="float64", nodata_value=np.nan, resampling=Resampling.nearest)
    if support_arr is not None:
        valid_support = finite & np.isfinite(support_arr)
        for code in sorted(int(c) for c in np.unique(support_arr[np.isfinite(support_arr)])):
            name = SUPPORT_CLASS_CODE_TO_NAME.get(code, f"support_class_{code}")
            zones[f"support_class::{name}"] = _raster_diff_stats(diff, valid_support & np.isclose(support_arr, code))
        auth_mask = valid_support & np.isclose(support_arr, 1)
        auth_stats = _raster_diff_stats(diff, auth_mask)
        zones["authoritative_locked"] = auth_stats
        zones["authoritative_locked_invariant"] = {
            "n": auth_stats["n"],
            "changed_pixels": auth_stats["changed_pixels"],
            "passes": bool(auth_stats["changed_pixels"] == 0),
            "max_abs_diff": auth_stats["max_abs_diff"],
            "tolerance": 1.0e-6,
        }
        sdb_mask = valid_support & np.isclose(support_arr, 3)
        river_zone_mask = valid_support & np.isin(support_arr, [4, 5])
        low_conf_mask = valid_support & np.isclose(support_arr, 6)
        zones["sdb_zone"] = _raster_diff_stats(diff, sdb_mask)
        zones["river_zone"] = _raster_diff_stats(diff, river_zone_mask)
        zones["low_confidence_fill_zone"] = _raster_diff_stats(diff, low_conf_mask)
    river_mask_arr = _align_raster_to_template(river_mask_raster, final_raster, dtype="float64", nodata_value=np.nan, resampling=Resampling.nearest)
    if river_mask_arr is not None:
        zones["river_mask"] = _raster_diff_stats(diff, finite & np.isfinite(river_mask_arr) & (river_mask_arr != 0))
    estuary_mask_arr = _align_raster_to_template(estuary_mask_raster, final_raster, dtype="float64", nodata_value=np.nan, resampling=Resampling.nearest)
    if estuary_mask_arr is not None:
        zones["estuary_mask"] = _raster_diff_stats(diff, finite & np.isfinite(estuary_mask_arr) & (estuary_mask_arr != 0))
    return zones


def _write_zone_diff_csv(path: Path, zone_summary: dict[str, Any]) -> None:
    rows=[]
    for name, stats in zone_summary.items():
        if not isinstance(stats, dict):
            continue
        row={"zone": name}
        row.update(stats)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)

def _metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    if obs.size == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan}
    resid = pred - obs
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"n": int(obs.size), "rmse": rmse, "mae": mae, "bias": bias, "r2": r2}


def _write_markdown_summary(path: Path, *, payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    baseline = overall["baseline"]
    final = overall["final"]
    delta = overall["delta_final_minus_baseline"]
    zone_summary = payload.get("zone_diff_summary", {}) if isinstance(payload.get("zone_diff_summary", {}), dict) else {}
    auth_inv = zone_summary.get("authoritative_locked_invariant", {}) if isinstance(zone_summary.get("authoritative_locked_invariant", {}), dict) else {}
    lines = [
        "# Workflow Benchmark Summary",
        "",
        f"Holdout: `{payload['inputs']['holdout']}`",
        f"Baseline raster: `{payload['inputs']['baseline_raster']}`",
        f"Final raster: `{payload['inputs']['final_raster']}`",
        "",
        "## Overall Holdout Comparison",
        "",
        f"- Points evaluated: {baseline['n']}",
        f"- Baseline RMSE: {baseline['rmse']:.3f} m",
        f"- Final RMSE: {final['rmse']:.3f} m",
        f"- Delta RMSE (final-baseline): {delta['rmse']:.3f} m",
        f"- Baseline MAE: {baseline['mae']:.3f} m",
        f"- Final MAE: {final['mae']:.3f} m",
        f"- Delta MAE (final-baseline): {delta['mae']:.3f} m",
        f"- Baseline bias: {baseline['bias']:.3f} m",
        f"- Final bias: {final['bias']:.3f} m",
        f"- Baseline R²: {baseline['r2']:.3f}",
        f"- Final R²: {final['r2']:.3f}",
        "",
        "## Raster Difference By Zone",
        "",
    ]
    if auth_inv:
        lines.extend([
            f"- Authoritative locked invariant passes: **{auth_inv.get('passes')}**",
            f"- Authoritative locked changed pixels: {auth_inv.get('changed_pixels')}",
            f"- Authoritative locked max abs diff: {auth_inv.get('max_abs_diff')}",
            "",
        ])
    for zone_name in ("authoritative_locked", "river_zone", "sdb_zone", "estuary_mask", "river_mask", "low_confidence_fill_zone"):
        stats = zone_summary.get(zone_name)
        if isinstance(stats, dict) and stats.get("n", 0) > 0:
            lines.extend([
                f"### {zone_name}",
                f"- Pixels: {stats['n']}",
                f"- Changed pixels: {stats['changed_pixels']} ({100.0 * stats['changed_frac']:.2f}%)",
                f"- Mean diff (final-baseline): {stats['mean_diff']:.3f} m",
                f"- Mean abs diff: {stats['mean_abs_diff']:.3f} m",
                f"- P95 abs diff: {stats['p95_abs_diff']:.3f} m",
                f"- Max abs diff: {stats['max_abs_diff']:.3f} m",
                "",
            ])
    lines.extend([
        "## Sampling",
        "",
        f"- Input rows: {payload['counts']['input_rows']}",
        f"- Finite point rows: {payload['counts']['finite_point_rows']}",
        f"- Points sampled on both rasters: {payload['counts']['points_after_sampling']}",
        f"- Points dropped by raster nodata/non-overlap: {payload['counts']['dropped_after_sampling']}",
        "",
        f"Detailed table: `{payload['table_csv']}`",
        f"Zone diff table: `{payload.get('zone_diff_csv', '')}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_existing(*paths: Optional[Path]) -> Optional[Path]:
    for p in paths:
        if p is None:
            continue
        pp = Path(p)
        if pp.exists():
            return pp
    return None




def _resolve_baseline_raster(*, args, cfg, report: dict, out_dir: Path) -> Optional[Path]:
    baseline_override = getattr(args, "benchmark_baseline_raster", None)
    if baseline_override:
        p = Path(str(baseline_override)).resolve()
        return p if p.exists() else None

    final_outputs = report.get("outputs", {}) if isinstance(report.get("outputs"), dict) else {}
    auth_section = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base"), dict) else {}
    auth_outputs = auth_section.get("outputs", {}) if isinstance(auth_section.get("outputs"), dict) else {}
    auth_auto = report.get("authoritative_base_auto", {}) if isinstance(report.get("authoritative_base_auto"), dict) else {}

    candidates = [
        _resolve_report_path(final_outputs.get("baseline_cudem_interpolation_aligned_to_final"), out_dir),
        _resolve_report_path(final_outputs.get("baseline_cudem_interpolation"), out_dir),
        _resolve_report_path(auth_outputs.get("baseline_cudem_interpolation"), out_dir),
    ]
    auto_native = auth_auto.get("baseline_cudem_interpolation")
    if auto_native:
        auto_path = Path(str(auto_native))
        candidates.append(auto_path if auto_path.exists() else None)

    auth_cfg = getattr(cfg, "authoritative_base", None)
    if auth_cfg:
        try:
            sibling = Path(str(auth_cfg)).with_name("cudem_baseline_interpolation.tif")
        except (TypeError, ValueError, OSError):
            sibling = None
        candidates.append(sibling if sibling and sibling.exists() else None)

    candidates.extend([
        out_dir / "comparison_package" / "baseline_cudem_interpolation_aligned_to_final.tif",
        _resolve_report_path(auth_outputs.get("aligned_authoritative_base"), out_dir),
        _resolve_report_path(auth_outputs.get("authoritative_aligned"), out_dir),
        out_dir / "combined" / "authoritative_base_aligned.tif",
    ])
    return _first_existing(*candidates)

def _resolve_report_path(v: Any, out_dir: Path) -> Optional[Path]:
    if not v:
        return None
    p = Path(str(v))
    if not p.is_absolute():
        p = (out_dir / p).resolve()
    return p if p.exists() else None


def run_workflow_benchmark(*, cfg, args, report: dict, logger, final_path: Optional[Path]) -> Optional[dict[str, Any]]:
    auto_holdout = bool(getattr(args, "benchmark_auto_holdout", False))
    explicit_holdout = getattr(args, "benchmark_holdout", None)
    if not explicit_holdout and not auto_holdout:
        return None

    points_epsg_arg = int(getattr(args, "benchmark_points_epsg", 0) or 0)

    out_dir = Path(cfg.out_dir)
    bench_dir = out_dir / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    final_outputs = report.get("outputs", {}) if isinstance(report.get("outputs"), dict) else {}
    auth_outputs = report.get("authoritative_base", {}).get("outputs", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}

    final_override = getattr(args, "benchmark_final_raster", None)

    baseline_raster = _resolve_baseline_raster(args=args, cfg=cfg, report=report, out_dir=out_dir)
    final_raster = _first_existing(
        Path(str(final_override)).resolve() if final_override else None,
        Path(final_path).resolve() if final_path else None,
        _resolve_report_path(final_outputs.get("selected_final_depth"), out_dir),
        out_dir / "combined" / "bathy_combined_depth_conditioned.tif",
    )

    if baseline_raster is None:
        raise FileNotFoundError("Could not resolve benchmark baseline raster")
    if final_raster is None:
        raise FileNotFoundError("Could not resolve benchmark final raster")

    auto_holdout_receipt = None
    if auto_holdout:
        candidate_path = _resolve_auto_holdout_candidate(args=args, report=report, out_dir=out_dir)
        if candidate_path is None:
            raise FileNotFoundError("Could not resolve auto-holdout candidate pool from --benchmark-holdout or authoritative support report")
        holdout_path, points_epsg, inferred_x, inferred_y, inferred_z, auto_holdout_receipt = _build_auto_holdout(
            candidate_path=candidate_path,
            bench_dir=bench_dir,
            baseline_raster=baseline_raster,
            final_raster=final_raster,
            points_epsg_arg=points_epsg_arg if points_epsg_arg > 0 else None,
            x_col=getattr(args, "benchmark_x_col", None),
            y_col=getattr(args, "benchmark_y_col", None),
            z_col=getattr(args, "benchmark_z_col", None),
            holdout_frac=float(getattr(args, "benchmark_holdout_frac", 0.2) or 0.2),
            holdout_min_points=int(getattr(args, "benchmark_holdout_min_points", 2000) or 2000),
            holdout_seed=int(getattr(args, "benchmark_holdout_seed", 42) or 42),
            logger=logger,
        )
    else:
        holdout_path = Path(str(explicit_holdout)).resolve()
        if not holdout_path.exists():
            raise FileNotFoundError(f"Benchmark holdout not found: {holdout_path}")
        points_epsg = points_epsg_arg
        inferred_x = inferred_y = inferred_z = None

    support_class_raster = _first_existing(
        _resolve_report_path(final_outputs.get("support_class"), out_dir),
        out_dir / "combined" / "support_class.tif",
    )
    provenance_raster = _first_existing(
        _resolve_report_path(final_outputs.get("selected_final_provenance"), out_dir),
        out_dir / "combined" / "bathy_combined_depth_conditioned_provenance.tif",
    )
    river_mask_raster = _first_existing(
        _resolve_report_path(river_outputs.get("river_channel_mask"), out_dir),
        out_dir / "derived_cache" / str(getattr(cfg, "run_id", "")) / "river" / "work" / "river_channel_mask.tif",
    )
    estuary_mask_raster = _first_existing(
        _resolve_report_path(river_outputs.get("estuary_clip_mask"), out_dir),
        out_dir / "derived_cache" / str(getattr(cfg, "run_id", "")) / "river" / "work" / "estuary_clip_mask.tif",
    )

    df = _read_table(holdout_path)
    x_name = inferred_x or getattr(args, "benchmark_x_col", None) or _infer_col(df, ["x", "X", "easting", "lon", "longitude"], "x")
    y_name = inferred_y or getattr(args, "benchmark_y_col", None) or _infer_col(df, ["y", "Y", "northing", "lat", "latitude"], "y")
    z_name = inferred_z or getattr(args, "benchmark_z_col", None) or _infer_col(df, ["z", "Z", "depth_m", "z_m", "depth"], "z/depth")

    input_rows = int(len(df))
    x = pd.to_numeric(df[x_name], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(df[y_name], errors="coerce").to_numpy(dtype="float64")
    obs = pd.to_numeric(df[z_name], errors="coerce").to_numpy(dtype="float64")

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(obs)
    finite_point_rows = int(np.count_nonzero(valid))
    x = x[valid]
    y = y[valid]
    obs = obs[valid]

    points_epsg = points_epsg if auto_holdout else (points_epsg_arg if points_epsg_arg > 0 else _infer_points_epsg(
        x=x, y=y, x_name=x_name, y_name=y_name, baseline_raster=baseline_raster, final_raster=final_raster
    ))
    logger.info(f"[BENCHMARK] Using point CRS EPSG:{points_epsg} for holdout sampling")

    baseline = _sample_raster(baseline_raster, x, y, src_epsg=points_epsg)
    final = _sample_raster(final_raster, x, y, src_epsg=points_epsg)
    finite = np.isfinite(obs) & np.isfinite(baseline) & np.isfinite(final)
    sampled_rows = int(np.count_nonzero(finite))
    dropped_after_sampling = int(obs.size - sampled_rows)
    x = x[finite]
    y = y[finite]
    obs = obs[finite]
    baseline = baseline[finite]
    final = final[finite]
    if sampled_rows == 0:
        raise ValueError("Benchmark produced zero overlapping finite samples between holdout points and both rasters")

    support_class = _sample_raster(support_class_raster, x, y, src_epsg=points_epsg) if support_class_raster else None
    provenance = _sample_raster(provenance_raster, x, y, src_epsg=points_epsg) if provenance_raster else None
    river_mask = _sample_mask(river_mask_raster, x, y, src_epsg=points_epsg) if river_mask_raster else None
    estuary_mask = _sample_mask(estuary_mask_raster, x, y, src_epsg=points_epsg) if estuary_mask_raster else None

    rows: list[dict[str, Any]] = []
    overall_baseline = _metrics(obs, baseline)
    overall_final = _metrics(obs, final)
    rows.append({"group": "overall", "model": "baseline", **overall_baseline})
    rows.append({"group": "overall", "model": "final", **overall_final})

    def _append_group(name: str, mask: np.ndarray) -> None:
        rows.append({"group": name, "model": "baseline", **_metrics(obs[mask], baseline[mask])})
        rows.append({"group": name, "model": "final", **_metrics(obs[mask], final[mask])})

    if river_mask is not None:
        _append_group("river_mask", river_mask)
    if estuary_mask is not None:
        _append_group("estuary_mask", estuary_mask)
    if support_class is not None:
        for cls in sorted(np.unique(support_class[np.isfinite(support_class)])):
            _append_group(f"support_class_{int(cls)}", np.isclose(support_class, cls))
    if provenance is not None:
        for cls in sorted(np.unique(provenance[np.isfinite(provenance)])):
            _append_group(f"provenance_{int(cls)}", np.isclose(provenance, cls))

    table_csv = bench_dir / "benchmark_table.csv"
    summary_json = bench_dir / "benchmark_summary.json"
    zone_diff_csv = bench_dir / "benchmark_zone_diff_table.csv"
    pd.DataFrame(rows).to_csv(table_csv, index=False)

    zone_diff_summary = _compute_zone_diff_summary(
        baseline_raster=baseline_raster,
        final_raster=final_raster,
        support_class_raster=support_class_raster,
        river_mask_raster=river_mask_raster,
        estuary_mask_raster=estuary_mask_raster,
    )
    _write_zone_diff_csv(zone_diff_csv, zone_diff_summary)

    markdown_path = bench_dir / "benchmark_summary.md"
    payload = {
        "inputs": {
            "holdout": str(holdout_path),
            "auto_holdout": bool(auto_holdout),
            "baseline_raster": str(baseline_raster),
            "final_raster": str(final_raster),
            "support_class_raster": str(support_class_raster) if support_class_raster else None,
            "provenance_raster": str(provenance_raster) if provenance_raster else None,
            "river_mask_raster": str(river_mask_raster) if river_mask_raster else None,
            "estuary_mask_raster": str(estuary_mask_raster) if estuary_mask_raster else None,
            "points_epsg": points_epsg,
            "x_col": x_name,
            "y_col": y_name,
            "z_col": z_name,
        },
        "counts": {
            "input_rows": input_rows,
            "finite_point_rows": finite_point_rows,
            "points_after_sampling": int(obs.size),
            "dropped_after_sampling": dropped_after_sampling,
        },
        "overall": {
            "baseline": overall_baseline,
            "final": overall_final,
            "delta_final_minus_baseline": {
                "rmse": float(overall_final["rmse"] - overall_baseline["rmse"]),
                "mae": float(overall_final["mae"] - overall_baseline["mae"]),
                "bias": float(overall_final["bias"] - overall_baseline["bias"]),
                "r2": float(overall_final["r2"] - overall_baseline["r2"]) if np.isfinite(overall_final["r2"]) and np.isfinite(overall_baseline["r2"]) else np.nan,
            },
        },
        "table_csv": str(table_csv),
        "zone_diff_csv": str(zone_diff_csv),
        "zone_diff_summary": zone_diff_summary,
        "markdown_summary": str(markdown_path),
        "auto_holdout_receipt": auto_holdout_receipt,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown_summary(markdown_path, payload=payload)
    auth_inv = zone_diff_summary.get("authoritative_locked_invariant", {}) if isinstance(zone_diff_summary, dict) else {}
    if isinstance(auth_inv, dict) and auth_inv.get("n", 0) > 0 and not bool(auth_inv.get("passes", False)):
        raise RuntimeError(
            f"Benchmark invariant failed: final raster differs from baseline on authoritative_locked cells "
            f"(changed_pixels={auth_inv.get('changed_pixels')}, max_abs_diff={auth_inv.get('max_abs_diff')})"
        )
    report.setdefault("benchmark", {})["workflow_benchmark"] = payload
    report.setdefault("outputs", {})["benchmark_summary_json"] = str(summary_json)
    report.setdefault("outputs", {})["benchmark_table_csv"] = str(table_csv)
    report.setdefault("outputs", {})["benchmark_summary_md"] = str(markdown_path)
    report.setdefault("outputs", {})["benchmark_zone_diff_csv"] = str(zone_diff_csv)
    if auto_holdout_receipt is not None:
        report.setdefault("outputs", {})["benchmark_auto_holdout_receipt_json"] = str(bench_dir / "auto_holdout_receipt.json")
        report.setdefault("outputs", {})["benchmark_auto_holdout_points"] = str(bench_dir / "auto_holdout_points.csv")
    logger.info("[BENCHMARK] Workflow benchmark written: %s ; %s ; %s", summary_json, table_csv, markdown_path)
    return payload
