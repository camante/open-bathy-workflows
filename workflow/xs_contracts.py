from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import pandas as pd
import pyogrio
import logging
log = logging.getLogger(__name__)



TRUTHY_STRINGS = {"1", "true", "t", "yes", "y"}
FALSY_STRINGS = {"0", "false", "f", "no", "n", "", "nan", "none", "null"}


def safe_float(value) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        log.debug("safe_float: suppressed exception", exc_info=True)
        return float("nan")


def coerce_truthy_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    vals = series.astype(str).str.strip().str.lower()
    out = vals.isin(TRUTHY_STRINGS)
    invalid = ~(vals.isin(TRUTHY_STRINGS) | vals.isin(FALSY_STRINGS))
    if bool(invalid.any()):
        sample = sorted(set(vals[invalid].tolist()))[:5]
        raise ValueError(f"Ambiguous truthy values encountered: {sample}")
    return out


def gpkg_layers(path: Path | str) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {path}")
    return list(pyogrio.list_layers(path)[:, 0])


def require_gpkg_layers(path: Path | str, required_layers: Sequence[str]) -> list[str]:
    layers = gpkg_layers(path)
    missing = [lyr for lyr in required_layers if lyr not in layers]
    if missing:
        raise ValueError(
            f"GeoPackage {path} missing required layers {missing}; available={layers}"
        )
    return layers


def read_gpkg_layer_strict(path: Path | str, layer: str) -> gpd.GeoDataFrame:
    require_gpkg_layers(path, [layer])
    gdf = gpd.read_file(path, layer=layer)
    if gdf.empty:
        raise ValueError(f"GeoPackage layer is empty: {path}:{layer}")
    return gdf


def require_columns(df: pd.DataFrame, required_columns: Iterable[str], *, label: str) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns {missing}; available={list(df.columns)}")


def _canonicalize_xs_points_columns(df: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, str]]:
    out = df.copy()
    alias_map: dict[str, str] = {}
    for canonical, aliases in XS_POINTS_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                out[canonical] = out[alias]
                alias_map[canonical] = alias
                break
    return out, alias_map


XS_LINES_REQUIRED_COLUMNS = [
    "xs_id",
    "bank_left_dist_m",
    "bank_right_dist_m",
    "bank_left_z_m",
    "bank_right_z_m",
    "s_center_m",
]

XS_POINTS_REQUIRED_COLUMNS = [
    "xs_id",
    "s_m",
    "z_m",
    "is_bank_left",
    "is_bank_right",
]

XS_POINTS_ALIASES = {
    "s_m": ["dist_m"],
    "z_m": ["z_dem"],
}

SOUNDINGS_SUBSET_REQUIRED_COLUMNS = [
    "x",
    "y",
    "z",
    "depth_m",
    "z_m",
    "_src_file",
    "crs",
]


def validate_xs_artifacts(xs_gpkg: Path | str) -> dict:
    _, _, info = load_validated_xs_artifacts(xs_gpkg, xs_lines_layer="xs_lines", xs_points_layer="xs_points")
    return info


def load_validated_xs_artifacts(xs_gpkg: Path | str, *, xs_lines_layer: str = "xs_lines", xs_points_layer: str = "xs_points"):
    xs_gpkg = Path(xs_gpkg)
    layers = require_gpkg_layers(xs_gpkg, [xs_lines_layer, xs_points_layer])
    xs_lines = read_gpkg_layer_strict(xs_gpkg, xs_lines_layer)
    xs_points = read_gpkg_layer_strict(xs_gpkg, xs_points_layer)
    xs_points, xs_points_alias_map = _canonicalize_xs_points_columns(xs_points)
    require_columns(xs_lines, XS_LINES_REQUIRED_COLUMNS, label=xs_lines_layer)
    require_columns(xs_points, XS_POINTS_REQUIRED_COLUMNS, label=xs_points_layer)
    if xs_lines.geometry.is_empty.any() or xs_lines.geometry.isna().any():
        raise ValueError(f"{xs_lines_layer} contains empty/null geometry: {xs_gpkg}")
    if xs_points.geometry.is_empty.any() or xs_points.geometry.isna().any():
        raise ValueError(f"{xs_points_layer} contains empty/null geometry: {xs_gpkg}")
    return xs_lines, xs_points, {
        "path": str(xs_gpkg),
        "layers": layers,
        "xs_lines_layer": xs_lines_layer,
        "xs_points_layer": xs_points_layer,
        "xs_lines_n": int(len(xs_lines)),
        "xs_points_n": int(len(xs_points)),
        "xs_points_alias_map": xs_points_alias_map,
    }


def summarize_bank_contract(xs_lines: pd.DataFrame, xs_points: pd.DataFrame) -> dict:
    total = int(len(xs_lines))
    left_dist = pd.to_numeric(xs_lines.get("bank_left_dist_m"), errors="coerce")
    right_dist = pd.to_numeric(xs_lines.get("bank_right_dist_m"), errors="coerce")
    left_z = pd.to_numeric(xs_lines.get("bank_left_z_m"), errors="coerce")
    right_z = pd.to_numeric(xs_lines.get("bank_right_z_m"), errors="coerce")
    left_ok = left_dist.notna() & left_z.notna()
    right_ok = right_dist.notna() & right_z.notna()
    both_ok = left_ok & right_ok
    left_flags = int(coerce_truthy_series(xs_points["is_bank_left"]).sum()) if "is_bank_left" in xs_points.columns else 0
    right_flags = int(coerce_truthy_series(xs_points["is_bank_right"]).sum()) if "is_bank_right" in xs_points.columns else 0
    return {
        "xs_total": total,
        "xs_with_left_bank_contract": int(left_ok.sum()),
        "xs_with_right_bank_contract": int(right_ok.sum()),
        "xs_with_both_bank_contract": int(both_ok.sum()),
        "xs_missing_left_bank_contract": int((~left_ok).sum()),
        "xs_missing_right_bank_contract": int((~right_ok).sum()),
        "xs_points_left_bank_flags": left_flags,
        "xs_points_right_bank_flags": right_flags,
    }


def validate_soundings_subset_parquet(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Soundings subset parquet not found: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Soundings subset must be parquet: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Soundings subset parquet is empty: {path}")
    require_columns(df, SOUNDINGS_SUBSET_REQUIRED_COLUMNS, label="soundings_subset")
    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")
    z = pd.to_numeric(df["z"], errors="coerce")
    depth_m = pd.to_numeric(df["depth_m"], errors="coerce")
    z_m = pd.to_numeric(df["z_m"], errors="coerce")
    finite_xyz = x.notna() & y.notna() & z.notna()
    if not bool(finite_xyz.all()):
        bad = int((~finite_xyz).sum())
        raise ValueError(f"Soundings subset has non-finite canonical x/y/z rows: {bad} in {path}")
    if not bool(depth_m.notna().all()):
        bad = int(depth_m.isna().sum())
        raise ValueError(f"Soundings subset has non-finite depth_m rows: {bad} in {path}")
    if not bool(z_m.notna().all()):
        bad = int(z_m.isna().sum())
        raise ValueError(f"Soundings subset has non-finite z_m rows: {bad} in {path}")
    crs_values = {str(v).strip() for v in df["crs"].astype(str).tolist() if str(v).strip()}
    if len(crs_values) != 1:
        raise ValueError(f"Soundings subset must carry exactly one non-empty CRS value; found {sorted(crs_values)} in {path}")
    return {
        "path": str(path),
        "rows": int(len(df)),
        "depth_col": "depth_m",
        "depth_min": float(depth_m.min()),
        "depth_max": float(depth_m.max()),
        "crs": next(iter(crs_values)),
    }
