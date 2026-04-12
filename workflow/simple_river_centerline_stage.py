from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from simple_river_stage_contract import STAGE_RIVER_CENTERLINE
from simple_river_stage_receipts import build_stage_receipt, write_stage_receipt

_REQUIRED_FIELDS = ("point_id", "station_m", "geometry")
_OPTIONAL_ALIASES = {
    "reach_id": ("reach_id", "comid", "nhdplusid"),
    "levelpath_id": ("levelpath_id", "levelpathi", "level_path_id"),
    "centerline_order": ("centerline_order", "order", "rank"),
}


def centerline_field_schema() -> dict[str, str]:
    return {
        "point_id": "string",
        "station_m": "float64",
        "geometry": "Point",
        "reach_id": "string",
        "levelpath_id": "string",
        "centerline_order": "int64",
        "source_reach_key": "string",
        "is_endpoint": "bool",
    }


def _normalize_point_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if "point_id" not in out.columns:
        out["point_id"] = [f"cl_{i:07d}" for i in range(len(out))]
    else:
        values = out["point_id"].astype(str).str.strip()
        mask = values.eq("") | values.eq("None") | values.eq("nan")
        if mask.any():
            repl = values.copy()
            idxs = list(out.index[mask])
            for pos, idx in enumerate(idxs):
                repl.loc[idx] = f"cl_{pos:07d}"
            values = repl
        out["point_id"] = values
    return out


def _canonicalize_centerline_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None:
        return gpd.GeoDataFrame(columns=list(centerline_field_schema().keys()), geometry="geometry")
    out = gdf.copy()
    if getattr(out, 'geometry', None) is None:
        out = gpd.GeoDataFrame(out, geometry='geometry', crs=getattr(gdf, 'crs', None))
    for target, aliases in _OPTIONAL_ALIASES.items():
        if target not in out.columns:
            for alias in aliases:
                if alias in out.columns:
                    out[target] = out[alias]
                    break
    if "station_m" not in out.columns:
        for cand in ("centerline_m", "chainage_m", "distance_m"):
            if cand in out.columns:
                out["station_m"] = pd.to_numeric(out[cand], errors="coerce")
                break
    if "station_m" in out.columns:
        out["station_m"] = pd.to_numeric(out["station_m"], errors="coerce")
    out = _normalize_point_ids(out)
    if "source_reach_key" not in out.columns:
        if "levelpath_id" in out.columns:
            out["source_reach_key"] = out["levelpath_id"].astype(str)
        elif "reach_id" in out.columns:
            out["source_reach_key"] = out["reach_id"].astype(str)
        else:
            out["source_reach_key"] = "unknown"
    if "centerline_order" not in out.columns:
        out["centerline_order"] = range(len(out))
    _centerline_order = pd.to_numeric(out["centerline_order"], errors="coerce")
    if _centerline_order.isna().any():
        _centerline_order = _centerline_order.where(~_centerline_order.isna(), pd.Series(range(len(out)), index=out.index, dtype="float64"))
    out["centerline_order"] = _centerline_order.astype(int)
    if "is_endpoint" not in out.columns:
        out["is_endpoint"] = False
    sort_cols = [c for c in ("levelpath_id", "reach_id", "station_m", "centerline_order") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    keep_cols = [c for c in centerline_field_schema().keys() if c in out.columns]
    extra_cols = [c for c in out.columns if c not in keep_cols]
    out = out[keep_cols + extra_cols]
    return gpd.GeoDataFrame(out, geometry='geometry', crs=out.crs)


def _derive_centerline_points_from_existing_workflow(*, river_context: dict[str, Any]) -> gpd.GeoDataFrame:
    centerline_points = river_context.get('centerline_points_gdf')
    if centerline_points is not None:
        return _canonicalize_centerline_points(centerline_points)
    existing_path = river_context.get('existing_centerline_path')
    if existing_path:
        path = Path(existing_path)
        if path.exists():
            return _canonicalize_centerline_points(gpd.read_file(path))
    raise RuntimeError('simple_river_centerline_stage_missing_centerline_source')


def validate_centerline_points(gdf) -> dict[str, Any]:
    missing = [f for f in _REQUIRED_FIELDS if f not in gdf.columns]
    record_count = int(len(gdf)) if gdf is not None else 0
    null_station_count = int(pd.to_numeric(gdf['station_m'], errors='coerce').isna().sum()) if 'station_m' in getattr(gdf, 'columns', []) else record_count
    null_geometry_count = int(gdf.geometry.isna().sum()) if gdf is not None and getattr(gdf, 'geometry', None) is not None else record_count
    geometry_valid = bool(getattr(gdf, 'geometry', None) is not None and gdf.geometry.notna().all() and gdf.is_valid.all()) if record_count > 0 else True
    point_id_unique = bool(gdf['point_id'].is_unique) if 'point_id' in getattr(gdf, 'columns', []) else False
    station_monotonic_by_reach = True
    if record_count > 0 and 'station_m' in gdf.columns:
        group_cols = [c for c in ('levelpath_id', 'reach_id') if c in gdf.columns]
        if group_cols:
            for _, grp in gdf.groupby(group_cols, dropna=False, sort=False):
                vals = pd.to_numeric(grp['station_m'], errors='coerce')
                if vals.isna().any() or not vals.is_monotonic_increasing:
                    station_monotonic_by_reach = False
                    break
        else:
            vals = pd.to_numeric(gdf['station_m'], errors='coerce')
            station_monotonic_by_reach = bool((not vals.isna().any()) and vals.is_monotonic_increasing)
    return {
        'valid': not missing and geometry_valid and point_id_unique and null_station_count == 0 and null_geometry_count == 0 and station_monotonic_by_reach,
        'record_count': record_count,
        'required_fields_present': not missing,
        'missing_required_fields': missing,
        'geometry_valid': geometry_valid,
        'nonempty_geometry': null_geometry_count == 0,
        'point_id_unique': point_id_unique,
        'station_monotonic_by_reach': bool(station_monotonic_by_reach),
        'duplicate_point_ids': int(0 if 'point_id' not in getattr(gdf, 'columns', []) else gdf['point_id'].duplicated().sum()),
        'null_station_count': null_station_count,
        'null_geometry_count': null_geometry_count,
    }


def write_centerline_points_gpkg(gdf, out_path: str) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver='GPKG')
    return str(path)


def build_simple_river_centerline_points(*, river_context: dict[str, Any], out_path: str, receipt_path: str | None = None) -> dict[str, Any]:
    gdf = _derive_centerline_points_from_existing_workflow(river_context=river_context)
    validation = validate_centerline_points(gdf)
    if not validation.get('valid'):
        raise RuntimeError(f"simple_river_centerline_stage_invalid:{validation}")
    written_path = write_centerline_points_gpkg(gdf, out_path)
    receipt = build_stage_receipt(
        stage_id=STAGE_RIVER_CENTERLINE,
        output_artifact=written_path,
        input_artifacts=[str(v) for v in river_context.get('input_artifacts', [])],
        record_count=int(len(gdf)),
        field_schema=centerline_field_schema(),
        vertical_reference=str(river_context.get("vertical_reference") or "unknown"),
        warnings=[],
        source_logic='canonicalized_existing_centerline_points',
        validation=validation,
    )
    written_receipt = None
    if receipt_path:
        written_receipt = write_stage_receipt(receipt, receipt_path)
    return {
        'stage_id': STAGE_RIVER_CENTERLINE,
        'output_artifact': written_path,
        'receipt_path': written_receipt,
        'record_count': int(len(gdf)),
        'validation': validation,
    }
