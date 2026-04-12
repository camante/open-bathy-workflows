from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from river_v2_context import RiverV2Context
from river_v2_contract import RiverV2StageResult, STAGE_CENTERLINE_BED_BACKBONE_DENSE
from river_v2_receipts import build_river_v2_stage_receipt, write_river_v2_receipt

_GROUP_COLS = ("component_id", "levelpath_id", "reach_id", "source_reach_key")
_REQUIRED_FIELDS = ("station_m", "bed_backbone_z_m", "geometry")


def _read_backbone_points(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise RuntimeError(f"river_v2_backbone_dense_missing_input:{path}")
    gdf = gpd.read_file(path)
    if len(gdf) == 0:
        raise RuntimeError("river_v2_backbone_dense_empty_input")
    gdf["station_m"] = pd.to_numeric(gdf.get("station_m"), errors="coerce")
    gdf["bed_backbone_z_m"] = pd.to_numeric(gdf.get("bed_backbone_z_m"), errors="coerce")
    return gdf




def _read_centerline_points(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise RuntimeError(f"river_v2_backbone_dense_missing_centerline:{path}")
    gdf = gpd.read_file(path)
    if len(gdf) == 0:
        raise RuntimeError("river_v2_backbone_dense_empty_centerline")
    gdf["station_m"] = pd.to_numeric(gdf.get("station_m"), errors="coerce")
    return gdf

def _group_cols(gdf: gpd.GeoDataFrame) -> list[str]:
    cols = [c for c in _GROUP_COLS if c in gdf.columns]
    return cols if cols else []


def _densify_group(backbone_grp: gpd.GeoDataFrame, centerline_grp: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    anchors = backbone_grp.sort_values([c for c in ("station_m", "point_id") if c in backbone_grp.columns], kind="mergesort").reset_index(drop=True)
    centerline = centerline_grp.sort_values([c for c in ("station_m", "point_id") if c in centerline_grp.columns], kind="mergesort").reset_index(drop=True)
    anchor_vals = pd.to_numeric(anchors["bed_backbone_z_m"], errors="coerce").to_numpy(dtype=float)
    anchor_station = pd.to_numeric(anchors["station_m"], errors="coerce").to_numpy(dtype=float)
    finite_anchor = np.isfinite(anchor_vals) & np.isfinite(anchor_station)
    if np.count_nonzero(finite_anchor) == 0:
        return []
    anchor_station = anchor_station[finite_anchor]
    anchor_vals = anchor_vals[finite_anchor]
    if anchor_station.size == 0:
        return []
    anchor_order = np.argsort(anchor_station, kind="mergesort")
    anchor_station = anchor_station[anchor_order]
    anchor_vals = anchor_vals[anchor_order]

    center_station = pd.to_numeric(centerline["station_m"], errors="coerce").to_numpy(dtype=float)
    finite_center = np.isfinite(center_station)
    if np.count_nonzero(finite_center) == 0:
        return []
    support_mask = finite_center & (center_station >= anchor_station[0] - 1.0e-9) & (center_station <= anchor_station[-1] + 1.0e-9)
    if np.count_nonzero(support_mask) == 0:
        return []

    interp_vals = np.interp(center_station[support_mask], anchor_station, anchor_vals)
    group_vals = {c: centerline.iloc[0][c] for c in _GROUP_COLS if c in centerline.columns}
    anchor_station_set = {float(s) for s in anchor_station.tolist()}
    rows: list[dict[str, Any]] = []
    support_idx = np.where(support_mask)[0]
    for out_i, center_i in enumerate(support_idx):
        row = centerline.iloc[int(center_i)]
        s = float(center_station[int(center_i)])
        rows.append({
            **group_vals,
            "point_id": str(row.get("point_id", f"dense_{center_i}")),
            "station_m": s,
            "bed_backbone_z_m": float(interp_vals[out_i]),
            "dense_source": "anchor" if s in anchor_station_set else "linear_interp",
            "geometry": row.geometry,
        })
    return rows


def build_dense_backbone(backbone_points_path: Path, centerline_points_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, Any], list[str]]:
    backbone_gdf = _read_backbone_points(backbone_points_path)
    centerline_gdf = _read_centerline_points(centerline_points_path)
    groups = _group_cols(centerline_gdf)
    rows: list[dict[str, Any]] = []
    if groups:
        center_groups = {tuple(k if isinstance(k, tuple) else (k,)): grp for k, grp in centerline_gdf.groupby(groups, dropna=False, sort=False)}
        backbone_iter = backbone_gdf.groupby(groups, dropna=False, sort=False)
    else:
        center_groups = {tuple(): centerline_gdf}
        backbone_iter = [(tuple(), backbone_gdf)]
    for key, backbone_grp in backbone_iter:
        key_t = tuple(key) if isinstance(key, tuple) else tuple([key]) if key is not None else tuple()
        center_grp = center_groups.get(key_t)
        if center_grp is None or len(center_grp) == 0:
            continue
        rows.extend(_densify_group(backbone_grp, center_grp))
    if not rows:
        raise RuntimeError("river_v2_backbone_dense_no_output_rows")
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=centerline_gdf.crs)
    out = out.sort_values([c for c in (*groups, "station_m", "point_id") if c in out.columns], kind="mergesort").reset_index(drop=True)
    diagnostics = {
        "record_count": int(len(out)),
        "anchor_count": int(np.count_nonzero(out["dense_source"].astype(str) == "anchor")),
        "interp_count": int(np.count_nonzero(out["dense_source"].astype(str) == "linear_interp")),
        "finite_backbone_count": int(np.count_nonzero(np.isfinite(pd.to_numeric(out["bed_backbone_z_m"], errors="coerce").to_numpy(dtype=float)))),
    }
    persist_cols = [c for c in ("point_id", "station_m", *_GROUP_COLS, "bed_backbone_z_m", "geometry") if c in out.columns]
    persisted = gpd.GeoDataFrame(out[persist_cols].copy(), geometry="geometry", crs=out.crs)
    return persisted, diagnostics, []


def validate_dense_backbone(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    missing = [f for f in _REQUIRED_FIELDS if f not in gdf.columns]
    vals = pd.to_numeric(gdf.get("bed_backbone_z_m"), errors="coerce").to_numpy(dtype=float) if "bed_backbone_z_m" in gdf.columns else np.array([], dtype=float)
    return {
        "valid": not missing and len(gdf) > 0 and int(np.count_nonzero(np.isfinite(vals))) == len(gdf),
        "record_count": int(len(gdf)),
        "missing_required_fields": missing,
        "finite_backbone_count": int(np.count_nonzero(np.isfinite(vals))),
    }


def run_backbone_dense_stage(ctx: RiverV2Context, *, backbone_points_path: Path, centerline_points_path: Path) -> RiverV2StageResult:
    dense_gdf, diagnostics, warnings = build_dense_backbone(Path(backbone_points_path), Path(centerline_points_path))
    out_path = ctx.paths.river_centerline_bed_backbone_dense_points
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dense_gdf.to_file(out_path, driver="GPKG")
    validation = validate_dense_backbone(dense_gdf)
    if not validation.get("valid"):
        raise RuntimeError(f"river_v2_backbone_dense_invalid:{validation}")
    receipt = build_river_v2_stage_receipt(
        stage_id=STAGE_CENTERLINE_BED_BACKBONE_DENSE,
        output_artifact=str(out_path),
        input_artifacts=ctx.direct_stage_input_artifacts(backbone_points_path, centerline_points_path),
        record_count=int(len(dense_gdf)),
        field_schema={
            "point_id": "string",
            "station_m": "float64",
            "bed_backbone_z_m": "float64",
            "geometry": "geometry",
        },
        vertical_reference=ctx.vertical_reference,
        warnings=warnings,
        source_logic="densify sparse backbone anchors along station by linear interpolation at centerline spacing",
        validation={**validation, **diagnostics},
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.river_centerline_bed_backbone_dense_points_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_CENTERLINE_BED_BACKBONE_DENSE,
        output_artifact=out_path,
        receipt_path=receipt_path,
        record_count=int(len(dense_gdf)),
        validation={**validation, **diagnostics},
        warnings=warnings,
    )
