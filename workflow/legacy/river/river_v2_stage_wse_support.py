from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from legacy.river.river_v2_context import RiverV2Context
from legacy.river.river_v2_contract import RiverV2StageResult, STAGE_CENTERLINE_WSE_SUPPORT
from legacy.river.river_v2_receipts import build_river_v2_stage_receipt, write_river_v2_receipt
from legacy.river.river_v2_stage_wse_proxy import build_wse_support


def wse_support_field_schema() -> dict[str, str]:
    return {
        "point_id": "str",
        "station_m": "float64",
        "wse_support_z_m": "float64",
        "support_distance_m": "float64",
        "support_type": "str",
        "has_support": "bool",
    }


def _load_centerline_points(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if "point_id" not in gdf.columns or "station_m" not in gdf.columns:
        raise RuntimeError("river_v2_wse_support_missing_centerline_identity_fields")
    return gdf


def _validate_wse_support(gdf: gpd.GeoDataFrame) -> dict[str, object]:
    required = ("point_id", "station_m", "wse_support_z_m", "support_distance_m", "support_type", "has_support")
    missing = [field for field in required if field not in gdf.columns]
    vals = pd.to_numeric(gdf["wse_support_z_m"], errors="coerce").to_numpy(dtype=float) if "wse_support_z_m" in gdf.columns else []
    finite = int(pd.notna(vals).sum()) if len(vals) else 0
    return {
        "valid": not missing and len(gdf) > 0 and finite == len(gdf),
        "missing_required_fields": missing,
        "finite_support_count": finite,
        "required_finite_support_count": int(len(gdf)),
        "record_count": int(len(gdf)),
    }


def run_wse_support_stage(
    ctx: RiverV2Context,
    *,
    centerline_points_path: Path,
    bank_wse_edge_guidance_path: Path | None = None,
) -> RiverV2StageResult:
    aux_outputs: dict[str, str] = {}
    support_raster_path = Path(bank_wse_edge_guidance_path) if bank_wse_edge_guidance_path is not None else None
    if support_raster_path is None:
        raise RuntimeError("river_v2_wse_support_missing_explicit_bank_guidance_raster")
    aux_outputs["bank_wse_edge_guidance_path"] = str(support_raster_path)
    centerline = _load_centerline_points(centerline_points_path)
    support_gdf, support_path, support_summary = build_wse_support(
        ctx,
        centerline,
        bank_wse_edge_guidance_path=support_raster_path,
    )
    validation = _validate_wse_support(support_gdf)
    if not validation.get("valid"):
        raise RuntimeError(f"river_v2_wse_support_invalid:{validation}")
    receipt = build_river_v2_stage_receipt(
        stage_id=STAGE_CENTERLINE_WSE_SUPPORT,
        output_artifact=str(support_path),
        input_artifacts=ctx.direct_stage_input_artifacts(
            centerline_points_path,
            aux_outputs.get("bank_wse_edge_guidance_path"),
        ),
        record_count=int(len(support_gdf)),
        field_schema=wse_support_field_schema(),
        vertical_reference=ctx.vertical_reference,
        warnings=[str(support_summary.get("warning"))] if support_summary.get("warning") else [],
        source_logic="river_v2_wse_support:explicit_bank_guidance_raster_to_support_points",
        validation={**validation, **support_summary},
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.centerline_wse_support_points_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_CENTERLINE_WSE_SUPPORT,
        output_artifact=support_path,
        receipt_path=receipt_path,
        record_count=int(len(support_gdf)),
        validation={**validation, **support_summary},
        warnings=[str(support_summary.get("warning"))] if support_summary.get("warning") else [],
        aux_outputs=aux_outputs,
    )


__all__ = ["run_wse_support_stage", "wse_support_field_schema"]
