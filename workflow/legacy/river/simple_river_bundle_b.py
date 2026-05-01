from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from legacy.river.simple_river_authoritative_bed_stage import build_centerline_authoritative_bed_points
from legacy.river.simple_river_centerline_stage import build_simple_river_centerline_points
from legacy.river.simple_river_observed_offset_stage import build_centerline_observed_offset_points
from legacy.river.simple_river_stage_contract import (
    STAGE_CENTERLINE_AUTHORITATIVE_BED,
    STAGE_CENTERLINE_OBSERVED_OFFSET,
    STAGE_CENTERLINE_WSE_PROXY,
    STAGE_RIVER_CENTERLINE,
    mark_stage_failed,
    mark_stage_implemented,
)
from legacy.river.simple_river_wse_stage import build_centerline_wse_proxy_points


_STAGE_RESULT_KEY = {
    STAGE_RIVER_CENTERLINE: "centerline",
    STAGE_CENTERLINE_WSE_PROXY: "wse_proxy",
    STAGE_CENTERLINE_AUTHORITATIVE_BED: "authoritative_bed",
    STAGE_CENTERLINE_OBSERVED_OFFSET: "observed_offset",
}

_ALLOWED_UPTO_STAGES = tuple(_STAGE_RESULT_KEY.keys())


def _extend_inputs(river_context: dict[str, Any], *paths: str | None, drop_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    ctx = dict(river_context)
    for key in drop_keys:
        ctx.pop(key, None)
    existing = list(ctx.get("input_artifacts") or [])
    for path in paths:
        if path:
            existing.append(str(path))
    seen: set[str] = set()
    deduped: list[str] = []
    for item in existing:
        item_str = str(item)
        if item_str in seen:
            continue
        seen.add(item_str)
        deduped.append(item_str)
    ctx["input_artifacts"] = deduped
    return ctx


def _mark_and_store(stage_outputs: Dict[str, dict[str, Any]], stage_status: dict[str, dict], result: dict[str, Any], *, stage_id: str) -> dict[str, dict]:
    key = _STAGE_RESULT_KEY[stage_id]
    stage_outputs[key] = dict(result)
    return mark_stage_implemented(
        stage_status,
        stage_id=stage_id,
        output_artifact=result["output_artifact"],
        record_count=result.get("record_count"),
        receipt_path=result.get("receipt_path"),
        warnings=result.get("warnings", []),
    )


def _bundle_summary(*, stage_id: str, stage_outputs: Dict[str, dict[str, Any]], stage_status: dict[str, dict], error: str | None = None) -> dict[str, Any]:
    latest = stage_outputs.get(_STAGE_RESULT_KEY[stage_id], {})
    out = {
        "stage_id": stage_id,
        "centerline": stage_outputs.get("centerline"),
        "wse_proxy": stage_outputs.get("wse_proxy"),
        "authoritative_bed": stage_outputs.get("authoritative_bed"),
        "observed_offset": stage_outputs.get("observed_offset"),
        "output_artifact": latest.get("output_artifact"),
        "receipt_path": latest.get("receipt_path"),
        "record_count": latest.get("record_count"),
        "validation": latest.get("validation", {}),
        "warnings": latest.get("warnings", []),
        "simple_river_stage_status": stage_status,
        "simple_river_stage_outputs": stage_outputs,
    }
    if error is not None:
        out["error"] = str(error)
        out["failed_stage"] = stage_id
    return out


def run_simple_river_bundle_b(*, river_context: dict[str, Any], out_dir: str, stage_status: dict[str, dict], upto_stage: str = STAGE_CENTERLINE_OBSERVED_OFFSET) -> dict[str, Any]:
    if upto_stage not in _ALLOWED_UPTO_STAGES:
        raise ValueError(f"unsupported simple river bundle B upto_stage: {upto_stage}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    updated_status = {str(k): dict(v) for k, v in (stage_status or {}).items()}
    stage_outputs: Dict[str, dict[str, Any]] = {}

    try:
        centerline_path = out_root / "river_centerline_points.gpkg"
        centerline_receipt_path = out_root / "river_centerline_points_receipt.json"
        centerline_result = build_simple_river_centerline_points(
            river_context=river_context,
            out_path=str(centerline_path),
            receipt_path=str(centerline_receipt_path),
        )
        updated_status = _mark_and_store(stage_outputs, updated_status, centerline_result, stage_id=STAGE_RIVER_CENTERLINE)
        if upto_stage == STAGE_RIVER_CENTERLINE:
            return _bundle_summary(stage_id=STAGE_RIVER_CENTERLINE, stage_outputs=stage_outputs, stage_status=updated_status)

        canonical_centerline_path = centerline_result.get("output_artifact")
        ctx_wse = _extend_inputs(
            dict(river_context, centerline_points_path=canonical_centerline_path),
            canonical_centerline_path,
            drop_keys=("centerline_points_gdf", "existing_centerline_path"),
        )
        wse_path = out_root / "centerline_wse_proxy_points.gpkg"
        wse_receipt_path = out_root / "centerline_wse_proxy_points_receipt.json"
        wse_result = build_centerline_wse_proxy_points(
            river_context=ctx_wse,
            out_path=str(wse_path),
            receipt_path=str(wse_receipt_path),
        )
        updated_status = _mark_and_store(stage_outputs, updated_status, wse_result, stage_id=STAGE_CENTERLINE_WSE_PROXY)
        if upto_stage == STAGE_CENTERLINE_WSE_PROXY:
            return _bundle_summary(stage_id=STAGE_CENTERLINE_WSE_PROXY, stage_outputs=stage_outputs, stage_status=updated_status)

        ctx_bed = _extend_inputs(
            dict(river_context, centerline_points_path=canonical_centerline_path),
            canonical_centerline_path,
            drop_keys=("centerline_points_gdf", "existing_centerline_path"),
        )
        bed_path = out_root / "centerline_authoritative_bed_points.gpkg"
        bed_receipt_path = out_root / "centerline_authoritative_bed_points_receipt.json"
        bed_result = build_centerline_authoritative_bed_points(
            river_context=ctx_bed,
            out_path=str(bed_path),
            receipt_path=str(bed_receipt_path),
        )
        updated_status = _mark_and_store(stage_outputs, updated_status, bed_result, stage_id=STAGE_CENTERLINE_AUTHORITATIVE_BED)
        if upto_stage == STAGE_CENTERLINE_AUTHORITATIVE_BED:
            return _bundle_summary(stage_id=STAGE_CENTERLINE_AUTHORITATIVE_BED, stage_outputs=stage_outputs, stage_status=updated_status)

        ctx_offset = _extend_inputs(
            {
                "wse_points_path": wse_result.get("output_artifact"),
                "authoritative_bed_points_path": bed_result.get("output_artifact"),
                "input_artifacts": river_context.get("input_artifacts", []),
                "vertical_reference": river_context.get("vertical_reference"),
            },
            wse_result.get("output_artifact"),
            bed_result.get("output_artifact"),
        )
        offset_path = out_root / "centerline_observed_offset_points.gpkg"
        offset_receipt_path = out_root / "centerline_observed_offset_points_receipt.json"
        offset_result = build_centerline_observed_offset_points(
            river_context=ctx_offset,
            out_path=str(offset_path),
            receipt_path=str(offset_receipt_path),
        )
        updated_status = _mark_and_store(stage_outputs, updated_status, offset_result, stage_id=STAGE_CENTERLINE_OBSERVED_OFFSET)
        return _bundle_summary(stage_id=STAGE_CENTERLINE_OBSERVED_OFFSET, stage_outputs=stage_outputs, stage_status=updated_status)
    except Exception as exc:
        if "observed_offset" in stage_outputs:
            failed_stage = STAGE_CENTERLINE_OBSERVED_OFFSET
        elif "authoritative_bed" in stage_outputs:
            failed_stage = STAGE_CENTERLINE_OBSERVED_OFFSET
        elif "wse_proxy" in stage_outputs:
            failed_stage = STAGE_CENTERLINE_AUTHORITATIVE_BED if upto_stage != STAGE_CENTERLINE_WSE_PROXY else STAGE_CENTERLINE_WSE_PROXY
        elif "centerline" in stage_outputs:
            failed_stage = STAGE_CENTERLINE_WSE_PROXY if upto_stage != STAGE_RIVER_CENTERLINE else STAGE_RIVER_CENTERLINE
        else:
            failed_stage = STAGE_RIVER_CENTERLINE
        updated_status = mark_stage_failed(updated_status, stage_id=failed_stage, error=str(exc))
        return _bundle_summary(stage_id=failed_stage, stage_outputs=stage_outputs, stage_status=updated_status, error=str(exc))
