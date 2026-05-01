from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ResolvedRiverV2Inputs:
    network_gpkg: Path | None
    river_dem_path: Path | None
    authoritative_sampling_source_raster_path: Path | None
    authoritative_sampling_raster_path: Path | None
    centerline_spacing_m: float
    min_stream_order: int
    authoritative_bed_support_points_path: Path | None
    channel_mask_path: Path | None
    export_channel_mask_path: Path | None
    authoritative_lock_base_path: Path | None
    solve_network_gpkg: Path | None = None
    solve_channel_mask_path: Path | None = None
    solve_authoritative_sampling_source_raster_path: Path | None = None
    solve_authoritative_lock_base_path: Path | None = None
    wse_support_source_raster_path: Path | None = None
    wse_support_source_contract: str | None = None
    solve_aoi: str | None = None
    trusted_support_mode: str | None = None
    support_policy_source: str | None = None
    vertical_reference: str | None = None


def resolved_inputs_to_dict(inputs: ResolvedRiverV2Inputs) -> Dict[str, Any]:
    payload = asdict(inputs)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def resolved_inputs_from_dict(payload: Dict[str, Any]) -> ResolvedRiverV2Inputs:
    path_fields = {
        'network_gpkg',
        'river_dem_path',
        'authoritative_sampling_source_raster_path',
        'solve_network_gpkg',
        'authoritative_sampling_raster_path',
        'authoritative_bed_support_points_path',
        'channel_mask_path',
        'export_channel_mask_path',
        'authoritative_lock_base_path',
        'solve_channel_mask_path',
        'solve_authoritative_sampling_source_raster_path',
        'solve_authoritative_lock_base_path',
        'wse_support_source_raster_path',
    }
    data: Dict[str, Any] = dict(payload)
    for field in path_fields:
        value = data.get(field)
        data[field] = Path(value) if value else None
    return ResolvedRiverV2Inputs(**data)


def write_resolved_inputs_manifest(inputs: ResolvedRiverV2Inputs, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(resolved_inputs_to_dict(inputs), indent=2, sort_keys=True), encoding='utf-8')
    return out_path


def read_resolved_inputs_manifest(path: Path) -> ResolvedRiverV2Inputs:
    return resolved_inputs_from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
