from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from simple_river_stage_contract import (
    ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
    ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
    simple_river_stage_status_placeholder,
)


@dataclass
class RiverExecutionPlan:
    should_run: bool
    skip_reason: Optional[str]
    requested_method: str
    effective_method: str
    river_dir: Path
    script_dir: Path
    cache_dir: Path
    work_dir: Path
    raw_hydro_cache: Path
    cached_bed_tif: Path
    cached_depth_tif: Path
    manifest_path: Path
    route_mode: str = ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION
    target_contract_mode: str = ROUTE_MODE_SIMPLE_RIVER_PLAN_V1
    simple_stage_status: Optional[Dict[str, Any]] = None
    legacy_transitional_components: Optional[List[str]] = None
    simple_river_stage_outputs: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_run": bool(self.should_run),
            "skip_reason": self.skip_reason,
            "requested_method": self.requested_method,
            "effective_method": self.effective_method,
            "river_dir": str(self.river_dir),
            "script_dir": str(self.script_dir),
            "cache_dir": str(self.cache_dir),
            "work_dir": str(self.work_dir),
            "raw_hydro_cache": str(self.raw_hydro_cache),
            "cached_bed_tif": str(self.cached_bed_tif),
            "cached_depth_tif": str(self.cached_depth_tif),
            "manifest_path": str(self.manifest_path),
            "route_mode": self.route_mode,
            "target_contract_mode": self.target_contract_mode,
            "simple_stage_status": self.simple_stage_status or {},
            "legacy_transitional_components": list(self.legacy_transitional_components or []),
            "simple_river_stage_outputs": dict(self.simple_river_stage_outputs or {}),
        }


def determine_river_execution_plan(*, cfg: Any, report: Dict[str, Any], ensure_dir_fn, normalize_channel_template_setting_fn, logger) -> RiverExecutionPlan:
    river_report = report.setdefault("river", {})
    river_report.setdefault("steps", {})
    river_notes = river_report.setdefault("notes", {})

    river_dir = ensure_dir_fn(Path(cfg.out_dir) / "river")
    script_dir = Path(__file__).parent
    normalize_channel_template_setting_fn(cfg, report)

    cache_dir = Path(cfg.derived_cache_root) / "river"
    ensure_dir_fn(cache_dir)
    work_dir = ensure_dir_fn(cache_dir / "work")
    raw_hydro_cache = ensure_dir_fn(Path(cfg.cache_root) / "hydrography")
    cached_bed_tif = cache_dir / "river_bed_elev_cached.tif"
    cached_depth_tif = cache_dir / "river_depth_cached.tif"
    manifest_path = cache_dir / "_manifest.json"

    requested_method = str(cfg.river_method).lower().strip()
    effective_method = requested_method
    if cfg.river_channel_template_enabled and effective_method == "skeleton":
        logger.info(
            "[RIVER] Channel template requested with method=skeleton; switching river execution to hybrid so the XS template path runs on the mainstem."
        )
        river_notes["channel_template_method_override"] = {
            "requested_method": "skeleton",
            "effective_method": "hybrid",
            "reason": "channel template is implemented on the XS path",
        }
        effective_method = "hybrid"

    payload = {
        "run_id": cfg.run_id,
        "aoi_tile": cfg.aoi_tile,
        "aoi_data": cfg.aoi,
        "start": cfg.start_date,
        "end": cfg.end_date,
        "working_srs": str(cfg.working_srs),
        "river_method": cfg.river_method,
        "river_dem_source": cfg.river_dem_source,
        "extra_xyz_cudem": cfg.extra_xyz_cudem,
        "timestamp": "run_scoped",
    }
    try:
        manifest_path.write_text(json.dumps(payload, indent=2))
    except (OSError, TypeError, ValueError):
        logger.debug("manifest write failed", exc_info=True)

    simple_stage_status = simple_river_stage_status_placeholder()
    legacy_transitional_components = [
        "legacy_structured_river_guidance",
        "channel_surface_transition_route",
    ]
    plan = RiverExecutionPlan(
        should_run=True,
        skip_reason=None,
        requested_method=requested_method,
        effective_method=effective_method,
        river_dir=river_dir,
        script_dir=script_dir,
        cache_dir=cache_dir,
        work_dir=work_dir,
        raw_hydro_cache=raw_hydro_cache,
        cached_bed_tif=cached_bed_tif,
        cached_depth_tif=cached_depth_tif,
        manifest_path=manifest_path,
        route_mode=ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
        target_contract_mode=ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
        simple_stage_status=simple_stage_status,
        legacy_transitional_components=legacy_transitional_components,
        simple_river_stage_outputs={},
    )
    river_report["execution_plan"] = plan.to_dict()
    river_report["route_mode"] = plan.route_mode
    river_report["target_contract_mode"] = plan.target_contract_mode
    river_report["simple_stage_status"] = simple_stage_status
    river_report["legacy_transitional_components"] = legacy_transitional_components
    return plan
