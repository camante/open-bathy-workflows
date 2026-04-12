from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SDBExecutionPlan:
    should_run: bool
    skip_reason: Optional[str]
    activation_should_run: bool
    shared_domain_mask: Optional[Path]
    shared_domain_pixels: Optional[int]
    sdb_dir: Path
    script_dir: Path
    logs_dir: Path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_run": bool(self.should_run),
            "skip_reason": self.skip_reason,
            "activation_should_run": bool(self.activation_should_run),
            "shared_domain_mask": str(self.shared_domain_mask) if self.shared_domain_mask is not None else None,
            "shared_domain_pixels": int(self.shared_domain_pixels) if self.shared_domain_pixels is not None else None,
            "sdb_dir": str(self.sdb_dir),
            "script_dir": str(self.script_dir),
            "logs_dir": str(self.logs_dir),
        }


def determine_sdb_execution_plan(*, cfg: Any, report: Dict[str, Any], ensure_dir_fn, parse_aoi_bbox_fn, count_mask_water_pixels_fn, logger) -> SDBExecutionPlan:
    sdb_report = report.setdefault("sdb", {})
    sdb_dir = ensure_dir_fn(Path(cfg.out_dir) / "sdb")
    logs_dir = ensure_dir_fn(Path(cfg.out_dir) / "logs")
    script_dir = Path(__file__).parent

    shared_sdb_mask = getattr(cfg, "sdb_guidance_domain_mask", None)
    shared_sdb_mask_resolved = Path(shared_sdb_mask).resolve() if shared_sdb_mask else None
    validated_shared_sdb_mask = getattr(cfg, "validated_sdb_guidance_domain_mask", None)
    validated_shared_sdb_mask_resolved = Path(validated_shared_sdb_mask).resolve() if validated_shared_sdb_mask else None
    validated_shared_sdb_pixels = getattr(cfg, "validated_sdb_guidance_domain_pixels", None)
    shared_sdb_pixels = None
    shared_domain_activation = ((report.get("guidance_domains") or {}).get("activation") or {}).get("derived_activation", {})
    shared_sdb_should_run = bool(shared_domain_activation.get("sdb_should_run", False))
    if validated_shared_sdb_mask_resolved and validated_shared_sdb_mask_resolved.exists() and validated_shared_sdb_pixels is not None:
        if (shared_sdb_mask_resolved is not None) and (shared_sdb_mask_resolved != validated_shared_sdb_mask_resolved):
            raise RuntimeError(
                f"shared_sdb_domain_path_mismatch: cfg.sdb_guidance_domain_mask={shared_sdb_mask_resolved} "
                f"validated={validated_shared_sdb_mask_resolved}"
            )
        shared_sdb_pixels = int(validated_shared_sdb_pixels)
        shared_sdb_mask = validated_shared_sdb_mask_resolved
        sdb_report["shared_domain_pixels"] = int(shared_sdb_pixels)
        sdb_report["shared_domain_mask_used"] = str(validated_shared_sdb_mask_resolved)
    elif shared_sdb_mask_resolved and shared_sdb_mask_resolved.exists():
        aoi_bbox = parse_aoi_bbox_fn(str(cfg.aoi))
        shared_sdb_pixels = count_mask_water_pixels_fn(shared_sdb_mask_resolved, aoi_bbox) if aoi_bbox else None
        if shared_sdb_pixels is not None:
            sdb_report["shared_domain_pixels"] = int(shared_sdb_pixels)
            sdb_report["shared_domain_mask_used"] = str(shared_sdb_mask_resolved)

    sdb_report["shared_domain_activation_should_run"] = bool(shared_sdb_should_run)
    if shared_sdb_should_run:
        if shared_sdb_pixels is None or shared_sdb_pixels <= 0:
            raise RuntimeError(
                f"shared_sdb_domain_activation_mismatch: activation requested SDB from shared domains, "
                f"but validated shared SDB mask is empty or missing. mask={shared_sdb_mask} pixels={shared_sdb_pixels}"
            )
        logger.info(
            "[SDB] Shared-domain activation confirmed: using shared SDB domain with %d pixels. shared_mask=%s",
            int(shared_sdb_pixels),
            shared_sdb_mask,
        )
        plan = SDBExecutionPlan(
            should_run=True,
            skip_reason=None,
            activation_should_run=True,
            shared_domain_mask=Path(shared_sdb_mask) if shared_sdb_mask else None,
            shared_domain_pixels=int(shared_sdb_pixels),
            sdb_dir=sdb_dir,
            script_dir=script_dir,
            logs_dir=logs_dir,
        )
    elif shared_sdb_pixels is None or shared_sdb_pixels <= 0:
        logger.info("[SDB] Shared-domain skip: validated shared SDB domain is empty; skipping SDB. shared_mask=%s", shared_sdb_mask)
        sdb_report["skipped_reason"] = "shared_domain_empty"
        plan = SDBExecutionPlan(
            should_run=False,
            skip_reason="shared_domain_empty",
            activation_should_run=False,
            shared_domain_mask=Path(shared_sdb_mask) if shared_sdb_mask else None,
            shared_domain_pixels=int(shared_sdb_pixels) if shared_sdb_pixels is not None else None,
            sdb_dir=sdb_dir,
            script_dir=script_dir,
            logs_dir=logs_dir,
        )
    else:
        logger.info(
            "[SDB] Shared-domain mask is nonempty (%d pixels) but activation did not request SDB; proceeding because run_sdb was called explicitly. shared_mask=%s",
            int(shared_sdb_pixels),
            shared_sdb_mask,
        )
        plan = SDBExecutionPlan(
            should_run=True,
            skip_reason=None,
            activation_should_run=False,
            shared_domain_mask=Path(shared_sdb_mask) if shared_sdb_mask else None,
            shared_domain_pixels=int(shared_sdb_pixels),
            sdb_dir=sdb_dir,
            script_dir=script_dir,
            logs_dir=logs_dir,
        )

    sdb_report["execution_plan"] = plan.to_dict()
    return plan
