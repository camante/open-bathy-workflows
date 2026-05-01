from __future__ import annotations

from typing import Any

ALLOWED_RIVER_VALIDATION_QUESTIONS: tuple[str, ...] = (
    "Did AOI export exactly match the canonical parent window?",
    "Did final DEM equal AOI export?",
    "Did authoritative lock preserve measured cells?",
    "Did WSE/backbone behave physically enough to inspect?",
    "Did north/south use the same parent?",
    "What was the first failing artifact?",
)

OPTIONAL_NOISE_POLICY: dict[str, Any] = {
    "phase": "phase9",
    "policy": "suppress_optional_placeholder_validation_when_not_configured",
    "does_not_change_construction": True,
    "does_not_drive_routing": True,
    "does_not_repair_outputs": True,
    "verify_only": True,
}


def benchmark_requested(args: Any) -> bool:
    """Return True only when benchmark validation was explicitly requested."""
    return bool(getattr(args, "benchmark_holdout", None) or getattr(args, "benchmark_auto_holdout", False))


def regression_case_requested(args: Any) -> bool:
    """Return True only when postrun adjacent/nested regression inputs were configured."""
    return bool(
        getattr(args, "adjacent_aoi_peer_dir", None)
        or getattr(args, "nested_aoi_parent_dir", None)
        or getattr(args, "nested_aoi_child_dir", None)
        or getattr(args, "seam_regression_case", None)
    )


def validation_case_requested(*, validation_truth: Any = None, case_specs: list[Any] | None = None, case_manifest: Any = None) -> bool:
    """Return True only when an external validation truth/case was configured."""
    return bool(validation_truth or case_specs or case_manifest)


def suppressed_optional_validation_payload(stage: str, *, reason: str = "not_configured") -> dict[str, Any]:
    """Small report payload for optional checks intentionally not run.

    This is intentionally not a stage receipt. It documents that the check was
    skipped to avoid placeholder validation noise; it must not affect routing.
    """
    return {
        "status": "not_run",
        "stage": str(stage),
        "reason": str(reason),
        "visibility": "suppressed_by_default",
        "diagnostic_only": True,
        "may_reroute_or_repair_outputs": False,
    }
