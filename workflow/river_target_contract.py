from __future__ import annotations

ACTIVE_INTERIOR_TARGET_AUTHORITATIVE = "authoritative_interior"
ACTIVE_INTERIOR_TARGET_BACKBONE = "backbone_led_interior"
ACTIVE_INTERIOR_TARGET_ALLOWED = (
    ACTIVE_INTERIOR_TARGET_AUTHORITATIVE,
    ACTIVE_INTERIOR_TARGET_BACKBONE,
)

_AUTHORITATIVE_ALIASES = {
    ACTIVE_INTERIOR_TARGET_AUTHORITATIVE,
    "local_authoritative_reconciled",
    "authoritative_interior_bed",
    "authoritative_in_channel",
    "authoritative_bed_core",
}

_BACKBONE_ALIASES = {
    ACTIVE_INTERIOR_TARGET_BACKBONE,
    "generalized_longitudinal_section",
    "longitudinal_backbone_template",
    "smoothed_backbone_target",
    "backbone_target",
}


def normalize_active_target_source(source: str | None) -> str:
    src = str(source or "").strip()
    if src in _AUTHORITATIVE_ALIASES:
        return ACTIVE_INTERIOR_TARGET_AUTHORITATIVE
    if src in _BACKBONE_ALIASES:
        return ACTIVE_INTERIOR_TARGET_BACKBONE
    return "missing"


def is_active_target_authoritative(source: str | None) -> bool:
    return normalize_active_target_source(source) == ACTIVE_INTERIOR_TARGET_AUTHORITATIVE


def is_active_target_backbone_led(source: str | None) -> bool:
    return normalize_active_target_source(source) == ACTIVE_INTERIOR_TARGET_BACKBONE


def build_active_target_record(*, source: str | None, z_m: float | None, reason: str | None = None) -> dict[str, object]:
    return {
        "active_interior_target_source": normalize_active_target_source(source),
        "active_interior_target_z_m": z_m,
        "active_interior_target_reason": str(reason or normalize_active_target_source(source)),
    }
