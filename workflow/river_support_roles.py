from __future__ import annotations

from typing import Optional


RIVER_SUPPORT_CANONICAL_CLASSES = (
    "authoritative_interior",
    "bank_margin_only",
    "weak_supported_interior",
    "unsupported_interior",
)

RIVER_SUPPORT_CLASS_ALIASES = {
    "authoritative_interior": "authoritative_interior",
    "bank_margin_only": "bank_margin_only",
    "weak_supported_interior": "weak_supported_interior",
    "unsupported_interior": "unsupported_interior",
    "bed_supported": "authoritative_interior",
    "unsupported": "unsupported_interior",
    "authoritative_backbone": "authoritative_interior",
    "anchored_interpolated": "weak_supported_interior",
}

RIVER_SUPPORT_CLASS_FAMILY = {
    "authoritative_interior": "authoritative",
    "bank_margin_only": "bank_margin",
    "weak_supported_interior": "weak_support",
    "unsupported_interior": "unsupported",
}


def _maybe_float(value):
    try:
        if value is None:
            return None
        out = float(value)
        return out
    except Exception:
        return None


def canonical_river_support_class(source=None, **kwargs) -> str:
    """Return the canonical river-internal support role for a station-like object."""
    getter = getattr(source, "get", None)

    def _get(name, default=None):
        if getter is not None:
            value = getter(name, default)
            if value is not None:
                return value
        return kwargs.get(name, default)

    support_regime = str(_get("station_support_regime", _get("xs_support_template_class", "missing")) or "missing").strip()
    component_support_class = str(_get("component_support_class", "unknown") or "unknown").strip()
    bed_support_present = bool(_get("station_authoritative_bed_support_present", False))
    bank_margin_present = bool(_get("station_authoritative_bank_margin_present", False))
    true_measured_fraction = _maybe_float(_get("station_true_measured_xs_fraction", 0.0)) or 0.0
    channel_fraction = _maybe_float(_get("station_authoritative_channel_fraction", 0.0)) or 0.0
    bed_core_fraction = _maybe_float(_get("station_authoritative_bed_core_fraction", 0.0)) or 0.0
    bed_support_fraction = _maybe_float(_get("station_authoritative_bed_support_fraction", 0.0)) or 0.0
    bank_margin_fraction = _maybe_float(_get("station_authoritative_bank_margin_fraction", 0.0)) or 0.0

    if true_measured_fraction > 0.0 or bed_support_present or bed_support_fraction > 0.0 or channel_fraction > 0.0 or bed_core_fraction > 0.0:
        return "authoritative_interior"
    if bank_margin_present or bank_margin_fraction > 0.0 or support_regime in {"bank_only_low_confidence", "bank_only_authoritative"}:
        return "bank_margin_only"
    if component_support_class in {"unsupported_mainstem", "unsupported_side_component", "tiny_detached_component"}:
        return "unsupported_interior"
    if support_regime in {"unsupported", "missing", "none", ""}:
        return "unsupported_interior"
    return "weak_supported_interior"


def canonical_river_support_family(source=None, **kwargs) -> str:
    """Return the broad family for the canonical river-internal support role."""
    cls = canonical_river_support_class(source, **kwargs)
    if cls in {"authoritative_interior", "bank_margin_only"}:
        return "authoritative_margin_or_interior"
    if cls == "unsupported_interior":
        return "unsupported_interior"
    return "supported_non_authoritative_interior"


def is_authoritative_interior(source=None, **kwargs) -> bool:
    return canonical_river_support_class(source, **kwargs) == "authoritative_interior"


def is_bank_margin_only(source=None, **kwargs) -> bool:
    return canonical_river_support_class(source, **kwargs) == "bank_margin_only"


def is_weak_supported_interior(source=None, **kwargs) -> bool:
    return canonical_river_support_class(source, **kwargs) == "weak_supported_interior"


def is_unsupported_interior(source=None, **kwargs) -> bool:
    return canonical_river_support_class(source, **kwargs) == "unsupported_interior"


def river_support_roles_summary() -> dict:
    return {
        "canonical_classes": list(RIVER_SUPPORT_CANONICAL_CLASSES),
        "aliases": dict(RIVER_SUPPORT_CLASS_ALIASES),
        "families": dict(RIVER_SUPPORT_CLASS_FAMILY),
    }


__all__ = [
    "RIVER_SUPPORT_CANONICAL_CLASSES",
    "RIVER_SUPPORT_CLASS_ALIASES",
    "RIVER_SUPPORT_CLASS_FAMILY",
    "canonical_river_support_class",
    "canonical_river_support_family",
    "is_authoritative_interior",
    "is_bank_margin_only",
    "is_weak_supported_interior",
    "is_unsupported_interior",
    "river_support_roles_summary",
]
