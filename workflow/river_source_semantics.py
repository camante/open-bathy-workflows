from __future__ import annotations

"""Canonical string semantics for legacy river channel-frame/source labels."""

_SURFACE_GUIDANCE_SOURCES = {
    "resolved_channel_bed",
    "authoritative_in_channel",
    "authoritative_backbone",
    "graph_backbone",
    "xs_profile_resampled",
    "station_target_section_tendency",
    "station_target_local_authoritative_reconciliation",
    "generalized_thalweg_default_tendency",
    "bank_edge_geometry_constraint",
}


def station_target_node_source(*, local_reconciled: bool = False, **_) -> str:
    return "station_target_local_authoritative_reconciliation" if local_reconciled else "station_target_section_tendency"


def is_surface_guidance_source(value: object) -> bool:
    return str(value or "").strip() in _SURFACE_GUIDANCE_SOURCES


__all__ = ["station_target_node_source", "is_surface_guidance_source"]
