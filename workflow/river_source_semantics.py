from __future__ import annotations

GENERALIZED_STATION_TARGET_SOURCES = {
    "generalized_longitudinal_section",
    "generalized_longitudinal_section_local_authoritative_reconciliation",
    "canonical_station_target",
    "fitted_bank_core_section",
    "longitudinal_backbone_template",
    "legacy_backbone_fallback",
}

RECONCILED_STATION_TARGET_SOURCES = {
    "generalized_longitudinal_section_local_authoritative_reconciliation",
    "station_target_local_authoritative_reconciliation",
}

SURFACE_GUIDANCE_SOURCES = {
    "graph_backbone",
    "resolved_channel_bed",
    "station_target_section_tendency",
    "station_target_local_authoritative_reconciliation",
    "generalized_thalweg_default_tendency",
}


def normalize_station_target_source_class(source: str | None, *, local_reconciled: bool = False, target_present: bool = False) -> str:
    src = str(source or "").strip()
    if src in RECONCILED_STATION_TARGET_SOURCES:
        return "generalized_longitudinal_section_local_authoritative_reconciliation"
    if src in {"canonical_station_target", "station_target_section_tendency"}:
        return "generalized_longitudinal_section"
    if src in {"longitudinal_backbone_template", "legacy_backbone_fallback", "fitted_bank_core_section"}:
        return "generalized_longitudinal_section"
    if src:
        return src
    if local_reconciled:
        return "generalized_longitudinal_section_local_authoritative_reconciliation"
    if target_present:
        return "generalized_longitudinal_section"
    return "missing"


def station_target_node_source(*, local_reconciled: bool = False) -> str:
    return (
        "station_target_local_authoritative_reconciliation"
        if bool(local_reconciled)
        else "station_target_section_tendency"
    )


def is_generalized_rebuild_target_source(source: str | None) -> bool:
    src = normalize_station_target_source_class(source)
    return src in {
        "generalized_longitudinal_section",
        "generalized_longitudinal_section_local_authoritative_reconciliation",
    }


def is_surface_guidance_source(source: str | None) -> bool:
    src = str(source or "").strip()
    return src in SURFACE_GUIDANCE_SOURCES
