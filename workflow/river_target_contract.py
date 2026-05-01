from __future__ import annotations

_ALIASES = {
    "local_authoritative_reconciliation": "station_target_local_authoritative_reconciliation",
    "section_tendency": "station_target_section_tendency",
    "thalweg_default": "generalized_thalweg_default_tendency",
    "": "missing",
    "nan": "missing",
    "none": "missing",
}


def normalize_active_target_source(value: object) -> str:
    text = str(value if value is not None else "missing").strip()
    key = text.lower()
    return _ALIASES.get(key, text or "missing")


__all__ = ["normalize_active_target_source"]
