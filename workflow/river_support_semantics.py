from __future__ import annotations

WEAK_SUPPORT_CLASSES = frozenset({
    "unsupported",
    "low_support_scaffolded",
    "unsupported_no_confident_offset",
    "bank_stage_only",
    "structure_only",
})
LONGITUDINAL_WEAK_SUPPORT = "weak"

__all__ = ["WEAK_SUPPORT_CLASSES", "LONGITUDINAL_WEAK_SUPPORT"]
