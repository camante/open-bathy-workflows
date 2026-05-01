from __future__ import annotations
from typing import Any


def compute_gap_fill_confidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"status": "not_evaluated", "reason": "channel_template_legacy_path_not_active"}

__all__ = ["compute_gap_fill_confidence"]
