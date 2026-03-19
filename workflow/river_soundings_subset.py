from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import logging

log = logging.getLogger(__name__)


def write_soundings_subset_from_inputs(
    *,
    soundings_items,
    target_crs,
    depth_col: Optional[str],
    elev_col: Optional[str],
    x_col: Optional[str],
    y_col: Optional[str],
    soundings_crs: Optional[str],
    cfg: Any,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Write a reusable river soundings subset without depending on XS validity.

    This is used for the Pass-1 handoff in the river workflow so that subset
    creation cannot be blocked by missing/invalid bank picks or empty XS
    parameterization. The implementation reuses the canonical loaders / writers
    from xs_infer_bathy_raster.py to keep subset semantics identical between the
    standalone subset pass and the full XS inference path.
    """
    from xs_infer_bathy_raster import (  # local import to avoid heavy startup cycles
        _cap_soundings_for_memory,
        _load_soundings_many,
        _soundings_one_line,
        _write_soundings_subset_or_fail,
    )

    active_log = logger or log
    soundings = _load_soundings_many(
        soundings_items,
        target_crs=target_crs,
        depth_col=depth_col,
        elev_col=elev_col,
        x_col=x_col,
        y_col=y_col,
        soundings_crs=soundings_crs,
    )
    if soundings is None or soundings.empty:
        raise RuntimeError("No usable soundings were loaded for subset creation.")

    active_log.info("Loaded soundings for fallback subset creation: n=%d", int(len(soundings)))
    soundings, n_in_all = _cap_soundings_for_memory(soundings, cfg)
    subset_path = _write_soundings_subset_or_fail(soundings, cfg, require_output=True)
    by_src = None
    if "_src_file" in soundings.columns:
        by_src = {}
        vc = soundings["_src_file"].astype(str).value_counts()
        for k, v in vc.items():
            kk = Path(str(k)).stem if str(k) not in ["", "nan", "None"] else "unknown"
            by_src[kk] = int(v)
    active_log.info("%s", _soundings_one_line(Path(subset_path), int(len(soundings)), int(n_in_all), by_src))
    return Path(subset_path)


__all__ = ["write_soundings_subset_from_inputs"]
