"""sdb_context.py — Shared run context for the SDB pipeline.

Consolidates the bag of paths, directories, config, and runtime state that
the extracted ``sdb_main`` helper functions (``_acquire_s2_and_atl_data``,
``_run_fusion``, ``_run_sdb_prediction``, ``_write_sdb_manifest``, etc.)
all need.  Instead of threading 13–16 positional parameters through each
call, create one ``SDBRunContext`` up front and pass it around.

Usage in sdb_main.py::

    ctx = SDBRunContext.from_args(args, out_root=out_root, run_id=run_id)
    _acquire_s2_and_atl_data(ctx, ...)
    _run_fusion(ctx, ...)

The dataclass is intentionally flat and uses ``Optional`` liberally so
that fields can be populated incrementally as the pipeline progresses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class SDBRunContext:
    """Shared context object for a single SDB pipeline run.

    Groups the runtime state that was previously passed as separate
    positional parameters to every extracted helper function.

    Attributes are grouped into logical sections:

    **Identity & config**
        run_id, args (the parsed argparse Namespace), config overrides.

    **Directories**
        out_root, dir_data, dir_rast, dir_model, dir_logs, dir_plot.

    **Cache directories**
        s2_cache, atl_cache, mask_cache.

    **AOI**
        bbox_wesn (W, E, S, N), bbox_list [W, S, E, N].

    **Acquired data**
        s2_paths, atl_files_map, land_mask_out.

    **Training state**
        df_train, df_test, fused_df, chosen_max_depth, stumpf_lr.

    **Modules (lazy-imported)**
        mod_s2_optics, mod_atl, mod_fusion, mod_predict.

    **Run reporting**
        rr (RunReport instance or None).
    """

    # --- Identity & config ---------------------------------------------------
    run_id: str = ""
    args: Any = None  # argparse.Namespace
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # --- Directories ---------------------------------------------------------
    out_root: Optional[Path] = None
    dir_data: Optional[Path] = None
    dir_rast: Optional[Path] = None
    dir_model: Optional[Path] = None
    dir_logs: Optional[Path] = None
    dir_plot: Optional[Path] = None

    # --- Cache directories ---------------------------------------------------
    s2_cache: Optional[Path] = None
    atl_cache: Optional[Path] = None
    mask_cache: Optional[Path] = None

    # --- AOI -----------------------------------------------------------------
    bbox_wesn: Optional[tuple] = None  # (W, E, S, N)
    bbox_list: Optional[list] = None   # [W, S, E, N] for STAC queries

    # --- Acquired data -------------------------------------------------------
    s2_paths: Optional[Dict[str, Any]] = None
    atl_files_map: Optional[Dict[str, List[str]]] = field(default_factory=lambda: {"ATL03": [], "ATL24": []})
    land_mask_out: Optional[Path] = None

    # --- Training state ------------------------------------------------------
    df_train: Any = None  # pd.DataFrame
    df_test: Any = None   # pd.DataFrame
    fused_df: Any = None  # pd.DataFrame
    chosen_max_depth: Optional[float] = None
    stumpf_lr: Any = None  # sklearn model or None
    model_tier: int = 1
    tier_quality: Any = None  # TrainingDataQuality or None

    # --- Land mask config (extracted from args for convenience) ---------------
    land_mask_type: Optional[str] = None
    land_mask_water_val: Optional[float] = None
    land_mask_invert: bool = False
    land_mask_threshold: float = 0.5

    # --- Output paths --------------------------------------------------------
    out_tif: Optional[Path] = None
    raster_quickstats: Optional[Dict[str, Any]] = None

    # --- Physics-derived depth limit (from Kd estimation) --------------------
    kd_max_depth_m: Optional[float] = None

    # --- Modules (lazy-imported) ---------------------------------------------
    mod_s2_optics: Any = None
    mod_atl: Any = None
    mod_fusion: Any = None
    mod_predict: Any = None

    # --- Run reporting -------------------------------------------------------
    rr: Any = None  # RunReport or None
    fallback_registry: Any = None

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------
    @classmethod
    def from_args(cls, args: Any, *, out_root: Path, run_id: str,
                  rr: Any = None) -> "SDBRunContext":
        """Create a context from parsed CLI args and the output root.

        Populates directory paths following the standard layout and copies
        land-mask config fields from ``args`` for easy access.
        """
        ctx = cls(
            run_id=run_id,
            args=args,
            out_root=Path(out_root),
            dir_data=Path(out_root) / "data",
            dir_rast=Path(out_root) / "rasters",
            dir_model=Path(out_root) / "model",
            dir_logs=Path(out_root) / "logs",
            dir_plot=Path(out_root) / "plots",
            rr=rr,
        )

        # Copy land-mask fields from args if present
        for attr in ("land_mask_type", "land_mask_water_val",
                     "land_mask_invert", "land_mask_threshold"):
            if hasattr(args, attr):
                setattr(ctx, attr, getattr(args, attr))

        return ctx

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------
    def ensure_dirs(self) -> None:
        """Create all output subdirectories if they don't exist."""
        for d in (self.out_root, self.dir_data, self.dir_rast,
                  self.dir_model, self.dir_logs, self.dir_plot):
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)

    def set_bbox_from_aoi(self, aoi_str: str) -> None:
        """Parse AOI string and populate bbox_wesn and bbox_list."""
        from pipeline.aoi import parse_aoi_wesn
        wesn = parse_aoi_wesn(aoi_str, strict=True)
        self.bbox_wesn = wesn
        w, e, s, n = wesn
        self.bbox_list = [w, s, e, n]

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of the context state."""
        return {
            "run_id": self.run_id,
            "out_root": str(self.out_root) if self.out_root else None,
            "bbox_wesn": self.bbox_wesn,
            "chosen_max_depth": self.chosen_max_depth,
            "has_s2": self.s2_paths is not None,
            "has_fused_df": self.fused_df is not None,
            "has_rr": self.rr is not None,
        }
