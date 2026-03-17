"""xs_infer_context.py — Shared context for cross-section bathymetry inference.

Consolidates the 46 parameters of ``xs_infer_bathy_raster.infer_bathy()``
into a structured dataclass.  Callers can build an ``XSInferContext`` and
unpack it via ``to_kwargs()`` without changing the function signature.

Usage::

    from xs_infer_context import XSInferContext
    ctx = XSInferContext.from_args(args, xs_gpkg=xs_out, dem_path=dem, ...)
    infer_bathy(**ctx.to_kwargs())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class XSInferContext:
    """Structured context for a single XS bathymetry inference run.

    Groups the 46 parameters of ``infer_bathy`` into logical sections.
    """

    # --- Input geometry -------------------------------------------------------
    xs_gpkg: Optional[Path] = None
    out_gpkg: Optional[Path] = None
    xs_lines_layer: str = "xs_lines"
    xs_points_layer: str = "xs_points"

    # --- DEM ------------------------------------------------------------------
    dem_path: Optional[Path] = None

    # --- Soundings ------------------------------------------------------------
    soundings_path: Any = None  # str, Path, or list
    soundings_depth_col: Optional[str] = None
    soundings_elev_col: Optional[str] = None
    soundings_x_col: Optional[str] = None
    soundings_y_col: Optional[str] = None
    soundings_crs: Optional[str] = None

    # --- InferConfig (from xs_infer_bathy_raster) -----------------------------
    cfg: Any = None  # InferConfig dataclass

    # --- River network attributes / calibration -------------------------------
    river_gpkg: Optional[Path] = None
    rivers_layer: str = "rivers_clip"
    drain_area_field: Optional[str] = None
    slope_field: Optional[str] = None
    manning_q_field: Optional[str] = None
    manning_dist_to_mouth_field: Optional[str] = None

    # --- USGS gauge calibration -----------------------------------------------
    usgs_sites: Optional[List[str]] = None
    usgs_start: Optional[str] = None
    usgs_end: Optional[str] = None
    usgs_cache_dir: Optional[Path] = None

    # --- Width-stage calibration ----------------------------------------------
    width_stage_csv: Optional[List[str]] = None

    # --- Raster output paths --------------------------------------------------
    template_raster: Optional[Path] = None
    out_bathy_raster: Optional[Path] = None
    out_mask_raster: Optional[Path] = None
    out_uncert_raster: Optional[Path] = None

    # --- Rasterization options ------------------------------------------------
    raster_value_col: str = "z_bed_pred_m"
    raster_uncert_col: str = "uncert_m"
    continuous: str = "walid"
    continuous_buffer_m: Optional[float] = None
    continuous_k: int = 12
    idw_power: float = 2.0
    aniso_along_scale_m: float = 500.0
    aniso_cross_scale_m: float = 30.0
    thalweg_weight: float = 6.0
    nodata: float = -9999.0
    overlap_reducer: str = "min"

    # --- Channel mask ---------------------------------------------------------
    channel_mask_raster: Optional[Path] = None
    channel_mask_inside_value: int = 1
    channel_mask_invert: bool = False

    # --- Interpolation limits -------------------------------------------------
    max_query_dist_m: Optional[float] = None
    thalweg_only: bool = False
    thalweg_densify_step_m: Optional[float] = None

    # --- Accounting / metadata output -----------------------------------------
    out_accounting_json: Optional[Path] = None
    out_meta_json: Optional[Path] = None

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------
    @classmethod
    def from_args(cls, args: Any, *,
                  xs_gpkg: Optional[Path] = None,
                  out_gpkg: Optional[Path] = None,
                  dem_path: Optional[Path] = None,
                  soundings_path: Any = None,
                  cfg: Any = None,
                  river_gpkg: Optional[Path] = None,
                  template_raster: Optional[Path] = None,
                  out_bathy_raster: Optional[Path] = None,
                  out_mask_raster: Optional[Path] = None,
                  out_uncert_raster: Optional[Path] = None,
                  ) -> "XSInferContext":
        """Build context from parsed CLI args with explicit path overrides."""
        return cls(
            xs_gpkg=xs_gpkg,
            out_gpkg=out_gpkg,
            dem_path=dem_path,
            soundings_path=soundings_path or getattr(args, "soundings", None),
            soundings_depth_col=getattr(args, "soundings_depth_col", None),
            soundings_elev_col=getattr(args, "soundings_elev_col", None),
            soundings_x_col=getattr(args, "soundings_x_col", None),
            soundings_y_col=getattr(args, "soundings_y_col", None),
            soundings_crs=getattr(args, "soundings_crs", None),
            cfg=cfg,
            river_gpkg=river_gpkg,
            rivers_layer=getattr(args, "rivers_layer", "rivers_clip"),
            drain_area_field=getattr(args, "drain_area_field", None),
            slope_field=getattr(args, "slope_field", None),
            manning_q_field=getattr(args, "manning_q_field", None),
            manning_dist_to_mouth_field=getattr(args, "manning_dist_to_mouth_field", None),
            usgs_sites=getattr(args, "usgs_sites", None),
            usgs_start=getattr(args, "usgs_start", None),
            usgs_end=getattr(args, "usgs_end", None),
            usgs_cache_dir=Path(args.usgs_cache_dir) if getattr(args, "usgs_cache_dir", None) else None,
            width_stage_csv=getattr(args, "width_stage_csv", None),
            template_raster=template_raster,
            out_bathy_raster=out_bathy_raster,
            out_mask_raster=out_mask_raster,
            out_uncert_raster=out_uncert_raster,
            raster_value_col=getattr(args, "raster_value_col", "z_bed_pred_m"),
            raster_uncert_col=getattr(args, "raster_uncert_col", "uncert_m"),
            continuous=getattr(args, "continuous", "walid"),
            continuous_buffer_m=getattr(args, "continuous_buffer_m", None),
            continuous_k=int(getattr(args, "continuous_k", 12)),
            idw_power=float(getattr(args, "idw_power", 2.0)),
            aniso_along_scale_m=float(getattr(args, "aniso_along_scale_m", 500.0)),
            aniso_cross_scale_m=float(getattr(args, "aniso_cross_scale_m", 30.0)),
            thalweg_weight=float(getattr(args, "thalweg_weight", 6.0)),
            nodata=float(getattr(args, "nodata", -9999.0)),
            overlap_reducer=getattr(args, "overlap_reducer", "min"),
            channel_mask_raster=Path(args.channel_mask_raster) if getattr(args, "channel_mask_raster", None) else None,
            channel_mask_inside_value=int(getattr(args, "channel_mask_inside_value", 1)),
            channel_mask_invert=bool(getattr(args, "channel_mask_invert", False)),
            max_query_dist_m=getattr(args, "max_query_dist_m", None),
            thalweg_only=bool(getattr(args, "thalweg_only", False)),
            thalweg_densify_step_m=getattr(args, "thalweg_densify_step_m", None),
            out_accounting_json=Path(args.out_accounting_json) if getattr(args, "out_accounting_json", None) else None,
            out_meta_json=Path(args.out_meta_json) if getattr(args, "out_meta_json", None) else None,
        )

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to keyword arguments for ``infer_bathy()``."""
        return {
            "xs_gpkg": self.xs_gpkg,
            "out_gpkg": self.out_gpkg,
            "dem_path": self.dem_path,
            "soundings_path": self.soundings_path,
            "soundings_depth_col": self.soundings_depth_col,
            "soundings_elev_col": self.soundings_elev_col,
            "soundings_x_col": self.soundings_x_col,
            "soundings_y_col": self.soundings_y_col,
            "soundings_crs": self.soundings_crs,
            "cfg": self.cfg,
            "xs_lines_layer": self.xs_lines_layer,
            "xs_points_layer": self.xs_points_layer,
            "river_gpkg": self.river_gpkg,
            "rivers_layer": self.rivers_layer,
            "drain_area_field": self.drain_area_field,
            "slope_field": self.slope_field,
            "manning_q_field": self.manning_q_field,
            "manning_dist_to_mouth_field": self.manning_dist_to_mouth_field,
            "usgs_sites": self.usgs_sites,
            "usgs_start": self.usgs_start,
            "usgs_end": self.usgs_end,
            "usgs_cache_dir": self.usgs_cache_dir,
            "width_stage_csv": self.width_stage_csv,
            "template_raster": self.template_raster,
            "out_bathy_raster": self.out_bathy_raster,
            "out_mask_raster": self.out_mask_raster,
            "out_uncert_raster": self.out_uncert_raster,
            "raster_value_col": self.raster_value_col,
            "raster_uncert_col": self.raster_uncert_col,
            "continuous": self.continuous,
            "continuous_buffer_m": self.continuous_buffer_m,
            "continuous_k": self.continuous_k,
            "idw_power": self.idw_power,
            "aniso_along_scale_m": self.aniso_along_scale_m,
            "aniso_cross_scale_m": self.aniso_cross_scale_m,
            "thalweg_weight": self.thalweg_weight,
            "nodata": self.nodata,
            "overlap_reducer": self.overlap_reducer,
            "channel_mask_raster": self.channel_mask_raster,
            "channel_mask_inside_value": self.channel_mask_inside_value,
            "channel_mask_invert": self.channel_mask_invert,
            "max_query_dist_m": self.max_query_dist_m,
            "thalweg_only": self.thalweg_only,
            "thalweg_densify_step_m": self.thalweg_densify_step_m,
            "out_accounting_json": self.out_accounting_json,
            "out_meta_json": self.out_meta_json,
        }
