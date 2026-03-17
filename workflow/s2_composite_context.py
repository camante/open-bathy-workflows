"""s2_composite_context.py — Shared context for Sentinel-2 composite building.

Consolidates the ~50 parameters of
``s2_optics.build_weighted_shared_date_composite()`` into a structured
dataclass.  Uses the same ``to_kwargs()`` bridge pattern as ``TrainContext``
for gradual migration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class S2CompositeContext:
    """Structured context for a single Sentinel-2 composite build.

    Groups the parameters of ``build_weighted_shared_date_composite()``
    into logical sections.
    """

    # --- Spatial / temporal extent -------------------------------------------
    out_dir: Optional[Path] = None
    bbox_wesn: Optional[Tuple[float, float, float, float]] = None
    start_date: str = ""
    end_date: str = ""

    # --- Cloud / scene filtering ---------------------------------------------
    max_cloud: float = 20.0
    scene_limit: int = 10
    preferred_months: Optional[List[int]] = None
    min_scene_valid_frac: float = 0.90

    # --- STAC search ---------------------------------------------------------
    shared_date_mode: str = "strict"
    stac_url: str = "https://earth-search.aws.element84.com/v1"
    collection: str = "sentinel-2-l2a"
    stac_limit: int = 200
    stac_page_limit: Optional[int] = None
    stac_max_items: int = 5000
    stac_chunk_months: int = 0

    # --- Download / cache ----------------------------------------------------
    cache_dir: Optional[Path] = None
    download_workers: int = 8
    clean_cache: bool = False
    cache_strict: bool = False
    cache_code_strict: bool = False
    cache_ignore_code: bool = True

    # --- SCL / masking -------------------------------------------------------
    scl_dilate: int = 1

    # --- Harmonization -------------------------------------------------------
    harmonize: bool = True
    harmonize_max_abs_offset: float = 0.03
    deepwater_nir_max: float = 0.03
    deepwater_bright_max: float = 0.15

    # --- Pixel-level quality gating ------------------------------------------
    b02_thresh: float = 0.25
    allow_bright_shallow_pixels: bool = False
    bright_shallow_nir_max: float = 0.03
    apply_gl_turbidity_reject: bool = False
    gl_nir_max: float = 0.12
    gl_nir_green_ratio_max: float = 0.60
    gl_red_max: float = 0.08

    # --- Spatial weighting ---------------------------------------------------
    feather_dist_cap: int = 300
    edge_weight_power: float = 2.0
    edge_weight_min: float = 0.0

    # --- Temporal aggregation ------------------------------------------------
    temporal_median_k: int = 0
    single_best_date: bool = False

    # --- Coastline-aware date QC ---------------------------------------------
    coastline_mask_path: str = ""
    coastline_mask_water_value: int = 0
    coastline_mask_invert: bool = False
    coastline_erode_px: int = 2
    date_qc_enable: bool = True
    date_outlier_z: float = 3.5
    date_score_weight_valid: float = 2.0
    date_score_weight_glint: float = 1.0
    date_score_weight_deepwater: float = 1.0

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------
    @classmethod
    def from_args(cls, args: Any, *,
                  out_dir: Optional[Path] = None,
                  bbox_wesn: Optional[Tuple[float, float, float, float]] = None,
                  cache_dir: Optional[Path] = None,
                  coastline_mask_path: str = "",
                  ) -> "S2CompositeContext":
        """Build from an argparse Namespace."""
        return cls(
            out_dir=Path(out_dir) if out_dir else None,
            bbox_wesn=bbox_wesn,
            start_date=str(getattr(args, "start", "")),
            end_date=str(getattr(args, "end", "")),
            max_cloud=float(getattr(args, "cloud", 20.0)),
            scene_limit=int(getattr(args, "s2_scene_limit", 10)),
            preferred_months=getattr(args, "preferred_months", None),
            min_scene_valid_frac=float(getattr(args, "min_scene_valid_frac", 0.90)),
            stac_max_items=int(getattr(args, "stac_max_items", 5000)),
            stac_page_limit=getattr(args, "stac_page_limit", None),
            stac_chunk_months=int(getattr(args, "stac_chunk_months", 0)),
            cache_dir=Path(cache_dir) if cache_dir else None,
            download_workers=int(getattr(args, "download_workers", 8)),
            scl_dilate=int(getattr(args, "scl_dilate", 1)),
            harmonize=bool(getattr(args, "harmonize", True)),
            deepwater_nir_max=float(getattr(args, "deepwater_nir_max", 0.03)),
            deepwater_bright_max=float(getattr(args, "deepwater_bright_max", 0.15)),
            b02_thresh=float(getattr(args, "mask_bright_pixels", 0.25)),
            allow_bright_shallow_pixels=bool(getattr(args, "allow_bright_shallow_pixels", False)),
            bright_shallow_nir_max=float(getattr(args, "bright_shallow_nir_max", 0.03)),
            apply_gl_turbidity_reject=bool(getattr(args, "apply_gl_turbidity_reject", False)),
            gl_nir_max=float(getattr(args, "gl_nir_max", 0.12)),
            gl_nir_green_ratio_max=float(getattr(args, "gl_nir_green_ratio_max", 0.60)),
            gl_red_max=float(getattr(args, "gl_red_max", 0.08)),
            edge_weight_power=float(getattr(args, "edge_weight_power", 2.0)),
            edge_weight_min=float(getattr(args, "edge_weight_min", 0.0)),
            temporal_median_k=int(getattr(args, "temporal_median_k", 0)),
            single_best_date=bool(getattr(args, "single_best_date", False)),
            coastline_mask_path=coastline_mask_path,
            date_qc_enable=bool(getattr(args, "date_qc_enable", True)),
            clean_cache=bool(getattr(args, "clean_cache", False)),
            cache_strict=bool(getattr(args, "cache_strict", False)),
            cache_code_strict=bool(getattr(args, "cache_code_strict", False)),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to keyword arguments for ``build_weighted_shared_date_composite()``."""
        return {f: getattr(self, f) for f in self.__dataclass_fields__}
