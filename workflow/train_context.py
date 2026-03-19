"""train_context.py — Shared context for SDB model training.

Consolidates the 38 parameters of ``train.train_sdb_model()`` into a
structured dataclass.  The function retains its original signature for
backward compatibility; callers can optionally pass a ``TrainContext``
via the ``ctx`` keyword argument instead of individual params.

Usage::

    from train_context import TrainContext
    tc = TrainContext.from_args(args, train_df=df, s2_paths=s2_paths, ...)
    rf, stumpf_lr, df_train, df_test, meta = train.train_sdb_model(ctx=tc)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class TrainContext:
    """Structured context for a single SDB training run.

    Groups the 38 parameters of ``train_sdb_model`` into logical sections.
    """

    # --- Input data ----------------------------------------------------------
    train_df: Any = None  # pd.DataFrame

    # --- Depth limits --------------------------------------------------------
    max_depth_sdb: float = 20.0
    min_training_points_for_sdb: int = 50
    max_depth_source: str = "physics"
    rmse_target_sdb: float = 0.5

    # --- Reproducibility -----------------------------------------------------
    seed: int = 42

    # --- Output dirs ---------------------------------------------------------
    plots_dir: Optional[Path] = None
    diagnostics_dir: Optional[Path] = None

    # --- Water class ---------------------------------------------------------
    water_class: str = "mixed"

    # --- Feature engineering --------------------------------------------------
    use_stumpf_depth: bool = False

    # --- Splitting strategy --------------------------------------------------
    spatial_split: bool = False
    spatial_cv_enabled: bool = False
    spatial_cv_strategy: str = "spatial_cluster"
    spatial_cv_folds: int = 5

    # --- Land mask -----------------------------------------------------------
    cw_min: float = 0.5
    land_max: float = 0.0
    land_mask_type: str = "auto"
    land_mask_water_val: Optional[int] = None
    land_mask_invert: bool = False
    land_mask_threshold: float = 0.5
    land_mask_nodata_is_water: bool = True

    # --- L-infinity (deep-water correction) ----------------------------------
    linf_enabled: bool = False
    linf_estimate: str = "none"
    linf_deepwater_nir_max: float = 0.03
    linf_deepwater_bright_max: float = 0.15
    linf_percentile: float = 1.0

    # --- Raster paths (for raster-based Linf estimation) ---------------------
    raster_paths: Optional[Dict[str, Any]] = None

    # --- Depth binning -------------------------------------------------------
    depth_bin_m: float = 0.5
    depth_binning: str = "quantile"
    max_depth_bins: int = 30
    min_samples_per_bin: int = 200

    # --- Run reporting -------------------------------------------------------
    rr: Any = None

    # --- Model bank ----------------------------------------------------------
    model_bank_dir: Optional[Path] = None
    model_bank_enabled: bool = True
    model_bank_max_samples: int = 100000
    model_bank_seed: int = 1337
    model_bank_retrain_min_new: int = 2000
    model_bank_context: Optional[Dict[str, Any]] = None

    # --- ATL03 admissibility summary -----------------------------------------
    atl03_admissibility_summary: Optional[Dict[str, Any]] = None

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------
    @classmethod
    def from_args(cls, args: Any, *, train_df: Any = None,
                  s2_paths: Optional[Dict[str, Any]] = None,
                  plots_dir: Optional[Path] = None,
                  diagnostics_dir: Optional[Path] = None,
                  rr: Any = None,
                  model_bank_dir: Optional[Path] = None,
                  model_bank_context: Optional[Dict[str, Any]] = None,
                  land_mask_type_override: Optional[str] = None,
                  land_mask_water_val_override: Optional[int] = None,
                  land_mask_invert_override: Optional[bool] = None,
                  linf_estimate_mode: Optional[str] = None,
                  ) -> "TrainContext":
        """Build a TrainContext from an argparse Namespace.

        Extracts all training-related attributes from ``args`` and applies
        optional overrides for land-mask config and Linf estimation mode.
        """
        # max_depth_sdb can be 'auto' (string) — resolve to float, default 20.0
        _raw_depth = getattr(args, "max_depth_sdb", 20.0)
        try:
            _max_depth = float(_raw_depth)
        except (ValueError, TypeError):
            _max_depth = 20.0  # 'auto' or other non-numeric → use default

        return cls(
            train_df=train_df,
            max_depth_sdb=_max_depth,
            seed=int(getattr(args, "seed", 42)),
            plots_dir=Path(plots_dir) if plots_dir else None,
            diagnostics_dir=Path(diagnostics_dir) if diagnostics_dir else None,
            water_class=str(getattr(args, "water_class", "mixed")),
            use_stumpf_depth=bool(getattr(args, "use_stumpf_depth", False)),
            min_training_points_for_sdb=int(getattr(args, "min_training_points_for_sdb", 50)),
            spatial_cv_enabled=bool(getattr(args, "spatial_cv", False)),
            spatial_cv_strategy=str(getattr(args, "spatial_cv_strategy", "spatial_cluster")),
            spatial_cv_folds=int(getattr(args, "spatial_cv_n_folds", 5)),
            cw_min=float(getattr(args, "cw_min", 0.5)),
            land_max=float(getattr(args, "land_max", 0.0)),
            land_mask_type=land_mask_type_override or str(getattr(args, "land_mask_type", "auto")),
            land_mask_water_val=land_mask_water_val_override if land_mask_water_val_override is not None else getattr(args, "land_mask_water_val", None),
            land_mask_invert=land_mask_invert_override if land_mask_invert_override is not None else bool(getattr(args, "land_mask_invert", False)),
            land_mask_threshold=float(getattr(args, "land_mask_threshold", 0.5)),
            linf_enabled=bool(getattr(args, "linf_enabled", False)),
            linf_estimate=linf_estimate_mode or str(getattr(args, "linf_estimate_deepwater", "none")),
            linf_deepwater_nir_max=float(getattr(args, "linf_deepwater_nir_max", 0.03)),
            linf_deepwater_bright_max=float(getattr(args, "linf_deepwater_bright_max", 0.15)),
            linf_percentile=float(getattr(args, "linf_percentile", 1.0)),
            raster_paths=s2_paths,
            rmse_target_sdb=float(getattr(args, "rmse_target_sdb", 0.5)),
            depth_bin_m=float(getattr(args, "depth_bin_m", 0.5)),
            depth_binning=str(getattr(args, "depth_binning", "quantile")),
            min_samples_per_bin=int(getattr(args, "min_samples_per_bin", 200)),
            max_depth_source=str(getattr(args, "max_depth_source", "physics")),
            rr=rr,
            model_bank_dir=Path(model_bank_dir) if model_bank_dir else None,
            model_bank_enabled=bool(getattr(args, "model_bank_enabled", True)),
            model_bank_max_samples=int(getattr(args, "bank_max_samples", 100000)),
            model_bank_seed=int(getattr(args, "bank_seed", 1337)),
            model_bank_retrain_min_new=int(getattr(args, "bank_retrain_min_new", 2000)),
            model_bank_context=model_bank_context,
        )

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to keyword arguments for ``train_sdb_model()``.

        This allows gradual migration: callers build a ``TrainContext`` and
        then unpack it into the existing function signature.
        """
        return {
            "train_df": self.train_df,
            "max_depth_sdb": self.max_depth_sdb,
            "seed": self.seed,
            "plots_dir": self.plots_dir,
            "water_class": self.water_class,
            "use_stumpf_depth": self.use_stumpf_depth,
            "min_training_points_for_sdb": self.min_training_points_for_sdb,
            "spatial_split": self.spatial_split,
            "spatial_cv_enabled": self.spatial_cv_enabled,
            "spatial_cv_strategy": self.spatial_cv_strategy,
            "spatial_cv_folds": self.spatial_cv_folds,
            "cw_min": self.cw_min,
            "land_max": self.land_max,
            "land_mask_type": self.land_mask_type,
            "land_mask_water_val": self.land_mask_water_val,
            "land_mask_invert": self.land_mask_invert,
            "land_mask_threshold": self.land_mask_threshold,
            "land_mask_nodata_is_water": self.land_mask_nodata_is_water,
            "linf_enabled": self.linf_enabled,
            "linf_estimate": self.linf_estimate,
            "linf_deepwater_nir_max": self.linf_deepwater_nir_max,
            "linf_deepwater_bright_max": self.linf_deepwater_bright_max,
            "linf_percentile": self.linf_percentile,
            "raster_paths": self.raster_paths,
            "rmse_target_sdb": self.rmse_target_sdb,
            "depth_bin_m": self.depth_bin_m,
            "depth_binning": self.depth_binning,
            "max_depth_bins": self.max_depth_bins,
            "min_samples_per_bin": self.min_samples_per_bin,
            "max_depth_source": self.max_depth_source,
            "diagnostics_dir": self.diagnostics_dir,
            "rr": self.rr,
            "model_bank_dir": self.model_bank_dir,
            "model_bank_enabled": self.model_bank_enabled,
            "model_bank_max_samples": self.model_bank_max_samples,
            "model_bank_seed": self.model_bank_seed,
            "model_bank_retrain_min_new": self.model_bank_retrain_min_new,
            "model_bank_context": self.model_bank_context,
            "atl03_admissibility_summary": self.atl03_admissibility_summary,
        }
