from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from nodata_utils import resolve_reproject_nodata_value
from support_classes import RegimeClass, SupportClass


def _align_raster_to_template(*, template_path: Path, src_path: Optional[Path], dtype: str = "uint8", nodata_value: float | int = 0):
    if src_path is None or not Path(src_path).exists():
        return None
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(template_path) as tmpl:
        arr = np.full((tmpl.height, tmpl.width), nodata_value, dtype=dtype)
        with rasterio.open(src_path) as src:
            src_nodata = resolve_reproject_nodata_value(src.nodata, default=float(nodata_value))
            reproject(
                source=rasterio.band(src, 1),
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=tmpl.transform,
                dst_crs=tmpl.crs,
                dst_nodata=nodata_value,
                resampling=Resampling.nearest,
            )
    return arr


def build_contract_target_mask_array(
    *,
    template_path: Path,
    eligible_fill_mask_path: Optional[Path] = None,
    support_class_path: Optional[Path] = None,
    regime_class_path: Optional[Path] = None,
    allow_low_confidence_fill: bool = True,
) -> Optional[np.ndarray]:
    """Return a shared runtime target mask for late-stage corrections.

    The mask is derived from the packaged authoritative-conditioning contract
    products when they exist. It excludes authoritative-locked cells and keeps
    only eligible water-regime cells.
    """
    eligible = _align_raster_to_template(
        template_path=template_path,
        src_path=eligible_fill_mask_path,
        dtype="uint8",
        nodata_value=0,
    )
    support = _align_raster_to_template(
        template_path=template_path,
        src_path=support_class_path,
        dtype="uint8",
        nodata_value=0,
    )
    regime = _align_raster_to_template(
        template_path=template_path,
        src_path=regime_class_path,
        dtype="uint8",
        nodata_value=0,
    )
    if eligible is None and support is None and regime is None:
        return None

    if eligible is None:
        eligible_mask = np.ones_like(support if support is not None else regime, dtype=bool)
    else:
        eligible_mask = np.asarray(eligible) > 0

    if regime is None:
        regime_mask = np.ones_like(eligible_mask, dtype=bool)
    else:
        regime_mask = np.isin(
            np.asarray(regime, dtype=np.uint8),
            [
                int(RegimeClass.ESTUARY_TRANSITION),
                int(RegimeClass.NEARSHORE_WATER),
                int(RegimeClass.RIVER_CHANNEL),
            ],
        )

    if support is None:
        support_mask = np.ones_like(eligible_mask, dtype=bool)
    else:
        support_arr = np.asarray(support, dtype=np.uint8)
        blocked = {int(SupportClass.AUTHORITATIVE_LOCKED)}
        if not allow_low_confidence_fill:
            blocked.add(int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL))
        support_mask = ~np.isin(support_arr, list(blocked))

    return eligible_mask & regime_mask & support_mask
