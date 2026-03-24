from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def _read_align(src_path: Path, template: Path, nodata: float, band: int = 1, resamp=Resampling.nearest):
    with rasterio.open(template) as tmpl:
        prof = tmpl.profile.copy()
        prof.update(dtype="float32", nodata=nodata, count=1, compress="deflate")
        prof.pop("blockxsize", None)
        prof.pop("blockysize", None)
        prof["tiled"] = False
        arr = np.full((tmpl.height, tmpl.width), nodata, dtype=np.float32)
        with rasterio.open(src_path) as src:
            src_arr = src.read(band)
            reproject(
                source=src_arr,
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tmpl.transform,
                dst_crs=tmpl.crs,
                resampling=resamp,
                src_nodata=src.nodata,
                dst_nodata=nodata,
            )
    return arr, prof


def merge_hybrid_river_bed(
    bed_xs: Path,
    bed_skel: Path,
    mainstem_mask: Path,
    river_channel_mask: Path | None,
    out_bed: Path,
    template: Path,
    nodata: float,
    receipt_json: Path | None = None,
) -> Dict[str, Any]:
    """Merge XS(mainstem) with skeleton(elsewhere) using explicit precedence.

    XS wins only where mainstem mask is true and XS is finite. Skeleton is used elsewhere.
    Optionally report unresolved pixels inside the retained river channel domain.
    """
    xs_a, prof = _read_align(Path(bed_xs), Path(template), nodata, resamp=Resampling.nearest)
    sk_a, _ = _read_align(Path(bed_skel), Path(template), nodata, resamp=Resampling.nearest)
    ms_a, _ = _read_align(Path(mainstem_mask), Path(template), nodata, resamp=Resampling.nearest)
    ch_a = None
    if river_channel_mask is not None and Path(river_channel_mask).exists():
        ch_a, _ = _read_align(Path(river_channel_mask), Path(template), nodata, resamp=Resampling.nearest)

    ms = ms_a > 0.5
    ch = (ch_a > 0.5) if ch_a is not None else None
    xs_ok = np.isfinite(xs_a) & (xs_a != nodata)
    sk_ok = np.isfinite(sk_a) & (sk_a != nodata)

    out = sk_a.copy()
    xs_take = ms & xs_ok
    out[xs_take] = xs_a[xs_take]

    out_bed.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_bed, 'w', **prof) as dst:
        dst.write(out.astype(np.float32), 1)

    merged_ok = np.isfinite(out) & (out != nodata)
    overlap = xs_ok & sk_ok & ms
    unresolved_mainstem = ms & ~merged_ok
    unresolved_channel = (ch & ~merged_ok) if ch is not None else np.zeros_like(merged_ok, dtype=bool)
    skeleton_only_mainstem = ms & sk_ok & ~xs_ok

    receipt = {
        'xs_raster': str(bed_xs),
        'skeleton_raster': str(bed_skel),
        'mainstem_mask': str(mainstem_mask),
        'river_channel_mask': str(river_channel_mask) if river_channel_mask is not None else None,
        'out_bed': str(out_bed),
        'template': str(template),
        'nodata': float(nodata),
        'mainstem_pixels': int(ms.sum()),
        'channel_pixels': int(ch.sum()) if ch is not None else None,
        'xs_valid_pixels': int(xs_ok.sum()),
        'skeleton_valid_pixels': int(sk_ok.sum()),
        'xs_wins_mainstem_pixels': int(xs_take.sum()),
        'skeleton_only_mainstem_pixels': int(skeleton_only_mainstem.sum()),
        'overlap_mainstem_pixels': int(overlap.sum()),
        'merged_valid_pixels': int(merged_ok.sum()),
        'unresolved_mainstem_pixels': int(unresolved_mainstem.sum()),
        'unresolved_channel_pixels': int(unresolved_channel.sum()) if ch is not None else None,
        'merge_rule': 'xs_dominates_mainstem_when_finite_else_skeleton',
    }
    if receipt_json is not None:
        receipt_json.parent.mkdir(parents=True, exist_ok=True)
        receipt_json.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding='utf-8')
    return receipt
