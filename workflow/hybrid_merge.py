from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any, Tuple

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt, gaussian_filter


def _read_align(src_path: Path, template: Path, nodata: float, band: int = 1, resamp=Resampling.nearest):
    with rasterio.open(template) as tmpl:
        prof = tmpl.profile.copy()
        prof.update(dtype="float32", nodata=nodata, count=1, compress="deflate")
        prof.pop("blockxsize", None)
        prof.pop("blockysize", None)
        prof["tiled"] = False
        arr = np.full((tmpl.height, tmpl.width), nodata, dtype=np.float32)
        with rasterio.open(src_path) as src:
            reproject(
                source=rasterio.band(src, band),
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


def _masked_gaussian(arr: np.ndarray, valid_mask: np.ndarray, sigma_px: float) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    if not np.any(valid) or sigma_px <= 0.0:
        out = np.full(arr.shape, np.nan, dtype=np.float32)
        out[valid] = arr[valid].astype(np.float32)
        return out
    vals = np.where(valid, arr, 0.0).astype(np.float32)
    w = valid.astype(np.float32)
    num = gaussian_filter(vals, sigma=float(sigma_px), mode="nearest")
    den = gaussian_filter(w, sigma=float(sigma_px), mode="nearest")
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    ok = den > np.float32(1.0e-6)
    out[ok] = (num[ok] / den[ok]).astype(np.float32)
    return out


def _nearest_fill_within_domain(values: np.ndarray, domain_mask: np.ndarray) -> np.ndarray:
    domain = np.asarray(domain_mask, dtype=bool)
    valid = domain & np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return out
    out[valid] = values[valid].astype(np.float32)
    fill = domain & ~valid
    if np.any(fill):
        _, nearest = distance_transform_edt(~valid, return_indices=True)
        rr, cc = np.nonzero(fill)
        out[rr, cc] = values[nearest[0][rr, cc], nearest[1][rr, cc]].astype(np.float32)
    return out


def reconstruct_xs_mainstem_relative(
    bed_xs: Path,
    bed_skel: Path,
    mainstem_mask: Path,
    out_xs: Path,
    template: Path,
    nodata: float,
    receipt_json: Path | None = None,
    backbone_sigma_px: float = 2.0,
    anomaly_sigma_px: float = 1.25,
    bank_core_halfwidth_px: float = 2.5,
) -> Dict[str, Any]:
    """Convert raw XS mainstem raster into backbone-relative mainstem guidance.

    The raw XS raster is treated as lateral shape guidance, not an absolute bed
    surface. A smooth mainstem backbone is derived from the skeleton raster, and
    the XS raster contributes only negative anomalies (deepening) relative to
    that backbone. This suppresses section-to-section transverse sills while
    preserving cross-channel carving.
    """
    xs_a, prof = _read_align(Path(bed_xs), Path(template), nodata, resamp=Resampling.nearest)
    sk_a, _ = _read_align(Path(bed_skel), Path(template), nodata, resamp=Resampling.nearest)
    ms_a, _ = _read_align(Path(mainstem_mask), Path(template), nodata, resamp=Resampling.nearest)

    ms = ms_a > 0.5
    xs_ok = ms & np.isfinite(xs_a) & (xs_a != nodata)
    sk_ok = ms & np.isfinite(sk_a) & (sk_a != nodata)

    reconstructed = np.full(xs_a.shape, nodata, dtype=np.float32)
    if not np.any(ms):
        out_xs.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_xs, "w", **prof) as dst:
            dst.write(reconstructed, 1)
        receipt = {
            "status": "empty_mainstem_mask",
            "raw_xs_raster": str(bed_xs),
            "skeleton_raster": str(bed_skel),
            "mainstem_mask": str(mainstem_mask),
            "out_xs": str(out_xs),
            "template": str(template),
            "nodata": float(nodata),
        }
        if receipt_json is not None:
            receipt_json.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return receipt

    backbone_seed = np.full(xs_a.shape, np.nan, dtype=np.float32)
    backbone_seed[sk_ok] = sk_a[sk_ok].astype(np.float32)
    fill_from_xs = ms & ~np.isfinite(backbone_seed) & xs_ok
    if np.any(fill_from_xs):
        backbone_seed[fill_from_xs] = xs_a[fill_from_xs].astype(np.float32)

    backbone_smoothed = _masked_gaussian(backbone_seed, ms & np.isfinite(backbone_seed), sigma_px=float(backbone_sigma_px))
    backbone = _nearest_fill_within_domain(backbone_smoothed, ms)
    missing_backbone = ms & ~np.isfinite(backbone)
    if np.any(missing_backbone):
        backbone[missing_backbone & np.isfinite(backbone_seed)] = backbone_seed[missing_backbone & np.isfinite(backbone_seed)]

    anomaly = np.full(xs_a.shape, np.nan, dtype=np.float32)
    anomaly[xs_ok & np.isfinite(backbone)] = (xs_a[xs_ok & np.isfinite(backbone)] - backbone[xs_ok & np.isfinite(backbone)]).astype(np.float32)
    # XS should deepen relative to the backbone, not raise it section-by-section.
    anomaly = np.where(np.isfinite(anomaly), np.minimum(anomaly, np.float32(0.0)), np.nan).astype(np.float32)
    anomaly_smoothed = _masked_gaussian(anomaly, xs_ok & np.isfinite(anomaly), sigma_px=float(anomaly_sigma_px))
    anomaly_smoothed = np.where(np.isfinite(anomaly_smoothed), np.minimum(anomaly_smoothed, np.float32(0.0)), np.float32(0.0)).astype(np.float32)

    dist_to_edge_px = distance_transform_edt(ms).astype(np.float32)
    core_weight = np.clip((dist_to_edge_px - np.float32(0.5)) / np.float32(max(float(bank_core_halfwidth_px), 1.0)), 0.0, 1.0).astype(np.float32)
    core_weight[~ms] = 0.0

    reconstruct_take = ms & np.isfinite(backbone)
    reconstructed[reconstruct_take] = backbone[reconstruct_take].astype(np.float32)
    anom_take = reconstruct_take & np.isfinite(anomaly_smoothed)
    if np.any(anom_take):
        reconstructed[anom_take] = (reconstructed[anom_take] + (anomaly_smoothed[anom_take] * core_weight[anom_take])).astype(np.float32)

    # Keep any finite raw XS seed where the reconstructed field could not be formed.
    raw_only = ms & ~np.isfinite(reconstructed) & xs_ok
    if np.any(raw_only):
        reconstructed[raw_only] = xs_a[raw_only].astype(np.float32)

    reconstructed[~ms] = nodata
    out_xs.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_xs, "w", **prof) as dst:
        dst.write(reconstructed.astype(np.float32), 1)

    receipt = {
        "status": "ok",
        "raw_xs_raster": str(bed_xs),
        "skeleton_raster": str(bed_skel),
        "mainstem_mask": str(mainstem_mask),
        "out_xs": str(out_xs),
        "template": str(template),
        "nodata": float(nodata),
        "backbone_sigma_px": float(backbone_sigma_px),
        "anomaly_sigma_px": float(anomaly_sigma_px),
        "bank_core_halfwidth_px": float(bank_core_halfwidth_px),
        "mainstem_pixels": int(ms.sum()),
        "raw_xs_mainstem_pixels": int(xs_ok.sum()),
        "skeleton_mainstem_pixels": int(sk_ok.sum()),
        "reconstructed_mainstem_pixels": int(np.count_nonzero(ms & np.isfinite(reconstructed) & (reconstructed != nodata))),
        "raw_only_fallback_pixels": int(np.count_nonzero(raw_only)),
        "negative_anomaly_pixels": int(np.count_nonzero(xs_ok & np.isfinite(anomaly) & (anomaly < 0.0))),
        "negative_anomaly_min_m": float(np.nanmin(anomaly)) if np.any(np.isfinite(anomaly)) else None,
        "negative_anomaly_p95abs_m": float(np.nanpercentile(np.abs(anomaly[np.isfinite(anomaly)]), 95.0)) if np.any(np.isfinite(anomaly)) else None,
        "reconstruction_rule": "skeleton_backbone_plus_xs_negative_core_anomaly",
    }
    if receipt_json is not None:
        receipt_json.parent.mkdir(parents=True, exist_ok=True)
        receipt_json.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt



def merge_hybrid_river_bed(
    bed_xs: Path,
    bed_skel: Path,
    mainstem_mask: Path,
    river_channel_mask: Path | None,
    out_bed: Path,
    template: Path,
    nodata: float,
    receipt_json: Path | None = None,
    blend_margin_px: int = 8,
    skeleton_stage_support: Path | None = None,
) -> Dict[str, Any]:
    """Merge XS(mainstem) with skeleton(elsewhere) using confidence-weighted blending.

    This function assumes the XS raster already represents a mainstem guidance
    surface suitable for absolute use on the mainstem. Upstream of this stage,
    the workflow now reconstructs the XS raster into a backbone-relative field
    so this merge no longer injects raw section-by-section absolute bed slices.
    """
    xs_a, prof = _read_align(Path(bed_xs), Path(template), nodata, resamp=Resampling.nearest)
    sk_a, _ = _read_align(Path(bed_skel), Path(template), nodata, resamp=Resampling.nearest)
    ms_a, _ = _read_align(Path(mainstem_mask), Path(template), nodata, resamp=Resampling.nearest)
    ch_a = None
    if river_channel_mask is not None and Path(river_channel_mask).exists():
        ch_a, _ = _read_align(Path(river_channel_mask), Path(template), nodata, resamp=Resampling.nearest)
    stage_a = None
    if skeleton_stage_support is not None and Path(skeleton_stage_support).exists():
        stage_a, _ = _read_align(Path(skeleton_stage_support), Path(template), 0.0, resamp=Resampling.nearest)

    ms = ms_a > 0.5
    ch = (ch_a > 0.5) if ch_a is not None else None
    xs_ok = np.isfinite(xs_a) & (xs_a != nodata)
    sk_ok = np.isfinite(sk_a) & (sk_a != nodata)
    sk_stage_ok = (stage_a >= 10.0) if stage_a is not None else np.ones_like(sk_ok, dtype=bool)

    # Phase 4: skeleton is no longer the default absolute bed everywhere.
    # Start with nodata and only let skeleton contribute as a mainstem helper
    # where XS is absent or at narrow edge-blend transitions.
    out = np.full_like(sk_a, np.float32(nodata))
    xs_mainstem = ms & xs_ok
    sk_mainstem = ms & sk_ok & sk_stage_ok
    n_xs_mainstem = int(xs_mainstem.sum())
    n_blended = 0
    n_skeleton_helper_only = 0

    if n_xs_mainstem > 0 and blend_margin_px > 0:
        d_inside_xs = distance_transform_edt(xs_mainstem).astype("float32")
        margin = float(max(1, blend_margin_px))
        w_xs = np.clip(d_inside_xs / margin, 0.0, 1.0).astype("float32")
        both = xs_mainstem & sk_mainstem
        blend_zone = both & (w_xs < 1.0) & (w_xs > 0.0)
        n_blended = int(blend_zone.sum())
        pure_xs = xs_mainstem & (w_xs >= 1.0)
        out[pure_xs] = xs_a[pure_xs]
        if n_blended > 0:
            out[blend_zone] = (
                w_xs[blend_zone] * xs_a[blend_zone]
                + (1.0 - w_xs[blend_zone]) * sk_a[blend_zone]
            ).astype("float32")
        xs_only = xs_mainstem & ~sk_mainstem
        out[xs_only] = xs_a[xs_only]
        # Allow skeleton only where mainstem exists but XS is absent.
        skeleton_helper_only = sk_mainstem & ~xs_ok
        n_skeleton_helper_only = int(skeleton_helper_only.sum())
        out[skeleton_helper_only] = sk_a[skeleton_helper_only]
    elif n_xs_mainstem > 0:
        out[xs_mainstem] = xs_a[xs_mainstem]
        skeleton_helper_only = sk_mainstem & ~xs_ok
        n_skeleton_helper_only = int(skeleton_helper_only.sum())
        out[skeleton_helper_only] = sk_a[skeleton_helper_only]
    else:
        # No XS available: degrade to skeleton only on the mainstem, not everywhere.
        skeleton_helper_only = sk_mainstem
        n_skeleton_helper_only = int(skeleton_helper_only.sum())
        out[skeleton_helper_only] = sk_a[skeleton_helper_only]

    out_bed.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_bed, "w", **prof) as dst:
        dst.write(out.astype(np.float32), 1)

    merged_ok = np.isfinite(out) & (out != nodata)
    overlap = xs_ok & sk_ok & ms
    unresolved_mainstem = ms & ~merged_ok
    unresolved_channel = (ch & ~merged_ok) if ch is not None else np.zeros_like(merged_ok, dtype=bool)
    skeleton_only_mainstem = ms & sk_ok & sk_stage_ok & ~xs_ok

    receipt = {
        "xs_raster": str(bed_xs),
        "skeleton_raster": str(bed_skel),
        "mainstem_mask": str(mainstem_mask),
        "river_channel_mask": str(river_channel_mask) if river_channel_mask is not None else None,
        "skeleton_stage_support": str(skeleton_stage_support) if skeleton_stage_support is not None else None,
        "out_bed": str(out_bed),
        "template": str(template),
        "nodata": float(nodata),
        "blend_margin_px": int(blend_margin_px),
        "mainstem_pixels": int(ms.sum()),
        "channel_pixels": int(ch.sum()) if ch is not None else None,
        "xs_valid_pixels": int(xs_ok.sum()),
        "skeleton_valid_pixels": int(sk_ok.sum()),
        "skeleton_stage_supported_pixels": int(np.count_nonzero(sk_ok & sk_stage_ok)),
        "skeleton_stage_weak_or_missing_pixels": int(np.count_nonzero(sk_ok & ~sk_stage_ok)),
        "xs_wins_mainstem_pixels": int(xs_mainstem.sum()),
        "blended_transition_pixels": n_blended,
        "skeleton_only_mainstem_pixels": int(skeleton_only_mainstem.sum()),
        "skeleton_helper_only_pixels": int(n_skeleton_helper_only),
        "skeleton_outside_mainstem_dropped_pixels": int(np.count_nonzero(sk_ok & ~ms)),
        "overlap_mainstem_pixels": int(overlap.sum()),
        "merged_valid_pixels": int(merged_ok.sum()),
        "unresolved_mainstem_pixels": int(unresolved_mainstem.sum()),
        "unresolved_channel_pixels": int(unresolved_channel.sum()) if ch is not None else None,
        "merge_rule": "xs_mainstem_with_skeleton_mainstem_helper_only",
    }
    if receipt_json is not None:
        receipt_json.parent.mkdir(parents=True, exist_ok=True)
        receipt_json.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt
