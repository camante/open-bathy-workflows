from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


@dataclass
class RiverMaskStageResult:
    channel_mask_tif: Path
    open_water_mask_tif: Path
    mainstem_mask_tif: Path
    estuary_clip_mask_tif: Path
    receipt_json: Path
    receipt: Dict[str, Any]

    def as_report_outputs(self) -> Dict[str, str]:
        return {
            "river_channel_mask": str(self.channel_mask_tif),
            "open_water_mask": str(self.open_water_mask_tif),
            "mainstem_mask": str(self.mainstem_mask_tif),
            "estuary_clip_mask": str(self.estuary_clip_mask_tif),
            "river_mask_stage_receipt": str(self.receipt_json),
        }


def _read_mask_bool(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
    nod = profile.get("nodata", None)
    if nod is not None:
        arr = np.where(arr == nod, 0, arr)
    return (arr == 1), profile


def _read_water_land_mask(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
    nod = profile.get("nodata", None)
    if nod is not None:
        arr = np.where(arr == nod, 1, arr)
    # fixed semantics for canonical WAFFLES masks: water=0 land=1
    return (arr == 0), profile


def _read_water_land_mask_aligned(path: Path, ref_profile: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a canonical WAFFLES-style water/land mask and align it to a reference grid.

    The river-domain builder warps the ocean mask to the river template before using it.
    Validation must do the same, rather than assuming the staged WAFFLES mask already
    matches the river-template grid on disk.
    """
    with rasterio.open(path) as ds:
        src = ds.read(1)
        src_profile = ds.profile.copy()
        src_nodata = ds.nodata
        dst_nodata = 255
        dst = np.full((int(ref_profile["height"]), int(ref_profile["width"])), dst_nodata, dtype=np.uint8)
        reproject(
            source=src,
            destination=dst,
            src_transform=ds.transform,
            src_crs=ds.crs,
            src_nodata=src_nodata,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=dst_nodata,
            resampling=Resampling.nearest,
        )
    nod = src_profile.get("nodata", None)
    if nod is not None:
        dst = np.where(dst == nod, 1, dst)
    dst = np.where(dst == dst_nodata, 1, dst)
    return (dst == 0), dict(ref_profile)


def _assert_same_grid(a: Dict[str, Any], b: Dict[str, Any], name_a: str, name_b: str) -> None:
    if (a.get("width"), a.get("height"), a.get("crs"), a.get("transform")) != (
        b.get("width"), b.get("height"), b.get("crs"), b.get("transform")
    ):
        raise RuntimeError(f"{name_a} grid does not match {name_b} grid")


def _connected_water_from_ocean(open_water: np.ndarray, ocean_water: np.ndarray | None = None) -> np.ndarray:
    from scipy import ndimage as ndi

    labels, n = ndi.label(open_water, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return np.zeros_like(open_water, dtype=bool)
    if ocean_water is not None:
        seed_labels = set(np.unique(labels[ocean_water]).tolist())
    else:
        seed_labels = set(np.unique(labels[0, :]).tolist() + np.unique(labels[-1, :]).tolist() +
                          np.unique(labels[:, 0]).tolist() + np.unique(labels[:, -1]).tolist())
    seed_labels.discard(0)
    if not seed_labels:
        return np.zeros_like(open_water, dtype=bool)
    keep = np.isin(labels, list(seed_labels))
    return keep & open_water


def _count_components(mask: np.ndarray) -> int:
    from scipy import ndimage as ndi
    _, n = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return int(n)


def validate_river_mask_stage_outputs(
    *,
    channel_mask_tif: Path,
    open_water_mask_tif: Path,
    mainstem_mask_tif: Path,
    estuary_clip_mask_tif: Path,
    ocean_mask_tif: Path | None = None,
) -> Dict[str, Any]:
    channel, p_ch = _read_mask_bool(channel_mask_tif)
    open_water, p_ow = _read_mask_bool(open_water_mask_tif)
    mainstem, p_ms = _read_mask_bool(mainstem_mask_tif)
    estuary_clip, p_est = _read_mask_bool(estuary_clip_mask_tif)
    ocean_water = None
    if ocean_mask_tif is not None:
        ocean_water, p_oc = _read_water_land_mask_aligned(ocean_mask_tif, p_ch)
        _assert_same_grid(p_ch, p_oc, "ocean_mask_aligned", "channel_mask")

    _assert_same_grid(p_ch, p_ow, "open_water_mask", "channel_mask")
    _assert_same_grid(p_ch, p_ms, "mainstem_mask", "channel_mask")
    _assert_same_grid(p_ch, p_est, "estuary_clip_mask", "channel_mask")

    if int(np.count_nonzero(channel)) <= 0:
        raise RuntimeError("river_channel_mask has zero channel pixels")
    if int(np.count_nonzero(mainstem)) <= 0:
        raise RuntimeError("mainstem_mask has zero pixels")
    if not np.all(mainstem <= channel):
        n_bad = int(np.count_nonzero(mainstem & ~channel))
        raise RuntimeError(f"mainstem_mask is not a subset of river_channel_mask (bad_pixels={n_bad})")
    if np.any(estuary_clip & channel):
        n_bad = int(np.count_nonzero(estuary_clip & channel))
        raise RuntimeError(f"estuary_clip_mask overlaps final river_channel_mask (overlap_pixels={n_bad})")
    if np.any(estuary_clip & open_water):
        n_bad = int(np.count_nonzero(estuary_clip & open_water))
        raise RuntimeError(f"estuary_clip_mask overlaps open_water_mask (overlap_pixels={n_bad})")
    if np.any(open_water & channel):
        n_bad = int(np.count_nonzero(open_water & channel))
        raise RuntimeError(f"open_water_mask overlaps river_channel_mask (overlap_pixels={n_bad})")

    full_water = open_water | channel
    ocean_connected_full = _connected_water_from_ocean(full_water, ocean_water=ocean_water)
    open_water_outside_ocean_connected_full = int(np.count_nonzero(open_water & ~ocean_connected_full))

    receipt = {
        "channel_pixels": int(np.count_nonzero(channel)),
        "open_water_pixels": int(np.count_nonzero(open_water)),
        "mainstem_pixels": int(np.count_nonzero(mainstem)),
        "estuary_pixels": int(np.count_nonzero(estuary_clip)),
        "channel_components": _count_components(channel),
        "open_water_components": _count_components(open_water),
        "mainstem_components": _count_components(mainstem),
        "estuary_components": _count_components(estuary_clip),
        "mainstem_subset_of_channel": True,
        "estuary_excluded_from_final_channel": True,
        "estuary_disjoint_from_open_water": True,
        "open_water_ocean_connected_only": open_water_outside_ocean_connected_full == 0,
        "open_water_outside_ocean_connected_full_pixels": int(open_water_outside_ocean_connected_full),
    }
    return receipt



def _apply_estuary_clip_to_mask(mask_path: Path, estuary_clip_mask_path: Path, *, mask_name: str) -> Dict[str, int]:
    """Remove estuary-clipped pixels from an exported mask on disk.

    This keeps staged masks consistent when the estuary clip already exists on disk
    and the clip function is therefore not re-run against the current channel/open-water
    exports inside this stage.
    """
    if not mask_path.exists() or not estuary_clip_mask_path.exists():
        return {"pixels_before": 0, "pixels_after": 0, "pixels_removed": 0}
    mask, mask_profile = _read_mask_bool(mask_path)
    estuary_clip, est_profile = _read_mask_bool(estuary_clip_mask_path)
    _assert_same_grid(mask_profile, est_profile, mask_name, "estuary_clip_mask")
    before = int(np.count_nonzero(mask))
    removed = int(np.count_nonzero(mask & estuary_clip))
    if removed <= 0:
        return {"pixels_before": before, "pixels_after": before, "pixels_removed": 0}
    updated = (mask & ~estuary_clip).astype(np.uint8)
    profile = mask_profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.update(dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(updated, 1)
    return {
        "pixels_before": before,
        "pixels_after": int(np.count_nonzero(updated)),
        "pixels_removed": removed,
    }

def _sync_mainstem_to_final_channel(
    *,
    mainstem_mask_tif: Path,
    channel_mask_tif: Path,
    estuary_clip_mask_tif: Path | None = None,
) -> Dict[str, int]:
    """Force the exported mainstem mask to match the current retained channel domain.

    This is intentionally used both:
    1) immediately after river_domain_mask export, to repair any broad corridor-style
       mainstem output from stale or mismatched domain-mask code paths, and
    2) after estuary clipping, to remove estuary-trimmed and border-eroded channel pixels.

    The core invariant is simple: the on-disk mainstem mask consumed by downstream hybrid
    routing must always be a subset of the on-disk channel mask.
    """
    mainstem, ms_profile = _read_mask_bool(mainstem_mask_tif)
    channel, ch_profile = _read_mask_bool(channel_mask_tif)
    _assert_same_grid(ms_profile, ch_profile, "mainstem_mask", "channel_mask")

    estuary_clip = np.zeros_like(channel, dtype=bool)
    if estuary_clip_mask_tif is not None:
        estuary_clip, est_profile = _read_mask_bool(estuary_clip_mask_tif)
        _assert_same_grid(ms_profile, est_profile, "mainstem_mask", "estuary_clip_mask")

    before = int(np.count_nonzero(mainstem))
    removed_by_estuary = int(np.count_nonzero(mainstem & estuary_clip))
    removed_total = int(np.count_nonzero(mainstem & ~channel))
    if removed_total <= 0:
        return {
            "mainstem_pixels_before": before,
            "mainstem_pixels_after": before,
            "mainstem_pixels_removed_total": 0,
            "mainstem_pixels_removed_by_estuary": 0,
            "mainstem_pixels_removed_by_non_estuary_channel_trim": 0,
        }

    updated = (mainstem & channel).astype("uint8")
    profile = ms_profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.update(dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(mainstem_mask_tif, "w", **profile) as dst:
        dst.write(updated, 1)

    after_mask, _ = _read_mask_bool(mainstem_mask_tif)
    after = int(np.count_nonzero(after_mask))
    remaining_bad = int(np.count_nonzero(after_mask & ~channel))
    if remaining_bad > 0:
        raise RuntimeError(f"mainstem sync failed to constrain output to channel (remaining_bad_pixels={remaining_bad})")

    return {
        "mainstem_pixels_before": before,
        "mainstem_pixels_after": after,
        "mainstem_pixels_removed_total": removed_total,
        "mainstem_pixels_removed_by_estuary": removed_by_estuary,
        "mainstem_pixels_removed_by_non_estuary_channel_trim": max(0, removed_total - removed_by_estuary),
    }


def run_river_mask_stage(
    *,
    work_dir: Path,
    report: Dict[str, Any],
    logger: Any,
    build_domain_masks_fn: Callable[..., Tuple[Path, Path, Path]],
    estuary_clip_fn: Callable[..., Tuple[int, Optional[Path]]],
    cfg: Any,
) -> RiverMaskStageResult:
    logger.info("[RIVER][MASK] Building and validating river masks...")
    channel_mask_tif, open_water_mask_tif, mainstem_mask_tif = build_domain_masks_fn(work_dir, strict=True)
    if channel_mask_tif is None or not Path(channel_mask_tif).exists():
        raise RuntimeError("River mask stage did not produce river_channel_mask.tif")
    if open_water_mask_tif is None or not Path(open_water_mask_tif).exists():
        raise RuntimeError("River mask stage did not produce open_water_mask.tif")
    if mainstem_mask_tif is None or not Path(mainstem_mask_tif).exists():
        raise RuntimeError("River mask stage did not produce mainstem_mask.tif")

    preclip_mainstem_receipt = _sync_mainstem_to_final_channel(
        mainstem_mask_tif=Path(mainstem_mask_tif),
        channel_mask_tif=Path(channel_mask_tif),
        estuary_clip_mask_tif=None,
    )
    if preclip_mainstem_receipt["mainstem_pixels_removed_total"] > 0:
        logger.info(
            "[RIVER][MASK] Repaired exported mainstem mask to current channel before estuary clip: %d → %d pixels (%d removed)",
            preclip_mainstem_receipt["mainstem_pixels_before"],
            preclip_mainstem_receipt["mainstem_pixels_after"],
            preclip_mainstem_receipt["mainstem_pixels_removed_total"],
        )

    precomputed_estuary = work_dir / "estuary_clip_mask.tif"
    cfg_estuary = Path(getattr(cfg, "estuary_clip_mask", "") or "")
    cfg_transition = Path(getattr(cfg, "estuary_transition_mask", "") or "")
    if (not precomputed_estuary.exists()) and cfg_estuary.exists():
        precomputed_estuary.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(cfg_estuary, precomputed_estuary)
        if cfg_transition.exists():
            shutil.copy2(cfg_transition, work_dir / "estuary_transition_mask.tif")
    if precomputed_estuary.exists():
        estuary_clip_mask_tif = precomputed_estuary
        logger.info("[RIVER][MASK] Reusing precomputed estuary clip from guidance-domain stage: %s", estuary_clip_mask_tif)
    else:
        _, estuary_clip_mask_tif = estuary_clip_fn(
            channel_mask_tif,
            cfg,
            ocean_mask_path=cfg.waffles_ocean_mask,
            report=report,
        )
        if estuary_clip_mask_tif is None or not Path(estuary_clip_mask_tif).exists():
            raise RuntimeError("River mask stage did not produce estuary_clip_mask.tif")

    channel_clip_receipt = _apply_estuary_clip_to_mask(
        Path(channel_mask_tif),
        Path(estuary_clip_mask_tif),
        mask_name="river_channel_mask",
    )
    if channel_clip_receipt["pixels_removed"] > 0:
        logger.info(
            "[ESTUARY-CLIP] Channel mask clipped by estuary: %d → %d pixels (%d removed)",
            channel_clip_receipt["pixels_before"],
            channel_clip_receipt["pixels_after"],
            channel_clip_receipt["pixels_removed"],
        )

    open_water_clip_receipt = _apply_estuary_clip_to_mask(
        Path(open_water_mask_tif),
        Path(estuary_clip_mask_tif),
        mask_name="open_water_mask",
    )
    if open_water_clip_receipt["pixels_removed"] > 0:
        logger.info(
            "[ESTUARY-CLIP] Open-water mask clipped by estuary: %d → %d pixels (%d removed)",
            open_water_clip_receipt["pixels_before"],
            open_water_clip_receipt["pixels_after"],
            open_water_clip_receipt["pixels_removed"],
        )

    mainstem_clip_receipt = _sync_mainstem_to_final_channel(
        mainstem_mask_tif=Path(mainstem_mask_tif),
        channel_mask_tif=Path(channel_mask_tif),
        estuary_clip_mask_tif=Path(estuary_clip_mask_tif),
    )
    logger.info(
        "[ESTUARY-CLIP] Mainstem mask synced to final channel: %d → %d pixels (%d removed: %d estuary, %d other channel trim)",
        mainstem_clip_receipt["mainstem_pixels_before"],
        mainstem_clip_receipt["mainstem_pixels_after"],
        mainstem_clip_receipt["mainstem_pixels_removed_total"],
        mainstem_clip_receipt["mainstem_pixels_removed_by_estuary"],
        mainstem_clip_receipt["mainstem_pixels_removed_by_non_estuary_channel_trim"],
    )

    receipt = validate_river_mask_stage_outputs(
        channel_mask_tif=Path(channel_mask_tif),
        open_water_mask_tif=Path(open_water_mask_tif),
        mainstem_mask_tif=Path(mainstem_mask_tif),
        estuary_clip_mask_tif=Path(estuary_clip_mask_tif),
        ocean_mask_tif=Path(cfg.waffles_ocean_mask) if getattr(cfg, "waffles_ocean_mask", None) else None,
    )
    receipt.update({
        "preclip_mainstem_pixels_before": preclip_mainstem_receipt["mainstem_pixels_before"],
        "preclip_mainstem_pixels_after": preclip_mainstem_receipt["mainstem_pixels_after"],
        "preclip_mainstem_pixels_removed_total": preclip_mainstem_receipt["mainstem_pixels_removed_total"],
    })
    receipt.update(mainstem_clip_receipt)
    receipt.update({
        "channel_pixels_before_estuary_clip": channel_clip_receipt["pixels_before"],
        "channel_pixels_after_estuary_clip": channel_clip_receipt["pixels_after"],
        "channel_pixels_removed_by_estuary_clip": channel_clip_receipt["pixels_removed"],
        "open_water_pixels_before_estuary_clip": open_water_clip_receipt["pixels_before"],
        "open_water_pixels_after_estuary_clip": open_water_clip_receipt["pixels_after"],
        "open_water_pixels_removed_by_estuary_clip": open_water_clip_receipt["pixels_removed"],
    })
    receipt_json = work_dir / "river_mask_stage_receipt.json"
    receipt_json.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    report.setdefault("river", {}).setdefault("mask_stage", {}).update(receipt)
    report.setdefault("river", {}).setdefault("outputs", {}).update({
        "river_mask_stage_receipt": str(receipt_json),
    })
    return RiverMaskStageResult(
        channel_mask_tif=Path(channel_mask_tif),
        open_water_mask_tif=Path(open_water_mask_tif),
        mainstem_mask_tif=Path(mainstem_mask_tif),
        estuary_clip_mask_tif=Path(estuary_clip_mask_tif),
        receipt_json=receipt_json,
        receipt=receipt,
    )
