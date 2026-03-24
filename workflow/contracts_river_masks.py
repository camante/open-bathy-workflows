from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio

from contract_enforcement import ContractResult, ContractSuiteResult


def _read_mask(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
    nod = profile.get("nodata")
    if nod is not None:
        arr = np.where(arr == nod, 0, arr)
    return (arr == 1), profile




def _write_debug_mask(path: Path, mask: np.ndarray, profile: Dict[str, Any]) -> str | None:
    if not np.any(mask):
        return None
    out_profile = profile.copy()
    out_profile.pop("blockxsize", None)
    out_profile.pop("blockysize", None)
    out_profile.pop("tiled", None)
    out_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as ds:
        ds.write(mask.astype("uint8"), 1)
    return str(path)

def _same_grid(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (a.get("width"), a.get("height"), str(a.get("crs")), a.get("transform")) == (
        b.get("width"), b.get("height"), str(b.get("crs")), b.get("transform")
    )


def run_river_mask_contracts(*, channel_mask_tif: Path, open_water_mask_tif: Path, mainstem_mask_tif: Path, estuary_clip_mask_tif: Path, debug_dir: Path | None = None) -> ContractSuiteResult:
    suite = ContractSuiteResult(stage="river_masks")
    paths = {
        "channel": Path(channel_mask_tif),
        "open_water": Path(open_water_mask_tif),
        "mainstem": Path(mainstem_mask_tif),
        "estuary": Path(estuary_clip_mask_tif),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    suite.add(ContractResult(
        name="required_outputs_exist",
        stage=suite.stage,
        severity="error",
        passed=not missing,
        message="All river mask stage outputs exist." if not missing else f"Missing river mask artifacts: {', '.join(missing)}",
    ))
    if missing:
        return suite

    channel, p_ch = _read_mask(paths["channel"])
    open_water, p_ow = _read_mask(paths["open_water"])
    mainstem, p_ms = _read_mask(paths["mainstem"])
    estuary, p_est = _read_mask(paths["estuary"])

    grid_ok = _same_grid(p_ch, p_ow) and _same_grid(p_ch, p_ms) and _same_grid(p_ch, p_est)
    suite.add(ContractResult(
        name="aligned_grids",
        stage=suite.stage,
        severity="error",
        passed=grid_ok,
        message="River mask stage outputs align to one grid." if grid_ok else "River mask stage outputs are not aligned to one grid.",
    ))

    suite.add(ContractResult(
        name="channel_nonempty",
        stage=suite.stage,
        severity="error",
        passed=bool(np.count_nonzero(channel) > 0),
        message="River channel mask has nonzero coverage." if np.count_nonzero(channel) > 0 else "River channel mask is empty.",
        metrics={"channel_pixels": int(np.count_nonzero(channel))},
    ))
    suite.add(ContractResult(
        name="mainstem_nonempty",
        stage=suite.stage,
        severity="error",
        passed=bool(np.count_nonzero(mainstem) > 0),
        message="Mainstem mask has nonzero coverage." if np.count_nonzero(mainstem) > 0 else "Mainstem mask is empty.",
        metrics={"mainstem_pixels": int(np.count_nonzero(mainstem))},
    ))
    mainstem_bad = mainstem & ~channel
    mainstem_artifacts = {}
    if debug_dir is not None:
        pth = _write_debug_mask(debug_dir / "river_masks_mainstem_outside_channel.tif", mainstem_bad, p_ms)
        if pth:
            mainstem_artifacts["bad_pixels_mask"] = pth
    suite.add(ContractResult(
        name="mainstem_subset_of_channel",
        stage=suite.stage,
        severity="error",
        passed=bool(np.all(~mainstem | channel)),
        message="Mainstem mask is a subset of river channel." if np.all(~mainstem | channel) else "Mainstem mask extends outside river channel.",
        metrics={"bad_pixels": int(np.count_nonzero(mainstem_bad))},
        artifact_paths=mainstem_artifacts,
    ))
    ow_overlap = open_water & channel
    ow_artifacts = {}
    if debug_dir is not None:
        pth = _write_debug_mask(debug_dir / "river_masks_open_water_channel_overlap.tif", ow_overlap, p_ch)
        if pth:
            ow_artifacts["overlap_mask"] = pth
    suite.add(ContractResult(
        name="open_water_disjoint_from_channel",
        stage=suite.stage,
        severity="error",
        passed=not np.any(ow_overlap),
        message="Open water is disjoint from river channel." if not np.any(ow_overlap) else "Open water overlaps river channel.",
        metrics={"overlap_pixels": int(np.count_nonzero(ow_overlap))},
        artifact_paths=ow_artifacts,
    ))
    est_overlap = estuary & channel
    est_artifacts = {}
    if debug_dir is not None:
        pth = _write_debug_mask(debug_dir / "river_masks_estuary_channel_overlap.tif", est_overlap, p_ch)
        if pth:
            est_artifacts["overlap_mask"] = pth
    suite.add(ContractResult(
        name="estuary_disjoint_from_channel",
        stage=suite.stage,
        severity="error",
        passed=not np.any(est_overlap),
        message="Estuary clip is excluded from final river channel." if not np.any(est_overlap) else "Estuary clip overlaps final river channel.",
        metrics={"overlap_pixels": int(np.count_nonzero(est_overlap))},
        artifact_paths=est_artifacts,
    ))
    suite.add(ContractResult(
        name="estuary_artifact_present_even_if_empty",
        stage=suite.stage,
        severity="warning",
        passed=True,
        message="Estuary clip artifact exists; zero-valued estuary masks are allowed.",
        metrics={"estuary_pixels": int(np.count_nonzero(estuary))},
    ))
    return suite
