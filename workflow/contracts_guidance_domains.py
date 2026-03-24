from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from contract_enforcement import ContractResult, ContractSuiteResult

REQUIRED = {
    "river_guidance_domain_mask",
    "sdb_guidance_domain_mask",
    "river_channel_mask",
    "open_water_mask",
    "mainstem_mask",
    "estuary_clip_mask",
    "estuary_transition_mask",
    "ocean_mask",
    "with_nhd_water_mask",
}


def _read_mask(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
    nod = profile.get("nodata")
    if nod is not None:
        arr = np.where(arr == nod, 0, arr)
    return (arr == 1), profile


def _read_water_mask(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
    nod = profile.get("nodata")
    if nod is not None:
        arr = np.where(arr == nod, 1, arr)
    return (arr == 0), profile






def _read_mask_aligned(path: Path, template_profile: Dict[str, Any]):
    with rasterio.open(path) as ds:
        src = ds.read(1)
        src_profile = ds.profile.copy()
        dst = np.zeros((int(template_profile["height"]), int(template_profile["width"])), dtype=src.dtype)
        src_nodata = ds.nodata
        if src_nodata in (0, 1):
            src_nodata = None
        reproject(
            source=src,
            destination=dst,
            src_transform=ds.transform,
            src_crs=ds.crs,
            src_nodata=src_nodata,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return (dst == 1), src_profile


def _read_water_mask_aligned(path: Path, template_profile: Dict[str, Any]):
    with rasterio.open(path) as ds:
        src = ds.read(1)
        src_profile = ds.profile.copy()
        dst = np.ones((int(template_profile["height"]), int(template_profile["width"])), dtype=src.dtype)
        src_nodata = ds.nodata
        if src_nodata in (0, 1):
            src_nodata = None
        reproject(
            source=src,
            destination=dst,
            src_transform=ds.transform,
            src_crs=ds.crs,
            src_nodata=src_nodata,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            dst_nodata=1,
            resampling=Resampling.nearest,
        )
    return (dst == 0), src_profile

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


def run_guidance_domain_contracts(outputs: Dict[str, str], debug_dir: Path | None = None) -> ContractSuiteResult:
    suite = ContractSuiteResult(stage="guidance_domains")
    missing = [k for k in REQUIRED if not outputs.get(k) or not Path(outputs[k]).exists()]
    suite.add(ContractResult(
        name="required_outputs_exist",
        stage=suite.stage,
        severity="error",
        passed=not missing,
        message="All required guidance-domain artifacts exist." if not missing else f"Missing required artifacts: {', '.join(sorted(missing))}",
        metrics={"missing_count": len(missing)},
    ))
    if missing:
        return suite

    river_guidance, p_rg = _read_mask(Path(outputs["river_guidance_domain_mask"]))
    sdb_guidance, p_sg = _read_mask(Path(outputs["sdb_guidance_domain_mask"]))
    river_channel, p_rc = _read_mask(Path(outputs["river_channel_mask"]))
    open_water, p_ow = _read_mask(Path(outputs["open_water_mask"]))
    mainstem, p_ms = _read_mask(Path(outputs["mainstem_mask"]))

    template_profile = p_rg
    estuary_clip, p_ec = _read_mask_aligned(Path(outputs["estuary_clip_mask"]), template_profile)
    estuary_transition, p_et = _read_mask_aligned(Path(outputs["estuary_transition_mask"]), template_profile)
    ocean_water, p_oc = _read_water_mask_aligned(Path(outputs["ocean_mask"]), template_profile)
    with_nhd_water, p_wh = _read_water_mask_aligned(Path(outputs["with_nhd_water_mask"]), template_profile)

    strict_profiles = [p_sg, p_rc, p_ow, p_ms]
    grid_ok = all(_same_grid(p_rg, p) for p in strict_profiles)
    suite.add(ContractResult(
        name="aligned_grids",
        stage=suite.stage,
        severity="error",
        passed=grid_ok,
        message="Core guidance-domain review rasters align to the same template grid." if grid_ok else "Core guidance-domain review rasters are not on the same grid.",
    ))

    mainstem_subset = bool(np.all(~mainstem | river_channel))
    mainstem_bad = mainstem & ~river_channel
    mainstem_artifacts = {}
    if debug_dir is not None:
        pth = _write_debug_mask(debug_dir / "guidance_mainstem_outside_channel.tif", mainstem_bad, p_ms)
        if pth:
            mainstem_artifacts["bad_pixels_mask"] = pth
    suite.add(ContractResult(
        name="mainstem_subset_of_channel",
        stage=suite.stage,
        severity="error",
        passed=mainstem_subset,
        message="Mainstem mask is a subset of the retained river channel." if mainstem_subset else "Mainstem mask extends outside retained river channel.",
        metrics={"bad_pixels": int(np.count_nonzero(mainstem_bad))},
        artifact_paths=mainstem_artifacts,
    ))

    river_subset = bool(np.all(~river_guidance | river_channel))
    river_bad = river_guidance & ~river_channel
    river_artifacts = {}
    if debug_dir is not None:
        pth = _write_debug_mask(debug_dir / "guidance_river_outside_channel.tif", river_bad, p_rg)
        if pth:
            river_artifacts["bad_pixels_mask"] = pth
    suite.add(ContractResult(
        name="river_guidance_subset_of_channel",
        stage=suite.stage,
        severity="error",
        passed=river_subset,
        message="River guidance domain is a subset of retained river channel." if river_subset else "River guidance domain extends outside retained river channel.",
        metrics={"bad_pixels": int(np.count_nonzero(river_bad))},
        artifact_paths=river_artifacts,
    ))

    # Supporting ocean/estuary allowance rasters may originate on broader source grids.
    # They are reprojected to the review template grid before set operations.
    sdb_allowed = ocean_water | estuary_clip | estuary_transition
    sdb_subset = bool(np.all(~sdb_guidance | sdb_allowed))
    sdb_bad = sdb_guidance & ~sdb_allowed
    sdb_artifacts = {}
    if debug_dir is not None:
        pth = _write_debug_mask(debug_dir / "guidance_sdb_outside_allowed_water.tif", sdb_bad, p_sg)
        if pth:
            sdb_artifacts["bad_pixels_mask"] = pth
    suite.add(ContractResult(
        name="sdb_guidance_subset_of_allowed_water",
        stage=suite.stage,
        severity="error",
        passed=sdb_subset,
        message="SDB guidance domain stays within ocean + estuary allowance." if sdb_subset else "SDB guidance domain includes pixels outside ocean + estuary allowance.",
        metrics={"bad_pixels": int(np.count_nonzero(sdb_bad))},
        artifact_paths=sdb_artifacts,
    ))

    overlap = river_guidance & sdb_guidance
    overlap_artifacts = {}
    if debug_dir is not None:
        pth = _write_debug_mask(debug_dir / "guidance_river_sdb_overlap.tif", overlap, p_rg)
        if pth:
            overlap_artifacts["overlap_mask"] = pth
    suite.add(ContractResult(
        name="river_sdb_domains_disjoint",
        stage=suite.stage,
        severity="error",
        passed=not np.any(overlap),
        message="River and SDB guidance domains are disjoint." if not np.any(overlap) else "River and SDB guidance domains overlap.",
        metrics={"overlap_pixels": int(np.count_nonzero(overlap))},
        artifact_paths=overlap_artifacts,
    ))

    open_subset = bool(np.all(~open_water | with_nhd_water))
    open_bad = open_water & ~with_nhd_water
    open_artifacts = {}
    if debug_dir is not None:
        pth = _write_debug_mask(debug_dir / "guidance_open_water_outside_with_nhd.tif", open_bad, p_ow)
        if pth:
            open_artifacts["bad_pixels_mask"] = pth
    suite.add(ContractResult(
        name="open_water_subset_of_with_nhd_water",
        stage=suite.stage,
        severity="error",
        passed=open_subset,
        message="Open-water mask stays within with-NHD water mask." if open_subset else "Open-water mask includes pixels outside with-NHD water mask.",
        metrics={"bad_pixels": int(np.count_nonzero(open_bad))},
        artifact_paths=open_artifacts,
    ))

    suite.add(ContractResult(
        name="review_masks_nonempty",
        stage=suite.stage,
        severity="warning",
        passed=bool(np.count_nonzero(river_guidance) > 0 and np.count_nonzero(sdb_guidance) > 0),
        message="Both river and SDB guidance review masks have nonzero coverage." if (np.count_nonzero(river_guidance) > 0 and np.count_nonzero(sdb_guidance) > 0) else "One or more guidance review masks are empty.",
        metrics={
            "river_guidance_pixels": int(np.count_nonzero(river_guidance)),
            "sdb_guidance_pixels": int(np.count_nonzero(sdb_guidance)),
        },
    ))
    return suite
