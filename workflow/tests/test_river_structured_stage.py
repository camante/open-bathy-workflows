from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from river_structured_stage import run_structured_river_stage


def _write_mask(path: Path, value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": 1,
        "width": 1,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32619",
        "transform": from_origin(0.0, 1.0, 1.0, 1.0),
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.array([[value]], dtype=np.uint8), 1)


def test_run_structured_river_stage_records_publish_mode(tmp_path: Path):
    river_dir = tmp_path / 'river'
    work_dir = tmp_path / 'work'
    cache_dir = tmp_path / 'cache'
    src_dem = tmp_path / 'river_dem.tif'
    src_dem.write_text('dem', encoding='utf-8')
    channel_mask = tmp_path / 'channel_mask.tif'
    _write_mask(channel_mask, value=1)

    class Cfg:
        river_dem = str(src_dem)
        strict = False
        out_dir = str(tmp_path / 'out')
        river_allow_absolute_bed_fallback = False
        _write_river_guidance_artifacts = lambda *args, **kwargs: None

    report = {'river': {'outputs': {}}}

    def build_domain_masks_fn(_work_dir, strict=False):
        return channel_mask, None, None

    def estuary_clip_fn(*args, **kwargs):
        return None

    def build_structured_helper_rasters_fn(**kwargs):
        bed_tif = kwargs['bed_tif']
        depth_tif = kwargs['depth_tif']
        Path(bed_tif).write_text('bed', encoding='utf-8')
        Path(depth_tif).write_text('depth', encoding='utf-8')
        return {'status': 'ok'}

    def write_guidance_artifacts_fn(**kwargs):
        return None

    def build_constraint_summary_fn(cfg, report):
        report.setdefault('river', {}).setdefault('summary', {})['ok'] = True

    def apply_depth_metadata_fn(*args, **kwargs):
        return None

    def apply_elevation_metadata_fn(*args, **kwargs):
        return None

    out = run_structured_river_stage(
        cfg=Cfg(),
        report=report,
        river_dir=river_dir,
        work_dir=work_dir,
        cache_dir=cache_dir,
        logger=__import__('logging').getLogger('test'),
        build_domain_masks_fn=build_domain_masks_fn,
        estuary_clip_fn=estuary_clip_fn,
        build_structured_helper_rasters_fn=build_structured_helper_rasters_fn,
        write_guidance_artifacts_fn=write_guidance_artifacts_fn,
        build_constraint_summary_fn=build_constraint_summary_fn,
        apply_depth_metadata_fn=apply_depth_metadata_fn,
        apply_elevation_metadata_fn=apply_elevation_metadata_fn,
    )

    assert out.exists()
    publish_mode = report['river']['structured_stage']['publish_mode']
    assert publish_mode['river_depth_terrain_patch.tif'] in {'symlink', 'copy'}
    assert publish_mode['river_bottom_navd88_patch.tif'] in {'symlink', 'copy'}


def test_run_structured_river_stage_reuses_precomputed_estuary_clip(tmp_path: Path):
    river_dir = tmp_path / "river"
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    src_dem = tmp_path / "river_dem.tif"
    src_dem.write_text("dem", encoding="utf-8")
    channel_mask = tmp_path / "channel_mask.tif"
    _write_mask(channel_mask, value=1)

    class Cfg:
        river_dem = str(src_dem)
        strict = False
        out_dir = str(tmp_path / "out")
        river_allow_absolute_bed_fallback = False
        waffles_ocean_mask = str(tmp_path / "ocean_only_mask.tif")
        _write_river_guidance_artifacts = lambda *args, **kwargs: None

    report = {"river": {"outputs": {"waffles_water_mask": str(tmp_path / "with_nhd_mask.tif")}}}
    called = {"estuary": 0}

    def build_domain_masks_fn(_work_dir, strict=False):
        _work_dir.mkdir(parents=True, exist_ok=True)
        _write_mask(_work_dir / "estuary_clip_mask.tif", value=0)
        _write_mask(_work_dir / "estuary_transition_mask.tif", value=0)
        return channel_mask, None, None

    def estuary_clip_fn(*args, **kwargs):
        called["estuary"] += 1
        raise AssertionError("structured stage should reuse precomputed estuary clip")

    def build_structured_helper_rasters_fn(**kwargs):
        Path(kwargs["bed_tif"]).write_text("bed", encoding="utf-8")
        Path(kwargs["depth_tif"]).write_text("depth", encoding="utf-8")
        return {"status": "ok"}

    def write_guidance_artifacts_fn(**kwargs):
        return None

    def build_constraint_summary_fn(cfg, report):
        return None

    def apply_depth_metadata_fn(*args, **kwargs):
        return None

    def apply_elevation_metadata_fn(*args, **kwargs):
        return None

    run_structured_river_stage(
        cfg=Cfg(),
        report=report,
        river_dir=river_dir,
        work_dir=work_dir,
        cache_dir=cache_dir,
        logger=__import__("logging").getLogger("test"),
        build_domain_masks_fn=build_domain_masks_fn,
        estuary_clip_fn=estuary_clip_fn,
        build_structured_helper_rasters_fn=build_structured_helper_rasters_fn,
        write_guidance_artifacts_fn=write_guidance_artifacts_fn,
        build_constraint_summary_fn=build_constraint_summary_fn,
        apply_depth_metadata_fn=apply_depth_metadata_fn,
        apply_elevation_metadata_fn=apply_elevation_metadata_fn,
    )

    assert called["estuary"] == 0
    exec_receipts = report["river"]["execution_receipts"]
    assert exec_receipts["structured_estuary_clip_source"] == "precomputed"
    assert report["river"]["outputs"]["estuary_clip_mask"].endswith("estuary_clip_mask.tif")
    assert report["river"]["outputs"]["estuary_transition"].endswith("estuary_transition_mask.tif")


def test_run_structured_river_stage_uses_ocean_only_mask_for_estuary_clip(tmp_path: Path):
    river_dir = tmp_path / "river"
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    src_dem = tmp_path / "river_dem.tif"
    src_dem.write_text("dem", encoding="utf-8")
    channel_mask = tmp_path / "channel_mask.tif"
    _write_mask(channel_mask, value=1)
    ocean_only_mask = tmp_path / "ocean_only_mask.tif"
    ocean_only_mask.write_text("ocean", encoding="utf-8")
    with_nhd_mask = tmp_path / "with_nhd_mask.tif"
    with_nhd_mask.write_text("with_nhd", encoding="utf-8")

    class Cfg:
        river_dem = str(src_dem)
        strict = False
        out_dir = str(tmp_path / "out")
        river_allow_absolute_bed_fallback = False
        waffles_ocean_mask = str(ocean_only_mask)
        _write_river_guidance_artifacts = lambda *args, **kwargs: None

    report = {"river": {"outputs": {"waffles_water_mask": str(with_nhd_mask), "waffles_ocean_mask": str(ocean_only_mask)}}}
    estuary_call = {}

    def build_domain_masks_fn(_work_dir, strict=False):
        _work_dir.mkdir(parents=True, exist_ok=True)
        return channel_mask, None, None

    def estuary_clip_fn(channel_mask_arg, cfg_arg, *, ocean_mask_path, report):
        estuary_call["channel_mask"] = channel_mask_arg
        estuary_call["ocean_mask_path"] = ocean_mask_path
        return 0, work_dir / "estuary_clip_mask.tif"

    def build_structured_helper_rasters_fn(**kwargs):
        Path(kwargs["bed_tif"]).write_text("bed", encoding="utf-8")
        Path(kwargs["depth_tif"]).write_text("depth", encoding="utf-8")
        return {"status": "ok"}

    def write_guidance_artifacts_fn(**kwargs):
        return None

    def build_constraint_summary_fn(cfg, report):
        return None

    def apply_depth_metadata_fn(*args, **kwargs):
        return None

    def apply_elevation_metadata_fn(*args, **kwargs):
        return None

    run_structured_river_stage(
        cfg=Cfg(),
        report=report,
        river_dir=river_dir,
        work_dir=work_dir,
        cache_dir=cache_dir,
        logger=__import__("logging").getLogger("test"),
        build_domain_masks_fn=build_domain_masks_fn,
        estuary_clip_fn=estuary_clip_fn,
        build_structured_helper_rasters_fn=build_structured_helper_rasters_fn,
        write_guidance_artifacts_fn=write_guidance_artifacts_fn,
        build_constraint_summary_fn=build_constraint_summary_fn,
        apply_depth_metadata_fn=apply_depth_metadata_fn,
        apply_elevation_metadata_fn=apply_elevation_metadata_fn,
    )

    assert estuary_call["channel_mask"] == channel_mask
    assert estuary_call["ocean_mask_path"] == ocean_only_mask
    assert report["river"]["execution_receipts"]["structured_estuary_clip_source"] == "recomputed"
