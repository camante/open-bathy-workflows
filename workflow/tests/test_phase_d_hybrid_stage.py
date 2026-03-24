from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from river_hybrid_stage import run_hybrid_river_stage
from xs_contracts import validate_xs_artifacts


def _write_raster(path: Path, arr: np.ndarray, nodata: float = -9999.0):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32619",
        "transform": from_origin(0, arr.shape[0], 1, 1),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr.astype("float32"), 1)


def test_phase_d_hybrid_stage_smoke(tmp_path: Path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    script_dir = tmp_path
    dem = tmp_path / "dem.tif"
    _write_raster(dem, np.zeros((20, 20), dtype=np.float32))
    network_gpkg = tmp_path / "river_network.gpkg"
    gdf = gpd.GeoDataFrame({"streamorde": [5], "geometry": [LineString([(0, 10), (19, 10)])]}, crs="EPSG:32619")
    gdf.to_file(network_gpkg, layer="rivers_clip", driver="GPKG")

    channel = work_dir / "river_channel_mask.tif"
    open_water = work_dir / "open_water_mask.tif"
    mainstem = work_dir / "mainstem_mask.tif"
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[2:18, 2:18] = 1
    _write_raster(channel, mask, nodata=255)
    _write_raster(open_water, np.zeros((20, 20), dtype=np.float32), nodata=255)
    _write_raster(mainstem, mask, nodata=255)

    soundings_subset = work_dir / "river_soundings_subset.parquet"

    def build_domain_masks_fn(_work_dir, strict=False):
        return channel, open_water, mainstem

    estuary = work_dir / "estuary_clip_mask.tif"
    _write_raster(estuary, np.where(mask == 1, 0.0, 1.0).astype(np.float32), nodata=255)

    def estuary_clip_fn(*args, **kwargs):
        return 0, estuary

    def append_river_soundings_args_fn(cmd, cfg, include_calib_args=False, include_mode_args=False):
        return None

    def build_river_skeleton_command_fn(cfg, **kwargs):
        return ["python", "river_skeleton_bathy.py"]

    def authoritative_passthrough_args_fn(cfg, for_river=False):
        return []

    def record_authoritative_child_passthrough_fn(report, stage, cmd):
        report.setdefault("river", {}).setdefault("authoritative_passthrough", {})[stage] = True

    def run_command_fn(cmd, cwd=None, prefix=""):
        tool = cmd[1]
        if tool == "xs_builder.py":
            xs_gpkg = work_dir / "cross_sections_mainstem.gpkg"
            xs_lines = gpd.GeoDataFrame(
                {
                    "xs_id": [1],
                    "bank_left_dist_m": [1.0],
                    "bank_right_dist_m": [1.0],
                    "bank_left_z_m": [0.0],
                    "bank_right_z_m": [0.0],
                    "s_center_m": [1.0],
                    "geometry": [LineString([(1, 1), (3, 1)])],
                },
                crs="EPSG:32619",
            )
            xs_points = gpd.GeoDataFrame(
                {
                    "xs_id": [1, 1],
                    "s_m": [0.0, 2.0],
                    "z_m": [0.0, 0.0],
                    "is_bank_left": [1, 0],
                    "is_bank_right": [0, 1],
                    "geometry": [Point(1, 1), Point(3, 1)],
                },
                crs="EPSG:32619",
            )
            xs_lines.to_file(xs_gpkg, layer="xs_lines", driver="GPKG")
            xs_points.to_file(xs_gpkg, layer="xs_points", driver="GPKG")
            return 0, "", ""
        if tool == "xs_infer_bathy_raster.py" and "--only-write-soundings-subset" in cmd:
            soundings_subset.write_text("placeholder")
            return 0, "", ""
        if tool == "xs_infer_bathy_raster.py":
            _write_raster(work_dir / "river_bed_elev_xs_mainstem.tif", np.where(mask == 1, -2.0, -9999.0))
            gpd.GeoDataFrame({"geometry": [Point(1, 1)]}, crs="EPSG:32619").to_file(work_dir / "river_bathy_xs_mainstem.gpkg", layer="xs_bathy_points", driver="GPKG")
            (work_dir / "xs_mainstem_constraints_meta.json").write_text('{"constraints": {"ok": true}}')
            (work_dir / "xs_mainstem_constraints_accounting.json").write_text('{"soundings_n_matched": 1}')
            return 0, "Matched XS: 1", ""
        if tool == "river_skeleton_bathy.py":
            _write_raster(work_dir / "river_bed_elev_skeleton_full.tif", np.where(mask == 1, -3.0, -9999.0))
            return 0, "", ""
        raise AssertionError(f"unexpected tool: {tool}")

    cfg = SimpleNamespace(
        river_dem=dem,
        waffles_ocean_mask=None,
        python_exe="python",
        xs_spacing_m=10.0,
        xs_length_m=20.0,
        xs_smoothing_window_m=30.0,
        xs_deconflict_tol_m=5.0,
        xs_junction_snap_m=5.0,
        xs_junction_buffer_m=5.0,
        xs_densify_step_m=2.0,
        river_mainstem_min_order=5,
        xs_trim_overlaps=True,
        xs_global_deconflict=True,
        xs_skip_junctions=True,
        river_continuous=True,
        river_continuous_k=8,
        river_idw_power=2.0,
        river_aniso_along_scale_m=40.0,
        river_aniso_cross_scale_m=10.0,
        river_thalweg_weight=1.0,
        river_nodata=-9999.0,
        river_overlap_reducer="mean",
        river_xs_profile_shape="smooth_v",
        river_enable_1d_energy_solver=False,
        river_energy_allow_dem_proxy_wse=False,
        river_prior_mode="none",
        river_mv_a0=0.0,
        river_mv_bw=0.0,
        river_mv_ba=0.0,
        river_mv_bs=0.0,
        river_mv_eps_a=0.0,
        river_mv_eps_s=0.0,
        river_slope_proxy_window=5,
        river_slope_min=0.0,
        river_slope_max=1.0,
        river_slope_proxy_min_n=1,
        river_wse_profile_enabled=False,
        river_wse_profile_window=5,
        river_wse_profile_min_n=1,
        river_wse_profile_monotonic=False,
        river_usgs_sites=None,
        river_width_stage_csv=None,
        river_soundings="dummy.csv",
        soundings_sample_seed=1,
        soundings_max_points=100,
        river_soundings_calib_max_dist_m=0.0,
        river_soundings_calib_stat="median",
        river_skeleton_asymmetry_mode="none",
        river_skeleton_asymmetry_strength=0.0,
        river_skeleton_asymmetry_curv_ref=0.0,
        river_skeleton_asymmetry_max_shift=0.0,
        river_skeleton_asymmetry_min_width_m=0.0,
        river_skeleton_asymmetry_min_curv=0.0,
        river_skeleton_asymmetry_densify_step_m=0.0,
    )
    report = {"river": {"steps": {}}}
    result = run_hybrid_river_stage(
        cfg=cfg,
        report=report,
        script_dir=script_dir,
        work_dir=work_dir,
        cache_dir=cache_dir,
        network_gpkg=network_gpkg,
        build_domain_masks_fn=build_domain_masks_fn,
        estuary_clip_fn=estuary_clip_fn,
        run_command_fn=run_command_fn,
        validate_xs_artifacts_fn=validate_xs_artifacts,
        validate_soundings_subset_fn=lambda p: Path(p),
        append_river_soundings_args_fn=append_river_soundings_args_fn,
        build_river_skeleton_command_fn=build_river_skeleton_command_fn,
        authoritative_passthrough_args_fn=authoritative_passthrough_args_fn,
        record_authoritative_child_passthrough_fn=record_authoritative_child_passthrough_fn,
        logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None),
    )
    assert result.merged_bed_tif.exists()
    assert result.hybrid_merge_receipt_json.exists()
    receipt = pd.read_json(result.hybrid_merge_receipt_json, typ="series")
    assert int(receipt["unresolved_mainstem_pixels"]) == 0
    assert report["river"]["steps"]["xs_builder_mainstem"]["status"] == "success"




def test_phase_d_hybrid_stage_skips_soundings_subset_without_raw_soundings(tmp_path: Path):
    work_dir = tmp_path / "work_no_soundings"
    work_dir.mkdir()
    cache_dir = tmp_path / "cache_no_soundings"
    cache_dir.mkdir()
    script_dir = tmp_path
    dem = tmp_path / "dem_no_soundings.tif"
    _write_raster(dem, np.zeros((12, 12), dtype=np.float32))
    network_gpkg = tmp_path / "river_network_no_soundings.gpkg"
    gdf = gpd.GeoDataFrame({"streamorde": [5], "geometry": [LineString([(0, 6), (11, 6)])]}, crs="EPSG:32619")
    gdf.to_file(network_gpkg, layer="rivers_clip", driver="GPKG")

    channel = work_dir / "river_channel_mask.tif"
    open_water = work_dir / "open_water_mask.tif"
    mainstem = work_dir / "mainstem_mask.tif"
    mask = np.zeros((12, 12), dtype=np.float32)
    mask[2:10, 2:10] = 1
    _write_raster(channel, mask, nodata=255)
    _write_raster(open_water, np.zeros((12, 12), dtype=np.float32), nodata=255)
    _write_raster(mainstem, mask, nodata=255)

    subset_calls = []

    def build_domain_masks_fn(_work_dir, strict=False):
        return channel, open_water, mainstem

    estuary = work_dir / "estuary_clip_mask.tif"
    _write_raster(estuary, np.zeros_like(mask, dtype=np.float32), nodata=255)

    def estuary_clip_fn(*args, **kwargs):
        return 0, estuary

    def append_river_soundings_args_fn(cmd, cfg, include_calib_args=False, include_mode_args=False):
        return None

    def build_river_skeleton_command_fn(cfg, **kwargs):
        return ["python", "river_skeleton_bathy.py"]

    def authoritative_passthrough_args_fn(cfg, for_river=False):
        return []

    def record_authoritative_child_passthrough_fn(report, stage, cmd):
        report.setdefault("river", {}).setdefault("authoritative_passthrough", {})[stage] = True

    def run_command_fn(cmd, cwd=None, prefix=""):
        tool = cmd[1]
        if tool == "xs_builder.py":
            xs_gpkg = work_dir / "cross_sections_mainstem.gpkg"
            xs_lines = gpd.GeoDataFrame(
                {
                    "xs_id": [1],
                    "bank_left_dist_m": [1.0],
                    "bank_right_dist_m": [1.0],
                    "bank_left_z_m": [0.0],
                    "bank_right_z_m": [0.0],
                    "s_center_m": [1.0],
                    "geometry": [LineString([(1, 1), (3, 1)])],
                },
                crs="EPSG:32619",
            )
            xs_points = gpd.GeoDataFrame(
                {
                    "xs_id": [1, 1],
                    "s_m": [0.0, 2.0],
                    "z_m": [0.0, 0.0],
                    "is_bank_left": [1, 0],
                    "is_bank_right": [0, 1],
                    "geometry": [Point(1, 1), Point(3, 1)],
                },
                crs="EPSG:32619",
            )
            xs_lines.to_file(xs_gpkg, layer="xs_lines", driver="GPKG")
            xs_points.to_file(xs_gpkg, layer="xs_points", driver="GPKG")
            return 0, "", ""
        if tool == "xs_infer_bathy_raster.py" and "--only-write-soundings-subset" in cmd:
            subset_calls.append(cmd)
            return 1, "", "unexpected subset call"
        if tool == "xs_infer_bathy_raster.py":
            assert not any(str(c).startswith("--soundings-subset=") for c in cmd)
            _write_raster(work_dir / "river_bed_elev_xs_mainstem.tif", np.where(mask == 1, -2.0, -9999.0))
            gpd.GeoDataFrame({"geometry": [Point(1, 1)]}, crs="EPSG:32619").to_file(work_dir / "river_bathy_xs_mainstem.gpkg", layer="xs_bathy_points", driver="GPKG")
            (work_dir / "xs_mainstem_constraints_meta.json").write_text('{"constraints": {"ok": true}}')
            (work_dir / "xs_mainstem_constraints_accounting.json").write_text('{"soundings_n_matched": 0}')
            return 0, "Matched XS: 1", ""
        if tool == "river_skeleton_bathy.py":
            _write_raster(work_dir / "river_bed_elev_skeleton_full.tif", np.where(mask == 1, -3.0, -9999.0))
            return 0, "", ""
        raise AssertionError(f"unexpected tool: {tool}")

    cfg = SimpleNamespace(
        river_dem=dem,
        waffles_ocean_mask=None,
        python_exe="python",
        xs_spacing_m=10.0,
        xs_length_m=20.0,
        xs_smoothing_window_m=30.0,
        xs_deconflict_tol_m=5.0,
        xs_junction_snap_m=5.0,
        xs_junction_buffer_m=5.0,
        xs_densify_step_m=2.0,
        river_mainstem_min_order=5,
        xs_trim_overlaps=True,
        xs_global_deconflict=True,
        xs_skip_junctions=True,
        river_continuous=True,
        river_continuous_k=8,
        river_idw_power=2.0,
        river_aniso_along_scale_m=40.0,
        river_aniso_cross_scale_m=10.0,
        river_thalweg_weight=1.0,
        river_nodata=-9999.0,
        river_overlap_reducer="mean",
        river_xs_profile_shape="smooth_v",
        river_enable_1d_energy_solver=False,
        river_energy_allow_dem_proxy_wse=False,
        river_prior_mode="none",
        river_mv_a0=0.0,
        river_mv_bw=0.0,
        river_mv_ba=0.0,
        river_mv_bs=0.0,
        river_mv_eps_a=0.0,
        river_mv_eps_s=0.0,
        river_slope_proxy_window=5,
        river_slope_min=0.0,
        river_slope_max=1.0,
        river_slope_proxy_min_n=1,
        river_wse_profile_enabled=False,
        river_wse_profile_window=5,
        river_wse_profile_min_n=1,
        river_wse_profile_monotonic=False,
        river_usgs_sites=None,
        river_width_stage_csv=None,
        river_soundings=None,
        soundings_sample_seed=1,
        soundings_max_points=100,
        river_soundings_calib_max_dist_m=0.0,
        river_soundings_calib_stat="median",
        river_skeleton_asymmetry_mode="none",
        river_skeleton_asymmetry_strength=0.0,
        river_skeleton_asymmetry_curv_ref=0.0,
        river_skeleton_asymmetry_max_shift=0.0,
        river_skeleton_asymmetry_min_width_m=0.0,
        river_skeleton_asymmetry_min_curv=0.0,
        river_skeleton_asymmetry_densify_step_m=0.0,
    )
    report = {"river": {"steps": {}}}
    result = run_hybrid_river_stage(
        cfg=cfg,
        report=report,
        script_dir=script_dir,
        work_dir=work_dir,
        cache_dir=cache_dir,
        network_gpkg=network_gpkg,
        build_domain_masks_fn=build_domain_masks_fn,
        estuary_clip_fn=estuary_clip_fn,
        run_command_fn=run_command_fn,
        validate_xs_artifacts_fn=validate_xs_artifacts,
        validate_soundings_subset_fn=lambda p: Path(p),
        append_river_soundings_args_fn=append_river_soundings_args_fn,
        build_river_skeleton_command_fn=build_river_skeleton_command_fn,
        authoritative_passthrough_args_fn=authoritative_passthrough_args_fn,
        record_authoritative_child_passthrough_fn=record_authoritative_child_passthrough_fn,
        logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None),
    )
    assert result.merged_bed_tif.exists()
    assert subset_calls == []
    receipt = pd.read_json(result.soundings_subset_receipt_json, typ="series")
    assert receipt["status"] == "skipped"
    assert report["river"]["steps"]["soundings_subset"]["status"] == "skipped"

def test_hybrid_stage_writes_mask_stage_receipt(tmp_path):
    import json
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from types import SimpleNamespace
    from river_hybrid_stage import run_hybrid_river_stage

    work = tmp_path / "work2"
    work.mkdir()
    script_dir = tmp_path
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    network = work / "river_network.gpkg"
    network.write_text("dummy")
    template = work / "template.tif"
    profile = {"driver":"GTiff","height":4,"width":4,"count":1,"dtype":"float32","transform":from_origin(0,4,1,1),"crs":"EPSG:32619","nodata":-9999.0}
    with rasterio.open(template, "w", **profile) as ds: ds.write(np.zeros((4,4), dtype="float32"), 1)
    cfg = SimpleNamespace(
        waffles_ocean_mask=None, river_dem=template, xs_spacing_m=10.0, xs_length_m=20.0, xs_smoothing_window_m=5.0,
        xs_deconflict_tol_m=1.0, xs_junction_snap_m=1.0, xs_junction_buffer_m=1.0, xs_densify_step_m=1.0,
        river_mainstem_min_order=5, xs_trim_overlaps=True, xs_global_deconflict=True, xs_skip_junctions=True,
        river_continuous=True, river_continuous_k=8, river_idw_power=2.0, river_aniso_along_scale_m=50.0,
        river_aniso_cross_scale_m=10.0, river_thalweg_weight=1.0, river_nodata=-9999.0, river_overlap_reducer="mean",
        river_xs_profile_shape="trapezoid", river_enable_1d_energy_solver=False, river_soundings=None,
        river_soundings_calib_max_dist_m=0.0, river_soundings_calib_stat="median", python_exe="python3",
        river_prior_mode="geomorphic_v1", river_mv_a0=0.0, river_mv_bw=0.0, river_mv_ba=0.0, river_mv_bs=0.0,
        river_mv_eps_a=0.0, river_mv_eps_s=0.0, river_slope_proxy_window=5, river_slope_min=0.0, river_slope_max=1.0,
        river_slope_proxy_min_n=1, river_wse_profile_enabled=False, river_wse_profile_window=5, river_wse_profile_min_n=1,
        river_wse_profile_monotonic=False, river_usgs_sites=None, river_width_stage_csv=None,
        river_skeleton_asymmetry_mode="none", river_skeleton_asymmetry_strength=0.0, river_skeleton_asymmetry_curv_ref=0.0,
        river_skeleton_asymmetry_max_shift=0.0, river_skeleton_asymmetry_min_width_m=0.0, river_skeleton_asymmetry_min_curv=0.0,
        river_skeleton_asymmetry_densify_step_m=1.0
    )
    report = {"river": {"outputs": {}, "steps": {}}}

    def build_domain_masks_fn(_work_dir, strict=False):
        arr = np.zeros((4,4), dtype="uint8")
        arr[1:3,1:3] = 1
        for name in ["river_channel_mask.tif","mainstem_mask.tif"]:
            with rasterio.open(_work_dir / name, "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8", transform=from_origin(0,4,1,1), crs="EPSG:32619", nodata=0, tiled=True, blockxsize=16, blockysize=16) as ds:
                ds.write(arr,1)
        ow = np.zeros((4,4), dtype="uint8")
        ow[0,:] = 1
        with rasterio.open(_work_dir / "open_water_mask.tif", "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8", transform=from_origin(0,4,1,1), crs="EPSG:32619", nodata=0, tiled=True, blockxsize=16, blockysize=16) as ds:
            ds.write(ow,1)
        return _work_dir / "river_channel_mask.tif", _work_dir / "open_water_mask.tif", _work_dir / "mainstem_mask.tif"

    def estuary_clip_fn(channel_mask_tif, cfg, ocean_mask_path=None, report=None):
        est = work / "estuary_clip_mask.tif"
        clip = np.zeros((4,4), dtype="uint8")
        clip[2,2] = 1
        with rasterio.open(est, "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8", transform=from_origin(0,4,1,1), crs="EPSG:32619", tiled=True, blockxsize=16, blockysize=16, nodata=0) as ds:
            ds.write(clip,1)
        with rasterio.open(channel_mask_tif) as src:
            arr = src.read(1)
            prof = src.profile.copy()
        arr[2,2] = 0
        with rasterio.open(channel_mask_tif, "w", **prof) as ds:
            ds.write(arr, 1)
        return 1, est

    def run_command_fn(cmd, cwd=None, prefix=""):
        if "xs_builder.py" in cmd:
            gpkg = work / "cross_sections_mainstem.gpkg"
            import geopandas as gpd
            from shapely.geometry import LineString, Point
            gpd.GeoDataFrame({"xs_id":[1],"bank_left_dist_m":[1.0],"bank_right_dist_m":[1.0],"bank_left_z_m":[0.0],"bank_right_z_m":[0.0],"s_center_m":[0.0]}, geometry=[LineString([(0,0),(1,1)])], crs="EPSG:32619").to_file(gpkg, layer="xs_lines", driver="GPKG")
            gpd.GeoDataFrame({"xs_id":[1,1],"s_m":[0.0,1.0],"z_m":[0.0,0.0],"is_bank_left":[True,False],"is_bank_right":[False,True]}, geometry=[Point(0,0),Point(1,1)], crs="EPSG:32619").to_file(gpkg, layer="xs_points", driver="GPKG")
            return 0, "", ""
        if "--only-write-soundings-subset" in cmd:
            out = next(Path(c.split("=",1)[1]) for c in cmd if str(c).startswith("--write-soundings-subset="))
            out.write_bytes(b"PAR1")
            return 0, "", ""
        if "xs_infer_bathy_raster.py" in cmd:
            out_tif = next(Path(c.split("=",1)[1]) for c in cmd if str(c).startswith("--out-bathy-raster="))
            out_gpkg = next(Path(c.split("=",1)[1]) for c in cmd if str(c).startswith("--out-gpkg="))
            meta = next(Path(c.split("=",1)[1]) for c in cmd if str(c).startswith("--out-meta-json="))
            acct = next(Path(c.split("=",1)[1]) for c in cmd if str(c).startswith("--out-accounting-json="))
            with rasterio.open(out_tif, "w", **profile) as ds: ds.write(np.ones((4,4), dtype="float32"),1)
            out_gpkg.write_text("gpkg")
            meta.write_text(json.dumps({"constraints": {}}))
            acct.write_text(json.dumps({}))
            return 0, "", ""
        if "river_skeleton_bathy.py" in cmd[-1] or "river_skeleton_bathy.py" in " ".join(map(str,cmd)):
            return 0, "", ""
        return 0, "", ""

    def validate_xs_artifacts_fn(path): return {"ok": True}
    def validate_soundings_subset_fn(path): return path
    def append_river_soundings_args_fn(cmd, cfg, include_calib_args=True, include_mode_args=False): cmd.append("--soundings=dummy")
    def build_river_skeleton_command_fn(*args, **kwargs):
        out = work / "river_bed_elev_skeleton_full.tif"
        with rasterio.open(out, "w", **profile) as ds: ds.write(np.full((4,4), 2.0, dtype="float32"),1)
        return ["python3", "river_skeleton_bathy.py"]
    def authoritative_passthrough_args_fn(*args, **kwargs): return []
    def record_authoritative_child_passthrough_fn(*args, **kwargs): return None
    import logging
    result = run_hybrid_river_stage(cfg=cfg, report=report, script_dir=script_dir, work_dir=work, cache_dir=cache_dir, network_gpkg=network, build_domain_masks_fn=build_domain_masks_fn, estuary_clip_fn=estuary_clip_fn, run_command_fn=run_command_fn, validate_xs_artifacts_fn=validate_xs_artifacts_fn, validate_soundings_subset_fn=validate_soundings_subset_fn, append_river_soundings_args_fn=append_river_soundings_args_fn, build_river_skeleton_command_fn=build_river_skeleton_command_fn, authoritative_passthrough_args_fn=authoritative_passthrough_args_fn, record_authoritative_child_passthrough_fn=record_authoritative_child_passthrough_fn, logger=logging.getLogger("test"))
    assert result.river_mask_stage_receipt_json.exists()
    rec = json.loads(result.river_mask_stage_receipt_json.read_text())
    assert rec["mainstem_subset_of_channel"] is True
    assert rec["estuary_excluded_from_final_channel"] is True
