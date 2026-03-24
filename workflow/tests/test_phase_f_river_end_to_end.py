from pathlib import Path
from types import SimpleNamespace
import json

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


def test_phase_f_end_to_end_receipt_chain(tmp_path: Path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    script_dir = tmp_path
    dem = tmp_path / "dem.tif"
    _write_raster(dem, np.zeros((12, 12), dtype=np.float32))
    network_gpkg = tmp_path / "river_network.gpkg"
    gdf = gpd.GeoDataFrame({"streamorde": [5], "geometry": [LineString([(0, 6), (11, 6)])]}, crs="EPSG:32619")
    gdf.to_file(network_gpkg, layer="rivers_clip", driver="GPKG")

    channel = work_dir / "river_channel_mask.tif"
    open_water = work_dir / "open_water_mask.tif"
    mainstem = work_dir / "mainstem_mask.tif"
    mask = np.zeros((12, 12), dtype=np.float32)
    mask[2:10, 2:10] = 1
    _write_raster(channel, mask, nodata=255)
    ocean = np.zeros((12, 12), dtype=np.float32)
    ocean[0, :] = 1
    _write_raster(open_water, ocean, nodata=255)
    ocean_mask = work_dir / "ocean_only.tif"
    ocean_land = np.ones((12, 12), dtype=np.float32)
    ocean_land[0, :] = 0
    _write_raster(ocean_mask, ocean_land, nodata=255)
    _write_raster(mainstem, mask, nodata=255)

    def build_domain_masks_fn(_work_dir, strict=False):
        return channel, open_water, mainstem

    pre_subset = work_dir / "river_soundings_subset.parquet"
    pre_subset.write_text("placeholder")
    (work_dir / "river_soundings_subset.meta.json").write_text(json.dumps({
        "signature": {"river_soundings": "dummy.csv", "soundings_sample_seed": 1, "soundings_max_points": 100},
        "subset_path": str(pre_subset),
    }))

    estuary = work_dir / "estuary_clip_mask.tif"
    est_arr = np.zeros((12, 12), dtype=np.float32)
    est_arr[2:4, 2:4] = 1
    _write_raster(estuary, est_arr, nodata=255)

    def estuary_clip_fn(channel_mask_tif, *args, **kwargs):
        with rasterio.open(channel_mask_tif) as ds:
            ch = ds.read(1)
            profile = ds.profile.copy()
        ch[est_arr == 1] = 0
        with rasterio.open(channel_mask_tif, "w", **profile) as ds:
            ds.write(ch, 1)
        return int(est_arr.sum()), estuary

    def append_river_soundings_args_fn(cmd, cfg, include_calib_args=False, include_mode_args=False):
        return None

    def build_river_skeleton_command_fn(cfg, **kwargs):
        return ["python", "river_skeleton_bathy.py"]

    def authoritative_passthrough_args_fn(cfg, for_river=False):
        return []

    def record_authoritative_child_passthrough_fn(report, stage, cmd):
        report.setdefault("river", {}).setdefault("authoritative_passthrough", {})[stage] = True

    def validate_soundings_subset_fn(path):
        return {"path": str(path), "rows": 1, "depth_col": "depth_m", "crs": "EPSG:32619"}

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
            return 0, "", ""
        if tool == "xs_infer_bathy_raster.py":
            _write_raster(work_dir / "river_bed_elev_xs_mainstem.tif", np.where(mask == 1, -2.0, -9999.0))
            gpd.GeoDataFrame({"geometry": [Point(1, 1)]}, crs="EPSG:32619").to_file(work_dir / "river_bathy_xs_mainstem.gpkg", layer="xs_bathy_points", driver="GPKG")
            (work_dir / "xs_mainstem_constraints_meta.json").write_text('{"constraints": {"ok": true}}')
            (work_dir / "xs_mainstem_constraints_accounting.json").write_text('{"soundings_n_matched": 1}')
            (work_dir / "xs_bank_contract_receipt.json").write_text('{"xs_total": 1, "xs_with_both_bank_contract": 1}')
            return 0, "Matched XS: 1", ""
        if tool == "river_skeleton_bathy.py":
            _write_raster(work_dir / "river_bed_elev_skeleton_full.tif", np.where(mask == 1, -3.0, -9999.0))
            return 0, "", ""
        raise AssertionError(f"unexpected tool: {tool}")

    cfg = SimpleNamespace(
        river_dem=dem,
        waffles_ocean_mask=ocean_mask,
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
    report = {"river": {"steps": {}, "outputs": {}}}
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
        validate_soundings_subset_fn=validate_soundings_subset_fn,
        append_river_soundings_args_fn=append_river_soundings_args_fn,
        build_river_skeleton_command_fn=build_river_skeleton_command_fn,
        authoritative_passthrough_args_fn=authoritative_passthrough_args_fn,
        record_authoritative_child_passthrough_fn=record_authoritative_child_passthrough_fn,
        logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None),
    )
    assert result.river_run_receipt_json.exists()
    chain = json.loads(result.river_run_receipt_json.read_text())
    for key in [
        "river_mask_stage_receipt",
        "xs_contract_receipt",
        "soundings_subset_receipt",
        "xs_bank_contract_receipt",
        "skeleton_receipt",
        "hybrid_merge_receipt",
    ]:
        assert chain[key] is not None
        assert Path(chain[key]).exists()
    assert chain["summary"]["unresolved_mainstem_pixels"] == 0
    assert report["river"]["outputs"]["river_run_receipt"].endswith("river_run_receipt.json")
