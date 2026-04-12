from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box, LineString

from river_v2_context import RiverV2Context, resolve_river_v2_bank_guidance_inputs
from river_v2_pipeline import (
    _ensure_wse_support_source_raster,
    run_river_v2_pass1,
    run_river_v2_pass2,
    run_river_v2_pass3,
    run_river_v2_pass4,
)
from river_v2_contract import (
    STAGE_CENTERLINE_AUTHORITATIVE_BED,
    STAGE_CENTERLINE_BED_BACKBONE,
    STAGE_CENTERLINE_BED_BACKBONE_DENSE,
    STAGE_CENTERLINE_OBSERVED_OFFSET,
    STAGE_CENTERLINE_OFFSET_MODELED,
    STAGE_CENTERLINE_WSE_PROXY,
    STAGE_RIVER_CENTERLINE,
    STAGE_RIVER_PRIMARY_SURFACE,
    STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
)


class _Cfg:
    river_mainstem_min_order = 5
    river_centerline_sample_spacing_m = 1.0


def _write_domain_mask(path: Path):
    arr = np.zeros((7, 7), dtype="uint8")
    arr[2:5, 1:6] = 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(-0.5, 5.5, 1.0, 1.0),
        nodata=0,
    ) as ds:
        ds.write(arr, 1)



def _write_authoritative_support_coverage(path: Path):
    support = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(0.5, 1.5, 4.5, 2.5)],
        crs="EPSG:4326",
    )
    support.to_file(path, driver="GPKG")


def _write_authoritative_base(path: Path):
    arr = np.full((7, 7), np.nan, dtype="float32")
    arr[3, 1:5] = np.array([0.8, 0.6, 0.4, 0.2], dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-0.5, 5.5, 1.0, 1.0),
        nodata=np.nan,
    ) as ds:
        ds.write(arr, 1)



def _write_network_gpkg(path: Path):
    flows = gpd.GeoDataFrame(
        {
            "streamorde": [6],
            "lengthkm": [0.5],
            "levelpathi": ["main"],
            "nhdplusid": ["r1"],
            "from_node": [1],
            "to_node": [2],
        },
        geometry=[LineString([(0, 3), (3, 3)])],
        crs="EPSG:4326",
    )
    nhd = gpd.GeoDataFrame({"id":[1]}, geometry=[box(-0.5, 2.5, 3.5, 3.5)], crs="EPSG:4326")
    flows.to_file(path, layer="rivers_clip", driver="GPKG")
    nhd.to_file(path, layer="nhdarea_clip", driver="GPKG")


def _write_authoritative_support_points_csv(path: Path):
    path.write_text(
        "x,y,depth_m,authoritative_role\n0.0,3.0,-1.5,authoritative_bed\n1.0,3.0,-1.3,authoritative_bed\n2.0,3.0,-1.1,authoritative_bed\n3.0,3.0,-1.0,authoritative_bed\n",
        encoding="utf-8",
    )

def _write_river_dem(path: Path):
    arr = np.full((7, 7), 1.0, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-0.5, 5.5, 1.0, 1.0),
        nodata=np.nan,
    ) as ds:
        ds.write(arr, 1)


def _build_context(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "point_id": ["a", "b", "c", "d"],
            "station_m": [0.0, 1.0, 2.0, 3.0],
            "bank_wse_proxy_monotone_m": [2.6, 2.8, 3.0, 3.2],
            "centerline_z_m": [1.5, 1.3, 1.1, 1.0],
        },
        geometry=[Point(0, 3), Point(1, 3), Point(2, 3), Point(3, 3)],
        crs="EPSG:4326",
    )
    network_gpkg = tmp_path / "network.gpkg"
    domain_mask = tmp_path / "river_guidance_domain_mask.tif"
    river_dem = tmp_path / "river_dem.tif"
    authoritative_base = tmp_path / "authoritative_base.tif"
    authoritative_support_coverage = tmp_path / "authoritative_support_coverage.gpkg"
    authoritative_support_points = tmp_path / "authoritative_support_points.csv"
    _write_network_gpkg(network_gpkg)
    _write_domain_mask(domain_mask)
    _write_river_dem(river_dem)
    _write_authoritative_base(authoritative_base)
    _write_authoritative_support_coverage(authoritative_support_coverage)
    _write_authoritative_support_points_csv(authoritative_support_points)
    return RiverV2Context(
        cfg=_Cfg(),
        report={},
        out_dir=tmp_path,
        network_gpkg=network_gpkg,
        river_dem_path=river_dem,
        channel_mask_path=domain_mask,
        authoritative_base_path=authoritative_base,
        authoritative_bed_path=authoritative_support_points,
        authoritative_support_coverage_path=authoritative_support_coverage,
        centerline_points_gdf=gdf,
        vertical_reference="NAVD88",
    )


def test_river_v2_pass1_runs_from_inline_centerline_points(tmp_path):
    ctx = _build_context(tmp_path)
    result = run_river_v2_pass1(ctx)
    assert result.success is True
    assert set(result.stage_results.keys()) == {
        STAGE_RIVER_CENTERLINE,
        STAGE_CENTERLINE_WSE_PROXY,
        STAGE_CENTERLINE_AUTHORITATIVE_BED,
        STAGE_CENTERLINE_OBSERVED_OFFSET,
    }
    assert ctx.paths.centerline_observed_offset_points.exists()
    assert ctx.paths.pass1_summary.exists()
    assert result.stage_status[STAGE_CENTERLINE_OBSERVED_OFFSET]["implemented"] is True
    wse = gpd.read_file(ctx.paths.centerline_wse_proxy_points).sort_values("station_m", kind="mergesort")
    assert {"wse_proxy_z_m", "has_support", "wse_valid", "stream_order", "mean_width_m"}.issubset(wse.columns)
    assert wse["stream_order"].notna().all()
    assert wse["mean_width_m"].notna().all()
    assert "wse_trend_z_m" not in wse.columns
    assert "wse_pre_smooth_z_m" not in wse.columns
    assert "final_adjustment_reason" not in wse.columns
    vals = wse["wse_proxy_z_m"].to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    station_direction = str(wse["station_direction"].iloc[0])
    if station_direction == "downstream_increasing":
        assert np.all(np.diff(finite_vals) <= 1.0e-6)
    else:
        assert np.all(np.diff(finite_vals) >= -1.0e-6)
    assert finite_vals.size > 0
    assert ctx.paths.authoritative_bed_support_points.exists()
    assert ctx.paths.centerline_wse_support_points.exists()
    assert ctx.paths.centerline_wse_trend_points.exists()
    assert ctx.paths.centerline_wse_pre_smooth_points.exists()
    assert ctx.paths.centerline_wse_stage_diagnostics.exists()
    assert ctx.paths.centerline_wse_stage_summary.exists()
    receipt_text = ctx.paths.centerline_wse_proxy_points_receipt.read_text(encoding="utf-8")
    import json
    if ctx.paths.bank_wse_edge_guidance.exists():
        assert str(ctx.paths.bank_wse_edge_guidance) in receipt_text
        mat_diag = json.loads((ctx.paths.support_dir / "river_bank_wse_materialization_diagnostics.json").read_text(encoding="utf-8"))
        assert mat_diag["status"] == "success"
        assert mat_diag["source_contract"] in {"river_dem_template", "aligned_authoritative_base", "authoritative_base", "measured_only_authoritative_sampling_raster"}
    diag = json.loads(ctx.paths.centerline_wse_stage_diagnostics.read_text(encoding="utf-8"))
    assert diag["artifacts"]["support_path"].endswith("centerline_wse_support_points.gpkg")
    assert diag["artifacts"]["trend_path"].endswith("centerline_wse_trend_points.gpkg")
    assert diag["artifacts"]["pre_smooth_path"].endswith("centerline_wse_pre_smooth_points.gpkg")
    assert diag["artifacts"]["final_wse_path"].endswith("centerline_wse_proxy_points.gpkg")
    assert "support" in diag and "trend" in diag and "pre_smooth" in diag and "final" in diag


def test_river_v2_pass2_builds_modeled_offset_and_backbone(tmp_path):
    ctx = _build_context(tmp_path)
    result = run_river_v2_pass2(ctx)
    assert result.success is True
    assert set(result.stage_results.keys()) == {
        STAGE_RIVER_CENTERLINE,
        STAGE_CENTERLINE_WSE_PROXY,
        STAGE_CENTERLINE_AUTHORITATIVE_BED,
        STAGE_CENTERLINE_OBSERVED_OFFSET,
        STAGE_CENTERLINE_OFFSET_MODELED,
        STAGE_CENTERLINE_BED_BACKBONE,
    STAGE_CENTERLINE_BED_BACKBONE_DENSE,
    }
    assert ctx.paths.centerline_offset_modeled_points.exists()
    assert ctx.paths.component_stream_summary.exists()
    assert ctx.paths.component_stream_summary_receipt.exists()
    assert ctx.paths.river_centerline_bed_backbone_points.exists()
    assert ctx.paths.river_centerline_bed_backbone_dense_points.exists()
    assert ctx.paths.pass2_summary.exists()
    modeled = gpd.read_file(ctx.paths.centerline_offset_modeled_points)
    summary = gpd.read_file(ctx.paths.component_stream_summary)
    backbone = gpd.read_file(ctx.paths.river_centerline_bed_backbone_points)
    assert "offset_modeled_m" in modeled.columns
    assert modeled["offset_modeled_m"].notna().all()
    assert "placeholder_offset_m" in summary.columns
    assert summary["mean_width_m"].notna().all()
    assert "width" in set(summary["placeholder_basis"].astype(str)) or any(str(v).startswith("width") for v in summary["placeholder_basis"].astype(str))
    assert "bed_backbone_z_m" in backbone.columns
    assert backbone["bed_backbone_z_m"].notna().all()


def test_river_v2_pass3_builds_domain_and_primary_surface_in_one_stage(tmp_path):
    ctx = _build_context(tmp_path)
    result = run_river_v2_pass3(ctx)
    assert result.success is True
    assert STAGE_RIVER_PRIMARY_SURFACE in result.stage_results
    assert ctx.paths.river_primary_surface_domain.exists()
    assert ctx.paths.river_primary_surface.exists()
    assert ctx.paths.pass3_summary.exists()
    with rasterio.open(ctx.paths.river_primary_surface_domain) as ds:
        domain = ds.read(1)
    assert np.count_nonzero(domain) > 0
    with rasterio.open(ctx.paths.river_primary_surface) as ds:
        arr = ds.read(1)
        assert np.count_nonzero(np.isfinite(arr)) > 0
        assert ds.crs.to_string() == "EPSG:4326"


def test_river_v2_pass4_builds_authoritative_locked_primary_surface(tmp_path):
    ctx = _build_context(tmp_path)
    authoritative_base = tmp_path / "authoritative_base_override.tif"
    arr = np.full((7, 7), np.nan, dtype="float32")
    arr[3, 3] = -2.0
    with rasterio.open(
        authoritative_base,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-0.5, 5.5, 1.0, 1.0),
        nodata=np.nan,
    ) as ds:
        ds.write(arr, 1)
    ctx.authoritative_base_path = authoritative_base
    result = run_river_v2_pass4(ctx)
    assert result.success is True
    assert STAGE_RIVER_PRIMARY_SURFACE_LOCKED in result.stage_results
    assert ctx.paths.river_primary_surface_authoritative_applied.exists()
    assert ctx.paths.pass4_summary.exists()
    with rasterio.open(ctx.paths.river_primary_surface_authoritative_applied) as ds:
        out = ds.read(1)
        assert np.isclose(out[3, 3], -2.0)


def test_river_v2_pass4_prefers_authoritative_base_over_aligned_baseline(tmp_path):
    ctx = _build_context(tmp_path)
    aligned_auth = tmp_path / "aligned_authoritative_base.tif"
    aligned = np.full((7, 7), 99.0, dtype="float32")
    with rasterio.open(
        aligned_auth,
        "w",
        driver="GTiff",
        width=aligned.shape[1],
        height=aligned.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-0.5, 5.5, 1.0, 1.0),
        nodata=np.nan,
    ) as ds:
        ds.write(aligned, 1)
    ctx.aligned_authoritative_base_path = aligned_auth
    result = run_river_v2_pass4(ctx)
    assert result.success is True
    with rasterio.open(ctx.paths.river_primary_surface_authoritative_applied) as ds:
        out = ds.read(1)
        assert np.isclose(out[3, 1], 0.8)
        assert np.isclose(out[3, 2], 0.6)
        assert np.isclose(out[3, 3], 0.4)
        assert np.isclose(out[3, 4], 0.2)
        assert not np.isclose(out[3, 3], 99.0)


def test_river_v2_pass4_succeeds_as_noop_when_no_lockable_authoritative_cells(tmp_path):
    ctx = _build_context(tmp_path)
    empty_auth = tmp_path / "empty_authoritative_base.tif"
    arr = np.full((7, 7), -9999.0, dtype="float32")
    with rasterio.open(
        empty_auth,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-0.5, 5.5, 1.0, 1.0),
        nodata=-9999.0,
    ) as ds:
        ds.write(arr, 1)
    ctx.authoritative_base_path = empty_auth
    result = run_river_v2_pass4(ctx)
    assert result.success is True
    validation = result.stage_results[STAGE_RIVER_PRIMARY_SURFACE_LOCKED].validation
    assert validation["valid"] is True
    assert validation["authoritative_lock_mode"] == "no_lockable_authoritative_cells"
    assert validation["authoritative_locked_cell_count"] == 0


def test_enforce_v2_final_route_selection_rejects_legacy_conflict(tmp_path):
    from bathy_main import _enforce_v2_final_route_selection

    locked = tmp_path / "river_primary_surface_authoritative_applied.tif"
    locked.write_text("x", encoding="utf-8")
    legacy = tmp_path / "legacy_river_surface.tif"
    legacy.write_text("x", encoding="utf-8")
    try:
        _enforce_v2_final_route_selection(
            river_outputs={
                "primary_river_guidance_surface": str(locked),
                "river_primary_surface_authoritative_applied": str(locked),
                "river_channel_surface": str(legacy),
            },
            active_locked_surface=str(locked),
        )
    except RuntimeError as exc:
        assert "legacy river outputs remain eligible for routing" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for legacy routing conflict")


def test_resolve_river_v2_bank_guidance_inputs_prefers_explicit_stage_outputs(tmp_path):
    edge = tmp_path / "wse_edge.tif"
    edge.write_text("x", encoding="utf-8")
    profile = tmp_path / "wse_profile.csv"
    profile.write_text("station_m,wse\n0,1\n", encoding="utf-8")
    report = {
        "river": {
            "simple_river_stage_outputs": {
                "wse_proxy": str(edge),
                "wse_profile_summary": str(profile),
            },
            "outputs": {
                "bank_elevation_path": str(tmp_path / "wrong_edge.tif"),
                "bank_profile_summary_path": str(tmp_path / "wrong_profile.csv"),
            },
        }
    }
    resolved = resolve_river_v2_bank_guidance_inputs(report, tmp_path / "river_v2")
    assert resolved["bank_wse_edge_guidance_path"] == str(edge)


def test_river_v2_wse_stage_builds_explicit_bank_guidance_when_inline_fields_missing(tmp_path):
    ctx = _build_context(tmp_path)
    result = run_river_v2_pass1(ctx)
    assert result.success is True
    assert ctx.paths.authoritative_bed_support_points.exists()
    if ctx.paths.bank_wse_edge_guidance.exists():
        assert ctx.paths.wse_support_source_raster.exists()
    assert STAGE_CENTERLINE_WSE_PROXY in result.stage_results


def test_river_v2_backbone_preserves_all_points_and_only_smooths_inferred_spans(tmp_path):
    from river_v2_stage_backbone import build_backbone

    geom = [Point(float(i), 0.0) for i in range(6)]
    wse = gpd.GeoDataFrame(
        {
            "point_id": [f"p{i}" for i in range(6)],
            "station_m": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "wse_proxy_z_m": [5.0, 5.0, 5.0, 4.0, 4.0, 3.0],
        },
        geometry=geom,
        crs="EPSG:4326",
    )
    modeled = gpd.GeoDataFrame(
        {
            "point_id": [f"p{i}" for i in range(6)],
            "offset_modeled_m": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        geometry=geom,
        crs="EPSG:4326",
    )
    wse_path = tmp_path / "wse.gpkg"
    modeled_path = tmp_path / "modeled.gpkg"
    wse.to_file(wse_path, driver="GPKG")
    modeled.to_file(modeled_path, driver="GPKG")

    backbone, diagnostics, warnings = build_backbone(wse_path, modeled_path)
    ordered = backbone.sort_values("station_m", kind="mergesort")
    smoothed = ordered["bed_backbone_z_m"].to_numpy(dtype=float)

    assert diagnostics["input_record_count"] == 6
    assert diagnostics["record_count"] == 6
    assert set(ordered["backbone_support_class"].astype(str).unique()) <= {"observed_anchor", "inferred"}
    assert np.all(np.isfinite(smoothed))
    assert warnings == []


def test_river_v2_uses_report_authoritative_base_when_cfg_base_missing(tmp_path):
    ctx = _build_context(tmp_path)
    ctx.aligned_authoritative_base_path = ctx.authoritative_base_path
    ctx.authoritative_base_path = None
    result = run_river_v2_pass1(ctx)
    assert result.success is True
    assert ctx.paths.authoritative_measured_only_projected.exists()
    auth = gpd.read_file(ctx.paths.centerline_authoritative_bed_points)
    assert len(auth) > 0


def test_river_v2_pass1_materializes_explicit_authoritative_bed_support_artifact(tmp_path):
    ctx = _build_context(tmp_path)
    result = run_river_v2_pass1(ctx)
    assert result.success is True
    assert ctx.paths.authoritative_bed_support_points.exists()
    bed_receipt = result.stage_results[STAGE_CENTERLINE_AUTHORITATIVE_BED].receipt_path
    import json
    payload = json.loads(Path(bed_receipt).read_text(encoding="utf-8"))
    assert any(str(ctx.paths.authoritative_bed_support_points) == str(v) for v in payload.get("input_artifacts", []))


def test_wse_support_materialization_requires_explicit_sampling_raster(tmp_path):
    ctx = _build_context(tmp_path)
    ctx.centerline_points_gdf = gpd.GeoDataFrame(
        {
            "point_id": ["a", "b"],
            "station_m": [0.0, 1.0],
            "centerline_z_m": [1.0, 0.9],
        },
        geometry=[Point(0, 3), Point(1, 3)],
        crs="EPSG:4326",
    )
    ctx.authoritative_sampling_raster_path = None
    ctx.aligned_authoritative_base_path = None
    ctx.authoritative_base_path = None
    ctx.river_dem_path = None
    result = run_river_v2_pass1(ctx)
    assert result.success is False
    stage_error = result.stage_status[STAGE_CENTERLINE_WSE_PROXY]["error"]
    assert "river_v2_wse_missing_support_source_raster" in stage_error
    import json
    mat_diag = json.loads((ctx.paths.support_dir / "river_bank_wse_materialization_diagnostics.json").read_text(encoding="utf-8"))
    assert mat_diag["status"] == "failed"
    assert mat_diag["error"] == "missing_wse_support_source_raster"
    assert not ctx.paths.wse_support_source_raster.exists()


def test_backbone_preserves_observed_anchor_points(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point
    from river_v2_stage_backbone import build_backbone

    wse = gpd.GeoDataFrame({
        "point_id": ["p1", "p2", "p3"],
        "station_m": [0.0, 1.0, 2.0],
        "wse_proxy_z_m": [1.0, 1.0, 0.5],
        "levelpath_id": ["L", "L", "L"],
        "geometry": [Point(0,0), Point(1,0), Point(2,0)],
    }, geometry="geometry", crs="EPSG:4326")
    off = gpd.GeoDataFrame({
        "point_id": ["p1", "p2", "p3"],
        "station_m": [0.0, 1.0, 2.0],
        "offset_modeled_m": [2.0, 2.0, 2.0],
        "observed_offset_m": [2.0, 2.0, float("nan")],
        "offset_support_class": ["observed_anchor", "observed_anchor", "continued_downstream"],
        "offset_model_source": ["observed_anchor", "observed_anchor", "continued_downstream"],
        "levelpath_id": ["L", "L", "L"],
        "geometry": [Point(0,0), Point(1,0), Point(2,0)],
    }, geometry="geometry", crs="EPSG:4326")
    wse_path = tmp_path / "wse.gpkg"
    off_path = tmp_path / "off.gpkg"
    wse.to_file(wse_path, driver="GPKG")
    off.to_file(off_path, driver="GPKG")
    backbone, diagnostics, warnings = build_backbone(wse_path, off_path)
    assert set(backbone["point_id"]) >= {"p1", "p2"}
    assert diagnostics["record_count"] == len(backbone)


def test_wse_support_source_prefers_authoritative_base_over_aligned_baseline_when_river_dem_missing(tmp_path):
    ctx = _build_context(tmp_path)
    ctx.river_dem_path = None
    aligned_auth = tmp_path / "aligned_authoritative_base.tif"
    aligned = np.full((7, 7), 99.0, dtype="float32")
    with rasterio.open(
        aligned_auth,
        "w",
        driver="GTiff",
        width=aligned.shape[1],
        height=aligned.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-0.5, 5.5, 1.0, 1.0),
        nodata=np.nan,
    ) as ds:
        ds.write(aligned, 1)
    ctx.aligned_authoritative_base_path = aligned_auth
    source_path, source_contract = _ensure_wse_support_source_raster(ctx)
    assert source_path == ctx.paths.wse_support_source_raster
    assert source_contract == "authoritative_base"
    with rasterio.open(source_path) as ds:
        out = ds.read(1)
    assert np.isclose(out[3, 1], 0.8)
    assert np.isclose(out[3, 2], 0.6)
    assert np.isclose(out[3, 3], 0.4)
    assert np.isclose(out[3, 4], 0.2)
    assert not np.isclose(out[3, 3], 99.0)


def test_wse_support_source_refreshes_stale_local_artifact_from_current_source(tmp_path):
    ctx = _build_context(tmp_path)
    ctx.paths.support_dir.mkdir(parents=True, exist_ok=True)
    stale = np.full((7, 7), -123.0, dtype="float32")
    with rasterio.open(
        ctx.paths.wse_support_source_raster,
        "w",
        driver="GTiff",
        width=stale.shape[1],
        height=stale.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-0.5, 5.5, 1.0, 1.0),
        nodata=np.nan,
    ) as ds:
        ds.write(stale, 1)
    source_path, source_contract = _ensure_wse_support_source_raster(ctx)
    assert source_path == ctx.paths.wse_support_source_raster
    assert source_contract == "river_dem_template"
    with rasterio.open(source_path) as ds:
        out = ds.read(1)
    assert np.allclose(out[np.isfinite(out)], 1.0)
