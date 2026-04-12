import pytest
import numpy as np
import json
from pathlib import Path
from types import SimpleNamespace

from river_guidance import (
    _derive_xs_support_status,
    build_river_guidance_manifest,
    guidance_artifact_paths,
    rasterize_river_guide_points_to_template,
    write_river_guidance_manifest,
    _load_and_validate_centerline_component_contract,
)


def test_build_and_write_river_guidance_manifest(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    for key in (
        "guidance_weight",
        "trusted_interior",
        "soft_guidance_domain",
        "admissibility",
        "corridor_mask",
        "authoritative_support",
        "authoritative_support_depth",
        "bank_edge_mask",
        "bank_distance",
        "bank_influence",
        "bank_continuity_weight",
        "bank_graph_confidence",
        "bank_confluence_damping",
        "bank_estuary_side_decay",
        "depth_terrain",
        "bottom_elevation",
        "scaffold_domains",
        "scaffold_manifest",
        "guide_points",
        "channel_frame_points",
        "channel_frame_contract",
        "authoritative_centerline_anchors",
        "authoritative_xs_anchors",
    ):
        p = paths[key]
        if p.suffix in {".json", ".gpkg"}:
            p.write_text("{}", encoding="utf-8")
        else:
            p.write_bytes(b"x")

    report = {
        "river": {
            "outputs": {
                "guidance_weight": str(paths["guidance_weight"]),
                "trusted_interior": str(paths["trusted_interior"]),
                "soft_guidance_domain": str(paths["soft_guidance_domain"]),
                "admissibility": str(paths["admissibility"]),
                "guide_points": str(paths["guide_points"]),
                "depth_terrain": str(paths["depth_terrain"]),
                "bottom_elevation": str(paths["bottom_elevation"]),
            },
            "guidance": {
                "trusted_interior_definition": "trusted export interior",
                "soft_guidance_definition": "soft guidance domain",
                "admissibility_definition": "anchor-excluded admissible guidance",
            },
        }
    }
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    assert manifest["guidance_only"] is True
    assert manifest["artifact_roles"]["depth_terrain"] == "diagnostic_only"
    assert "depth_terrain" in manifest["final_route_contract"]["diagnostic_only_artifacts"]
    assert "guide_points" in manifest["final_route_contract"]["allowed_structural_artifacts"]
    assert manifest["artifact_roles"]["guide_points"] == "structured_scaffold_points"
    assert manifest["artifact_roles"]["bank_influence"] == "corridor_bank_influence"
    assert manifest["notes"]["trusted_interior"] == "trusted export interior"
    assert manifest["artifact_roles"]["bank_continuity_weight"] == "xs_bank_longitudinal_continuity"
    assert manifest["artifact_roles"]["bank_graph_confidence"] == "graph_informed_bank_confidence"
    assert manifest["artifact_roles"]["channel_frame_points"] == "authoritative_first_channel_frame_stations"
    assert "channel_frame_points" in manifest["final_route_contract"]["allowed_structural_artifacts"]

    manifest_path = write_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["artifacts"]["guide_points"].endswith("river_guide_points.gpkg")



def test_rasterize_river_guide_points_to_template(tmp_path: Path):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Point

    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    gpkg = paths["guide_points"]
    gdf = gpd.GeoDataFrame({"centerline_z_m": [5.0, 7.0]}, geometry=[Point(0.5, 3.5), Point(1.5, 2.5)], crs="EPSG:4326")
    gdf.to_file(gpkg, driver="GPKG")

    tmpl = tmp_path / "template.tif"
    with rasterio.open(tmpl, 'w', driver='GTiff', height=4, width=4, count=1, dtype='float32', crs='EPSG:4326', transform=from_origin(0, 4, 1, 1), nodata=-9999.0) as ds:
        ds.write(np.full((4,4), -9999.0, dtype='float32'), 1)

    arr = rasterize_river_guide_points_to_template(gpkg, tmpl)
    assert arr is not None
    assert float(arr[0, 0]) == 5.0
    assert float(arr[1, 1]) == 7.0


def test_rasterize_river_guide_points_to_template_uses_mixed_structured_values(tmp_path: Path):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Point

    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    gpkg = paths["guide_points"]
    gdf = gpd.GeoDataFrame(
        {
            "feature_role": ["centerline", "xs_support", "bank_point"],
            "centerline_z_m": [5.0, np.nan, np.nan],
            "xs_z_m": [np.nan, 3.0, np.nan],
            "bank_z_m": [np.nan, np.nan, 6.0],
        },
        geometry=[Point(0.5, 3.5), Point(2.5, 2.5), Point(3.5, 0.5)],
        crs="EPSG:4326",
    )
    gdf.to_file(gpkg, driver="GPKG")

    tmpl = tmp_path / "template.tif"
    with rasterio.open(tmpl, 'w', driver='GTiff', height=5, width=5, count=1, dtype='float32', crs='EPSG:4326', transform=from_origin(0, 5, 1, 1), nodata=-9999.0) as ds:
        ds.write(np.full((5,5), -9999.0, dtype='float32'), 1)

    arr = rasterize_river_guide_points_to_template(gpkg, tmpl)
    assert arr is not None
    finite = np.isfinite(arr)
    assert int(finite.sum()) >= 3
    # The XS-only row must contribute; previously a file-level centerline_z_m choice
    # left this row as NaN and the rasterized guide surface ignored it.
    assert np.isfinite(arr[2, 2])


def test_write_guidance_artifacts_with_reporting_raises_on_writer_failure(tmp_path: Path):
    import logging
    from river_guidance import write_guidance_artifacts_with_reporting

    def _writer(*args, **kwargs):
        raise RuntimeError("synthetic guidance failure")

    cfg = SimpleNamespace(out_dir=tmp_path)
    report = {}
    logger = logging.getLogger("test_river_guidance")
    with pytest.raises(RuntimeError, match="synthetic guidance failure"):
        write_guidance_artifacts_with_reporting(
            writer=_writer,
            cfg=cfg,
            out_bed=tmp_path / "bed.tif",
            out_depth=tmp_path / "depth.tif",
            channel_mask_tif=None,
            river_dir=tmp_path / "river",
            report=report,
            logger=logger,
        )
    assert report["river"]["guidance"]["artifact_write_error"] == "synthetic guidance failure"


def test_write_guidance_artifacts_does_not_discover_wse_or_legacy_xs_from_disk(tmp_path: Path, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import Point
    import logging
    from river_guidance import write_guidance_artifacts_with_reporting

    river_dir = tmp_path / "river"
    river_dir.mkdir()
    gpd.GeoDataFrame({"component_id": ["c1"], "station_m": [0.0]}, geometry=[Point(0, 0)], crs="EPSG:32619").to_file(river_dir / "centerline_points.gpkg", driver="GPKG")
    out_bed = river_dir / "bed.tif"
    out_depth = river_dir / "depth.tif"
    out_bed.write_text('x')
    out_depth.write_text('x')
    calls = {}

    def fake_writer(cfg, **kwargs):
        return {
            'centerline_points': str(river_dir / 'centerline_points.gpkg'),
            'centerline_elevation': str(river_dir / 'centerline_elevation.tif'),
            'centerline_influence': str(river_dir / 'centerline_influence.tif'),
            'centerline_stationing': str(river_dir / 'centerline_stationing.tif'),
            'xs_support_points': str(river_dir / 'xs_support_points.gpkg'),
            'xs_support_elevation': str(river_dir / 'xs_support_elevation.tif'),
            'xs_support_weight': str(river_dir / 'xs_support_weight.tif'),
            'bank_elevation_xs': str(river_dir / 'bank_elevation_xs.tif'),
            'bank_influence': str(river_dir / 'bank_influence.tif'),
            'corridor_mask': str(river_dir / 'corridor_mask.tif'),
            'authoritative_support': str(river_dir / 'authoritative_support.tif'),
            'authoritative_support_depth': str(river_dir / 'authoritative_support_depth.tif'),
        }

    def fake_profile(**kwargs):
        calls['wse_elevation_path'] = kwargs.get('wse_elevation_path')
        return {'longitudinal_profile_elevation': str(river_dir / 'lp.tif')}

    def fake_frame(**kwargs):
        return {'channel_frame_points': str(river_dir / 'frame.gpkg')}

    def fake_scaffold(**kwargs):
        calls['xs_bathy_gpkg_path'] = kwargs.get('xs_bathy_gpkg_path')
        return {'channel_scaffold_nodes': str(river_dir / 'scaffold.gpkg')}

    monkeypatch.setattr('river_guidance.build_and_write_longitudinal_profile', fake_profile)
    monkeypatch.setattr('river_guidance.build_channel_frame_products', fake_frame)
    monkeypatch.setattr('river_guidance.build_channel_scaffold_products', fake_scaffold)
    monkeypatch.setattr('river_guidance.build_channel_surface_products', lambda **kwargs: {'channel_surface': str(river_dir / 'channel_surface.tif')})
    monkeypatch.setattr('river_guidance.build_river_runtime_diagnostics', lambda **kwargs: {})
    monkeypatch.setattr('river_guidance._write_channel_surface_primary_products', lambda **kwargs: {})
    monkeypatch.setattr('river_guidance.write_river_guidance_manifest', lambda **kwargs: river_dir / 'manifest.json')
    monkeypatch.setattr('river_guidance.write_river_longitudinal_profile_contract', lambda *args, **kwargs: None)

    derived_cache_root = tmp_path / 'derived_cache' / 'river' / 'work'
    derived_cache_root.mkdir(parents=True)
    (derived_cache_root / 'river_bathy_xs_mainstem.gpkg').write_text('{}', encoding='utf-8')

    cfg = SimpleNamespace(river_method='hybrid', derived_cache_root=tmp_path / 'derived_cache', out_dir=str(tmp_path))
    report = {}
    write_guidance_artifacts_with_reporting(
        writer=fake_writer,
        cfg=cfg,
        out_bed=out_bed,
        out_depth=out_depth,
        channel_mask_tif=None,
        river_dir=river_dir,
        report=report,
        logger=logging.getLogger('test'),
    )

    receipts = report['river']['execution_receipts']
    assert receipts['legacy_xs_inputs_detected'] == []
    assert calls['xs_bathy_gpkg_path'] is None
    assert calls['wse_elevation_path'] is None


def test_load_and_validate_centerline_component_contract_rejects_stale_single_component_file(tmp_path: Path):
    import geopandas as gpd
    from shapely.geometry import Point

    gpkg = tmp_path / "river_centerline_points.gpkg"
    gdf = gpd.GeoDataFrame(
        {
            "component_id": ["main", "main"],
            "station_m": [0.0, 100.0],
        },
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:32619",
    )
    gdf.to_file(gpkg, driver="GPKG")

    with pytest.raises(RuntimeError, match="coarse_component_namespace"):
        _load_and_validate_centerline_component_contract(
            gpkg,
            expected_component_count=3,
            expected_component_source="levelpathi",
        )


def test_load_and_validate_centerline_component_contract_uses_retained_network_station_contract_metadata(tmp_path: Path):
    import geopandas as gpd
    from shapely.geometry import Point

    gpkg = tmp_path / "river_centerline_points.gpkg"
    gdf = gpd.GeoDataFrame(
        {
            "component_id": ["levelpathi:101", "levelpathi:202"],
            "station_m": [0.0, 0.0],
        },
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:32619",
    )
    gdf.to_file(gpkg, driver="GPKG")
    station_contract = tmp_path / "river_centerline_station_contract.json"
    station_contract.write_text(
        json.dumps(
            {
                "component_id_source": "levelpathi",
                "component_count_before": 1,
                "component_count_after": 2,
                "expected_component_count": 2,
                "expected_component_source": "levelpathi",
            }
        ),
        encoding="utf-8",
    )

    contract = _load_and_validate_centerline_component_contract(
        gpkg,
        retained_network_meta={"centerline_station_contract_path": str(station_contract)},
    )

    assert contract["expected_component_count"] == 2
    assert contract["component_id_source"] == "levelpathi"
    assert contract["upstream_station_contract_loaded"] is True
    assert contract["upstream_component_count_after"] == 2


def test_build_river_guidance_manifest_omits_xs_artifacts_when_xs_support_inactive(tmp_path: Path):
    from final_route_contract import validate_guidance_manifest, _validate_manifest_outputs_present

    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    for key in (
        "guidance_weight",
        "trusted_interior",
        "soft_guidance_domain",
        "admissibility",
        "guide_points",
        "corridor_mask",
        "centerline_points",
        "centerline_elevation",
        "centerline_influence",
        "centerline_stationing",
        "bank_influence",
        "bank_elevation_xs",
        "bank_continuity_weight",
        "bank_graph_confidence",
        "bank_confluence_damping",
        "bank_estuary_side_decay",
        "authoritative_support",
        "authoritative_support_depth",
    ):
        pth = paths[key]
        if pth.suffix == ".gpkg":
            pth.write_text("{}", encoding="utf-8")
        else:
            pth.write_bytes(b"x")

    report = {
        "river": {
            "outputs": {
                "guidance_weight": str(paths["guidance_weight"]),
                "trusted_interior": str(paths["trusted_interior"]),
                "soft_guidance_domain": str(paths["soft_guidance_domain"]),
                "admissibility": str(paths["admissibility"]),
                "guide_points": str(paths["guide_points"]),
                "corridor_mask": str(paths["corridor_mask"]),
                "centerline_points": str(paths["centerline_points"]),
                "centerline_elevation": str(paths["centerline_elevation"]),
                "centerline_influence": str(paths["centerline_influence"]),
                "centerline_stationing": str(paths["centerline_stationing"]),
                "bank_influence": str(paths["bank_influence"]),
                "bank_elevation_xs": str(paths["bank_elevation_xs"]),
                "bank_continuity_weight": str(paths["bank_continuity_weight"]),
                "bank_graph_confidence": str(paths["bank_graph_confidence"]),
                "bank_confluence_damping": str(paths["bank_confluence_damping"]),
                "bank_estuary_side_decay": str(paths["bank_estuary_side_decay"]),
                "authoritative_support": str(paths["authoritative_support"]),
                "authoritative_support_depth": str(paths["authoritative_support_depth"]),
            },
            "retained_network": {
                "xs_support_point_count": 0,
                "xs_support_contract": {"status": "empty_after_sampling"},
            },
            "execution_receipts": {"legacy_xs_inputs_used": False},
            "guidance": {},
        }
    }
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    allowed = set(manifest["final_route_contract"]["allowed_structural_artifacts"])
    assert "xs_support_points" not in allowed
    assert "xs_support_elevation" not in allowed
    assert "xs_support_weight" not in allowed
    assert manifest["xs_support_status"]["structural_artifacts_active"] is False
    validation = validate_guidance_manifest(manifest=manifest, family="river_guidance")
    assert validation["valid"] is True
    presence = _validate_manifest_outputs_present(manifest=manifest, report=report, family="river_guidance")
    assert "xs_support_points" not in presence["missing_declared_artifacts"]
    assert "xs_support_elevation" not in presence["missing_critical_artifacts"]


def test_build_river_guidance_manifest_reports_canonical_artifacts_missing_from_outputs(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    paths["guide_points"].write_text("{}", encoding="utf-8")
    paths["centerline_points"].write_text("{}", encoding="utf-8")

    report = {"river": {"outputs": {"guide_points": str(paths["guide_points"])}, "guidance": {}}}
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    assert manifest["artifacts"]["guide_points"].endswith("river_guide_points.gpkg")
    assert "centerline_points" not in manifest["artifacts"]
    assert "centerline_points" in manifest.get("diagnostics", {}).get("output_contract_missing_from_outputs", [])


def _write_contract(path: Path, metrics: dict):
    path.write_text(json.dumps({"metrics": metrics}, indent=2), encoding="utf-8")
    return str(path)


def test_build_river_guidance_manifest_marks_xs_structural_only_when_artifacts_exist_without_bed_support(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    for key in ("xs_support_points", "xs_support_elevation", "xs_support_weight"):
        pth = paths[key]
        if pth.suffix == ".gpkg":
            pth.write_text("{}", encoding="utf-8")
        else:
            pth.write_bytes(b"x")
    report = {
        "river": {
            "outputs": {
                "xs_support_points": str(paths["xs_support_points"]),
                "xs_support_elevation": str(paths["xs_support_elevation"]),
                "xs_support_weight": str(paths["xs_support_weight"]),
            },
            "retained_network": {
                "xs_support_point_count": 0,
                "xs_support_contract": {"status": "empty_after_sampling"},
            },
            "execution_receipts": {"legacy_xs_inputs_used": False},
            "guidance": {},
        }
    }
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    assert manifest["xs_support_status"]["status"] == "structural_only"
    assert manifest["xs_support_status"]["bed_support_active"] is False
    assert manifest["xs_support_status"]["structural_artifacts_active"] is True
    assert "xs_support_points" in manifest["final_route_contract"]["optional_structural_artifacts"]
    assert "xs_support_points" not in manifest["final_route_contract"]["required_structural_artifacts"]


def test_build_river_guidance_manifest_marks_xs_bed_support_active_from_downstream_contracts(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    for key in ("xs_support_points", "xs_support_elevation", "xs_support_weight"):
        pth = paths[key]
        if pth.suffix == ".gpkg":
            pth.write_text("{}", encoding="utf-8")
        else:
            pth.write_bytes(b"x")
    frame_contract = _write_contract(tmp_path / "frame_contract.json", {
        "support_class_counts": {"xs_supported": 0},
        "authoritative_xs_anchor_count": 0,
    })
    scaffold_contract = _write_contract(tmp_path / "scaffold_contract.json", {
        "xs_profile_resampled_node_count": 8,
        "xs_only_station_count": 2,
    })
    surface_contract = _write_contract(tmp_path / "surface_contract.json", {
        "xs_profile_cells": 5,
    })
    report = {
        "river": {
            "outputs": {
                "xs_support_points": str(paths["xs_support_points"]),
                "xs_support_elevation": str(paths["xs_support_elevation"]),
                "xs_support_weight": str(paths["xs_support_weight"]),
                "channel_frame_contract": frame_contract,
                "channel_scaffold_contract": scaffold_contract,
                "channel_surface_contract": surface_contract,
            },
            "retained_network": {
                "xs_support_point_count": 0,
                "xs_support_contract": {"status": "empty_after_sampling"},
            },
            "execution_receipts": {"legacy_xs_inputs_used": False},
            "guidance": {},
        }
    }
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    assert manifest["xs_support_status"]["status"] == "active_bed_support"
    assert manifest["xs_support_status"]["bed_support_active"] is True
    assert manifest["xs_support_status"]["surface_xs_profile_cell_count"] == 5
    assert "xs_support_points" in manifest["final_route_contract"]["required_structural_artifacts"]


def test_build_river_guidance_manifest_keeps_scaffold_only_xs_as_structural_only(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    for key in ("xs_support_points", "xs_support_elevation", "xs_support_weight"):
        pth = paths[key]
        if pth.suffix == ".gpkg":
            pth.write_text("{}", encoding="utf-8")
        else:
            pth.write_bytes(b"x")
    scaffold_contract = _write_contract(tmp_path / "scaffold_contract.json", {
        "xs_profile_resampled_node_count": 11,
        "xs_only_station_count": 3,
    })
    surface_contract = _write_contract(tmp_path / "surface_contract.json", {
        "xs_profile_cells": 0,
    })
    frame_contract = _write_contract(tmp_path / "frame_contract.json", {
        "support_class_counts": {"xs_supported": 0},
        "authoritative_xs_anchor_count": 0,
        "xs_candidate_station_count": 0,
    })
    report = {
        "river": {
            "outputs": {
                "xs_support_points": str(paths["xs_support_points"]),
                "xs_support_elevation": str(paths["xs_support_elevation"]),
                "xs_support_weight": str(paths["xs_support_weight"]),
                "channel_frame_contract": frame_contract,
                "channel_scaffold_contract": scaffold_contract,
                "channel_surface_contract": surface_contract,
            },
            "retained_network": {
                "xs_support_point_count": 0,
                "xs_support_contract": {"status": "empty_after_sampling", "authoritative_sample_count": 0},
            },
            "execution_receipts": {"legacy_xs_inputs_used": False},
            "guidance": {},
        }
    }
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    assert manifest["xs_support_status"]["status"] == "structural_only"
    assert manifest["xs_support_status"]["bed_support_active"] is False
    assert manifest["xs_support_status"]["scaffold_xs_profile_node_count"] == 11
    assert "xs_support_points" in manifest["final_route_contract"]["optional_structural_artifacts"]
    assert "xs_support_points" not in manifest["final_route_contract"]["required_structural_artifacts"]


def test_build_river_guidance_manifest_resolves_relative_output_paths_against_out_root(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    frame = river_dir / "river_channel_frame_points.gpkg"
    frame.write_text("{}", encoding="utf-8")
    report = {
        "river": {
            "outputs": {
                "channel_frame_points": str(Path("river") / "river_channel_frame_points.gpkg"),
            }
        }
    }
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    assert manifest["artifacts"]["channel_frame_points"] == str(Path("river") / "river_channel_frame_points.gpkg")


def test_rasterize_river_guide_points_to_template_reprojects_with_pyproj_crs(tmp_path: Path, monkeypatch):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Point
    import xs_infer_bathy_raster

    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    gpkg = paths["guide_points"]
    gdf = gpd.GeoDataFrame({"centerline_z_m": [5.0]}, geometry=[Point(-71.0, 42.75)], crs="EPSG:4326")
    gdf.to_file(gpkg, driver="GPKG")

    tmpl = tmp_path / "template_3857.tif"
    with rasterio.open(
        tmpl,
        'w',
        driver='GTiff',
        height=4,
        width=4,
        count=1,
        dtype='float32',
        crs='EPSG:3857',
        transform=from_origin(-7905000.0, 5265000.0, 1000.0, 1000.0),
        nodata=-9999.0,
    ) as ds:
        ds.write(np.full((4, 4), -9999.0, dtype='float32'), 1)

    def fake_continuous_surface(*, pts_gdf, value_col, template_ds, **kwargs):
        assert pts_gdf.crs is not None
        assert pts_gdf.crs.to_epsg() == 3857
        arr = np.full((template_ds.height, template_ds.width), -9999.0, dtype=np.float32)
        arr[1, 1] = np.float32(5.0)
        mask = np.zeros((template_ds.height, template_ds.width), dtype=np.uint8)
        mask[1, 1] = 1
        return arr, mask

    monkeypatch.setattr(xs_infer_bathy_raster, '_continuous_surface', fake_continuous_surface)
    arr = rasterize_river_guide_points_to_template(gpkg, tmpl)
    assert arr is not None
    assert float(arr[1, 1]) == 5.0


def test_derive_xs_support_status_reports_disabled_by_option():
    report = {"river": {"execution_receipts": {"xs_influence_disabled": True}}}
    status = _derive_xs_support_status(report, {})
    assert status['status'] == 'disabled_by_option'
    assert status['bed_support_active'] is False
    assert status['structural_artifacts_active'] is False


def test_rasterize_river_guide_points_to_template_skips_nonfinite_geometry(tmp_path: Path):
    import geopandas as gpd
    import rasterio
    import warnings
    from rasterio.transform import from_origin
    from shapely.geometry import Point

    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    gpkg = paths["guide_points"]
    gdf = gpd.GeoDataFrame(
        {"centerline_z_m": [5.0, 7.0]},
        geometry=[Point(float("nan"), 3.5), Point(1.5, 2.5)],
        crs="EPSG:4326",
    )
    gdf.to_file(gpkg, driver="GPKG")

    tmpl = tmp_path / "template.tif"
    with rasterio.open(tmpl, 'w', driver='GTiff', height=4, width=4, count=1, dtype='float32', crs='EPSG:4326', transform=from_origin(0, 4, 1, 1), nodata=-9999.0) as ds:
        ds.write(np.full((4,4), -9999.0, dtype='float32'), 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arr = rasterize_river_guide_points_to_template(gpkg, tmpl)
    assert arr is not None
    assert all("invalid value encountered in cast" not in str(w.message) for w in caught)
    finite_vals = arr[np.isfinite(arr)]
    assert finite_vals.size > 0
    assert np.allclose(finite_vals, 7.0)
