from pathlib import Path
from types import SimpleNamespace
import logging

import river_guidance



def test_structured_mode_skips_legacy_xs_rebuild_and_scaffold_input(tmp_path, monkeypatch):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    import geopandas as gpd
    from shapely.geometry import Point
    gpd.GeoDataFrame({"component_id": ["c1"], "station_m": [0.0]}, geometry=[Point(0, 0)], crs="EPSG:32619").to_file(river_dir / 'centerline_points.gpkg', driver='GPKG')
    out_bed = river_dir / 'bed.tif'
    out_depth = river_dir / 'depth.tif'
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
        return {'longitudinal_profile_elevation': str(river_dir / 'lp.tif')}

    def fake_frame(**kwargs):
        return {'channel_frame_points': str(river_dir / 'frame.gpkg')}

    def fake_scaffold(**kwargs):
        calls['xs_bathy_gpkg_path'] = kwargs.get('xs_bathy_gpkg_path')
        return {'channel_scaffold_nodes': str(river_dir / 'scaffold.gpkg')}

    def fake_surface(**kwargs):
        return {'channel_surface': str(river_dir / 'channel_surface.tif')}

    monkeypatch.setattr(river_guidance, 'build_and_write_longitudinal_profile', fake_profile)
    monkeypatch.setattr(river_guidance, 'build_channel_frame_products', fake_frame)
    monkeypatch.setattr(river_guidance, 'build_channel_scaffold_products', fake_scaffold)
    monkeypatch.setattr(river_guidance, 'build_channel_surface_products', fake_surface)
    monkeypatch.setattr(river_guidance, 'build_river_runtime_diagnostics', lambda **kwargs: {})
    monkeypatch.setattr(river_guidance, '_write_channel_surface_primary_products', lambda **kwargs: {})
    monkeypatch.setattr(river_guidance, 'write_river_guidance_manifest', lambda **kwargs: river_dir / 'manifest.json')
    monkeypatch.setattr(river_guidance, 'write_river_longitudinal_profile_contract', lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        river_method='structured',
        river_allow_absolute_bed_fallback=False,
        derived_cache_root=tmp_path / 'derived_cache',
        out_dir=str(tmp_path),
    )
    report = {}

    river_guidance.write_guidance_artifacts_with_reporting(
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
    assert receipts['legacy_xs_inputs_used'] is False
    assert receipts['legacy_xs_inputs_detected'] == []
    assert receipts['legacy_xs_inputs_permitted'] is False
    assert receipts['legacy_xs_inputs_blocked'] is True
    assert calls['xs_bathy_gpkg_path'] is None


def test_hybrid_mode_blocks_derived_cache_legacy_xs_path_in_structured_guidance_route(tmp_path, monkeypatch):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    import geopandas as gpd
    from shapely.geometry import Point
    gpd.GeoDataFrame({"component_id": ["c1"], "station_m": [0.0]}, geometry=[Point(0, 0)], crs="EPSG:32619").to_file(river_dir / 'centerline_points.gpkg', driver='GPKG')
    out_bed = river_dir / 'bed.tif'
    out_depth = river_dir / 'depth.tif'
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
        return {'longitudinal_profile_elevation': str(river_dir / 'lp.tif')}

    def fake_frame(**kwargs):
        return {'channel_frame_points': str(river_dir / 'frame.gpkg')}

    def fake_scaffold(**kwargs):
        calls['xs_bathy_gpkg_path'] = kwargs.get('xs_bathy_gpkg_path')
        return {'channel_scaffold_nodes': str(river_dir / 'scaffold.gpkg')}

    monkeypatch.setattr(river_guidance, 'build_and_write_longitudinal_profile', fake_profile)
    monkeypatch.setattr(river_guidance, 'build_channel_frame_products', fake_frame)
    monkeypatch.setattr(river_guidance, 'build_channel_scaffold_products', fake_scaffold)
    monkeypatch.setattr(river_guidance, 'build_channel_surface_products', lambda **kwargs: {'channel_surface': str(river_dir / 'channel_surface.tif')})
    monkeypatch.setattr(river_guidance, 'build_river_runtime_diagnostics', lambda **kwargs: {})
    monkeypatch.setattr(river_guidance, '_write_channel_surface_primary_products', lambda **kwargs: {})
    monkeypatch.setattr(river_guidance, 'write_river_guidance_manifest', lambda **kwargs: river_dir / 'manifest.json')
    monkeypatch.setattr(river_guidance, 'write_river_longitudinal_profile_contract', lambda *args, **kwargs: None)

    legacy_xs = tmp_path / 'derived_cache' / 'river' / 'work' / 'river_bathy_xs_mainstem.gpkg'
    legacy_xs.parent.mkdir(parents=True)
    legacy_xs.write_text('{}', encoding='utf-8')

    cfg = SimpleNamespace(
        river_method='hybrid',
        river_allow_absolute_bed_fallback=True,
        derived_cache_root=tmp_path / 'derived_cache',
        out_dir=str(tmp_path),
    )
    report = {}

    river_guidance.write_guidance_artifacts_with_reporting(
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
    assert receipts['legacy_xs_inputs_permitted'] is False
    assert receipts['legacy_xs_inputs_blocked'] is True
    assert receipts['legacy_xs_profile_rebuild_removed'] is True
    assert receipts['legacy_xs_inputs_used'] is False
    assert receipts['legacy_xs_inputs_detected'] == []
    assert calls['xs_bathy_gpkg_path'] is None

