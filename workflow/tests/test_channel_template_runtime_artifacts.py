import json
from pathlib import Path

import pandas as pd

from xs_infer_bathy_raster import _write_channel_template_runtime_artifacts


def test_runtime_artifacts_written_on_exception(tmp_path):
    summary = {
        "enabled": True,
        "built": False,
        "selected_profiles": 0,
        "raw_profiles": 0,
        "dense_points_written": 0,
        "dense_points_path": None,
        "dense_points_expected_path": str(tmp_path / "channel_template_dense_points.gpkg"),
        "write_error": "build_exception: RuntimeError: boom",
        "stage": "build_exception",
    }
    summary_path, diagnostics_path, filter_path = _write_channel_template_runtime_artifacts(
        tmp_path, summary, diagnostics=None, filter_info=None
    )
    assert summary_path.exists()
    assert diagnostics_path.exists()
    assert filter_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["stage"] == "build_exception"
    diag = pd.read_csv(diagnostics_path)
    assert len(diag) == 1
    assert diag.loc[0, "template_final_reject_reason"].startswith("build_exception")


def test_append_channel_template_args_adds_runtime_flags(tmp_path):
    import bathy_main

    cfg = type('Cfg', (), {
        'river_channel_template_enabled': True,
        'river_channel_template_min_xs': 3,
        'river_channel_template_fit_min_xs': 5,
        'river_channel_template_n_bins': 64,
        'river_channel_template_min_depth_m': 0.4,
        'river_channel_template_distance_sigma_m': 1500.0,
        'river_channel_template_estuary_buffer_m': 600.0,
        'river_channel_template_junction_buffer_m': 90.0,
        'river_channel_template_width_depth_ratio_max': 120.0,
        'river_channel_template_loo_max_rmse_norm': 0.25,
        'river_channel_template_loo_max_dmax_error_m': 1.5,
    })()
    cmd = ['python', 'xs_infer_bathy_raster.py']
    river_dir = tmp_path / 'river'
    out_dir = bathy_main._append_channel_template_args(cmd, cfg, river_dir)
    assert out_dir == river_dir / 'channel_template'
    assert '--channel-template-enabled' in cmd
    assert f'--channel-template-out-dir={out_dir}' in cmd
    assert '--channel-template-min-xs=3' in cmd
    assert '--channel-template-fit-min-xs=5' in cmd
    assert '--channel-template-n-bins=64' in cmd
    assert '--channel-template-min-depth-m=0.4' in cmd
    assert '--channel-template-distance-sigma-m=1500.0' in cmd
    assert '--channel-template-estuary-buffer-m=600.0' in cmd
    assert '--channel-template-junction-buffer-m=90.0' in cmd
    assert '--channel-template-width-depth-ratio-max=120.0' in cmd


def test_resolve_channel_template_artifacts_prefers_river_dir(tmp_path):
    import bathy_main

    work_dir = tmp_path / "work"
    river_dir = tmp_path / "river"
    cache_dir = tmp_path / "cache"
    (river_dir / "channel_template").mkdir(parents=True)
    work_dir.mkdir(parents=True)
    (cache_dir / "river" / "channel_template").mkdir(parents=True)

    river_summary = river_dir / "channel_template" / "channel_template_runtime_summary.json"
    river_summary.write_text('{"built": true}', encoding='utf-8')
    river_template = river_dir / "channel_template" / "channel_template.json"
    river_template.write_text('{}', encoding='utf-8')

    cfg = type('Cfg', (), {
        'derived_cache_root': str(cache_dir),
        'aoi': 'aoi',
        'river_dem': 'dem.tif',
        'river_method': 'hybrid',
    })()

    resolved = bathy_main._resolve_channel_template_artifacts(cfg, work_dir=work_dir, river_dir=river_dir)
    assert resolved['summary'] == river_summary
    assert resolved['template_json'] == river_template
    assert resolved['dir'] == river_dir / 'channel_template'


def test_channel_template_cache_key_is_stable():
    import bathy_main

    cfg1 = type('Cfg', (), {'aoi': 'aoi', 'river_dem': 'dem.tif', 'river_method': 'hybrid'})()
    cfg2 = type('Cfg', (), {'aoi': 'aoi', 'river_dem': 'dem.tif', 'river_method': 'hybrid'})()
    cfg3 = type('Cfg', (), {'aoi': 'aoi2', 'river_dem': 'dem.tif', 'river_method': 'hybrid'})()

    assert bathy_main._channel_template_cache_key(cfg1) == bathy_main._channel_template_cache_key(cfg2)
    assert bathy_main._channel_template_cache_key(cfg1) != bathy_main._channel_template_cache_key(cfg3)


def test_load_channel_template_json_prefers_resolved_primary_dir(tmp_path):
    import bathy_main

    river_dir = tmp_path / "river"
    work_dir = tmp_path / "work"
    (river_dir / "channel_template").mkdir(parents=True)
    work_dir.mkdir(parents=True)
    template_path = river_dir / "channel_template" / "channel_template.json"
    template_path.write_text('{"depth_fit_source": "local_power_law", "depth_a": 1.0}', encoding='utf-8')

    cfg = type('Cfg', (), {
        'derived_cache_root': None,
        'aoi': 'aoi',
        'river_dem': 'dem.tif',
        'river_method': 'hybrid',
    })()

    resolved_path, payload = bathy_main._load_channel_template_json(cfg, work_dir=work_dir, river_dir=river_dir)
    assert resolved_path == template_path
    assert payload['depth_fit_source'] == 'local_power_law'


def test_resolve_channel_template_artifacts_keeps_bundle_dir_coherent(tmp_path):
    import bathy_main

    river_dir = tmp_path / "river"
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    primary = river_dir / "channel_template"
    fallback = work_dir / "channel_template"
    primary.mkdir(parents=True)
    fallback.mkdir(parents=True)
    (primary / "channel_template_runtime_summary.json").write_text('{"built": true}', encoding='utf-8')
    (fallback / "channel_template_dense_points.gpkg").write_text('placeholder', encoding='utf-8')
    (fallback / "channel_template.json").write_text('{"depth_fit_source": "local_power_law"}', encoding='utf-8')

    cfg = type('Cfg', (), {
        'derived_cache_root': str(cache_dir),
        'aoi': 'aoi',
        'river_dem': 'dem.tif',
        'river_method': 'hybrid',
    })()

    resolved = bathy_main._resolve_channel_template_artifacts(cfg, work_dir=work_dir, river_dir=river_dir)
    assert resolved['dir'] == primary
    assert resolved['summary'] == primary / 'channel_template_runtime_summary.json'
    assert resolved['dense_points'] == primary / 'channel_template_dense_points.gpkg'
    assert resolved['template_json'] == primary / 'channel_template.json'


def test_channel_template_cache_key_changes_when_template_controls_change():
    import bathy_main

    base = {
        'aoi': 'aoi',
        'river_dem': 'dem.tif',
        'river_method': 'hybrid',
        'xs_spacing_m': 50.0,
        'river_channel_template_min_xs': 3,
        'river_channel_template_fit_min_xs': 5,
        'river_channel_template_n_bins': 50,
        'river_channel_template_min_depth_m': 0.3,
        'river_channel_template_distance_sigma_m': 2000.0,
        'river_channel_template_estuary_buffer_m': 500.0,
        'river_channel_template_junction_buffer_m': 120.0,
        'river_channel_template_width_depth_ratio_max': None,
        'river_channel_template_loo_max_rmse_norm': 0.25,
        'river_channel_template_loo_max_dmax_error_m': 1.5,
        'river_shape_exp': 2.0,
        'river_mv_a0': 1.0,
        'river_mv_bw': 0.5,
    }
    cfg1 = type('Cfg', (), dict(base))()
    cfg2 = type('Cfg', (), dict(base, river_channel_template_min_xs=4))()
    cfg3 = type('Cfg', (), dict(base, xs_spacing_m=25.0))()

    cfg4 = type('Cfg', (), dict(base, river_channel_template_distance_sigma_m=1000.0))()
    key1 = bathy_main._channel_template_cache_key(cfg1)
    assert key1 != bathy_main._channel_template_cache_key(cfg2)
    assert key1 != bathy_main._channel_template_cache_key(cfg3)
    assert key1 != bathy_main._channel_template_cache_key(cfg4)


def test_channel_template_output_entries_are_gated_by_enable_flag(tmp_path):
    import bathy_main

    dense = tmp_path / "channel_template_dense_points.gpkg"
    summary = tmp_path / "channel_template_runtime_summary.json"
    dense.write_text("x", encoding="utf-8")
    summary.write_text("{}", encoding="utf-8")

    disabled_cfg = type('Cfg', (), {'river_channel_template_enabled': False})()
    enabled_cfg = type('Cfg', (), {'river_channel_template_enabled': True})()

    disabled = bathy_main._channel_template_output_entries(
        disabled_cfg,
        dense_points_path=dense,
        runtime_summary_path=summary,
    )
    enabled = bathy_main._channel_template_output_entries(
        enabled_cfg,
        dense_points_path=dense,
        runtime_summary_path=summary,
    )

    assert disabled['channel_template_dense_points'] is None
    assert disabled['channel_template_runtime_summary'] is None
    assert enabled['channel_template_dense_points'] == str(dense)
    assert enabled['channel_template_runtime_summary'] == str(summary)


def test_channel_template_runtime_dir_is_explicit(tmp_path):
    import bathy_main

    river_dir = tmp_path / 'river'
    assert bathy_main._channel_template_runtime_dir(river_dir) == river_dir / 'channel_template'


def test_runtime_artifacts_preserve_existing_diagnostics_and_filter_on_late_summary_update(tmp_path):
    diagnostics = pd.DataFrame([
        {
            "xs_id": "xs_001",
            "keep_for_template": True,
            "template_mean_usable": True,
            "template_selected": True,
            "template_final_reject_reason": "",
        },
        {
            "xs_id": "xs_002",
            "keep_for_template": False,
            "template_mean_usable": False,
            "template_selected": False,
            "template_final_reject_reason": "too_few_bins",
        },
    ])
    initial_summary = {
        "enabled": True,
        "built": True,
        "selected_profiles": 1,
        "raw_profiles": 2,
        "dense_points_written": 0,
        "dense_points_path": None,
        "dense_points_expected_path": str(tmp_path / "channel_template_dense_points.gpkg"),
        "write_error": None,
        "stage": "post_build",
    }
    _write_channel_template_runtime_artifacts(
        tmp_path,
        initial_summary,
        diagnostics=diagnostics,
        filter_info={"kept": 1, "rejected": 1},
    )

    late_summary = dict(initial_summary, stage="post_dense_export", dense_points_written=5)
    _, diagnostics_path, filter_path = _write_channel_template_runtime_artifacts(
        tmp_path,
        late_summary,
        diagnostics=None,
        filter_info=None,
    )

    diag = pd.read_csv(diagnostics_path)
    assert len(diag) == 2
    assert set(diag["xs_id"].astype(str)) == {"xs_001", "xs_002"}
    filter_payload = json.loads(filter_path.read_text())
    assert filter_payload["kept"] == 1
    assert filter_payload["rejected"] == 1


def test_channel_template_summary_payload_includes_measurement_and_fit_fields():
    from xs_infer_bathy_raster import _channel_template_summary_payload

    payload = _channel_template_summary_payload(
        enabled=True,
        built=True,
        selected_profiles=4,
        raw_profiles=7,
        measurement_source="authoritative_dem",
        measurement_profile_count=5,
        dense_points_written=11,
        dense_points_path=Path("/tmp/dense.gpkg"),
        dense_points_expected_path=Path("/tmp/dense.gpkg"),
        write_error=None,
        stage="post_build",
        depth_fit_source="local_power_law",
        depth_fit_r2=0.91,
        loo_gate={"pass": True},
    )

    assert payload["measurement_source"] == "authoritative_dem"
    assert payload["measurement_profile_count"] == 5
    assert payload["depth_fit_source"] == "local_power_law"
    assert payload["depth_fit_r2"] == 0.91
    assert payload["loo_gate"] == {"pass": True}


def test_validate_channel_template_runtime_summary_rejects_missing_dense_file(tmp_path):
    import bathy_main

    summary_path = tmp_path / "channel_template_runtime_summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    summary = {
        "built": True,
        "dense_points_written": 3,
        "dense_points_path": str(tmp_path / "missing_dense_points.gpkg"),
    }

    try:
        bathy_main._validate_channel_template_runtime_summary(summary_path, summary)
    except RuntimeError as exc:
        assert "file does not exist" in str(exc)
    else:
        raise AssertionError("Expected missing dense file to raise")


def test_normalize_channel_template_setting_preserves_false_request():
    import bathy_main

    cfg = type('Cfg', (), {'river_channel_template_enabled': False})()
    report = {'river': {}}
    requested = bathy_main._normalize_channel_template_setting(cfg, report)

    assert requested is False
    assert cfg.river_channel_template_enabled is False
    note = report['river']['notes']['channel_template_enablement']
    assert note['effective_enabled'] is False
    assert note['requested_enabled'] is False


def test_append_channel_template_args_skips_enable_flag_when_disabled(tmp_path):
    import bathy_main

    cfg = type('Cfg', (), {
        'river_channel_template_enabled': False,
        'river_channel_template_min_xs': 3,
        'river_channel_template_fit_min_xs': 5,
        'river_channel_template_n_bins': 64,
        'river_channel_template_min_depth_m': 0.4,
        'river_channel_template_distance_sigma_m': 1500.0,
        'river_channel_template_estuary_buffer_m': 600.0,
        'river_channel_template_junction_buffer_m': 90.0,
        'river_channel_template_width_depth_ratio_max': 120.0,
        'river_channel_template_loo_max_rmse_norm': 0.25,
        'river_channel_template_loo_max_dmax_error_m': 1.5,
    })()
    cmd = ['python', 'xs_infer_bathy_raster.py']
    river_dir = tmp_path / 'river'
    out_dir = bathy_main._append_channel_template_args(cmd, cfg, river_dir)

    assert out_dir is None
    assert '--channel-template-enabled' not in cmd
    assert '--no-channel-template' in cmd


def test_normalize_channel_template_setting_records_shared_attrs_without_forcing():
    import bathy_main

    cfg = type('Cfg', (), {'river_channel_template_enabled': False})()
    report = {'river': {}}
    requested = bathy_main._normalize_channel_template_setting(cfg, report)

    assert requested is False
    assert cfg.river_channel_template_enabled is False
    assert cfg.river_channel_template_requested is False
    assert cfg.river_channel_template_forced_enabled is False
    note = report['river']['notes']['channel_template_enablement']
    assert note['requested_attr_recorded'] is True
    assert note['forced_attr_recorded'] is True
    assert note['reason'] is None
