from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from river_execution_plan import determine_river_execution_plan
from sdb_execution_plan import determine_sdb_execution_plan


def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_determine_sdb_execution_plan_uses_validated_shared_domain_mask(tmp_path):
    sdb_mask = tmp_path / "sdb_guidance_domain_mask.tif"
    sdb_mask.write_bytes(b"mask")
    cfg = SimpleNamespace(
        out_dir=tmp_path / "out",
        aoi="0/1/2/3",
        sdb_guidance_domain_mask=str(sdb_mask),
        validated_sdb_guidance_domain_mask=str(sdb_mask),
        validated_sdb_guidance_domain_pixels=42,
    )
    report = {"guidance_domains": {"activation": {"derived_activation": {"sdb_should_run": True}}}}

    plan = determine_sdb_execution_plan(
        cfg=cfg,
        report=report,
        ensure_dir_fn=_ensure_dir,
        parse_aoi_bbox_fn=lambda _: (0, 1, 2, 3),
        count_mask_water_pixels_fn=lambda *_: 0,
        logger=SimpleNamespace(info=lambda *a, **k: None),
    )

    assert plan.should_run is True
    assert plan.shared_domain_pixels == 42
    assert report["sdb"]["execution_plan"]["shared_domain_pixels"] == 42


def test_determine_sdb_execution_plan_skips_when_shared_domain_empty(tmp_path):
    cfg = SimpleNamespace(
        out_dir=tmp_path / "out",
        aoi="0/1/2/3",
        sdb_guidance_domain_mask=None,
        validated_sdb_guidance_domain_mask=None,
        validated_sdb_guidance_domain_pixels=None,
    )
    report = {"guidance_domains": {"activation": {"derived_activation": {"sdb_should_run": False}}}}

    plan = determine_sdb_execution_plan(
        cfg=cfg,
        report=report,
        ensure_dir_fn=_ensure_dir,
        parse_aoi_bbox_fn=lambda _: None,
        count_mask_water_pixels_fn=lambda *_: 0,
        logger=SimpleNamespace(info=lambda *a, **k: None),
    )

    assert plan.should_run is False
    assert plan.skip_reason == "shared_domain_empty"
    assert report["sdb"]["skipped_reason"] == "shared_domain_empty"


def test_determine_river_execution_plan_applies_channel_template_override(tmp_path):
    cfg = SimpleNamespace(
        out_dir=tmp_path / "out",
        derived_cache_root=tmp_path / "derived",
        cache_root=tmp_path / "cache",
        run_id="run1",
        aoi_tile="0/1/2/3",
        aoi="0/1/2/3",
        start_date="2025-01-01",
        end_date="2025-12-31",
        working_srs="EPSG:32619",
        river_method="skeleton",
        river_dem_source="authoritative",
        extra_xyz_cudem=True,
        river_channel_template_enabled=True,
    )
    report = {}

    plan = determine_river_execution_plan(
        cfg=cfg,
        report=report,
        ensure_dir_fn=_ensure_dir,
        normalize_channel_template_setting_fn=lambda *_: None,
        logger=SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None),
    )

    assert plan.requested_method == "skeleton"
    assert plan.effective_method == "hybrid"
    assert report["river"]["execution_plan"]["effective_method"] == "hybrid"
    assert report["river"]["notes"]["channel_template_method_override"]["effective_method"] == "hybrid"
