from pathlib import Path

from final_run_stage import execute_final_run_stage


class DummyCfg:
    authoritative_base = "/tmp/auth.tif"
    gapfill_enabled = False
    out_dir = Path("/tmp/out")


class DummyArgs:
    pass


def test_support_aware_direct_mode_skips_legacy_fusion():
    called = {"fuse": 0, "condition": 0, "finalize": 0}
    report = {}

    def fuse_fn(*args, **kwargs):
        called["fuse"] += 1
        raise AssertionError("legacy fuse should be skipped in direct mode")

    def condition_fn(cfg, final, provenance, report):
        called["condition"] += 1
        assert final is None
        assert provenance is None
        report.setdefault("authoritative_base", {})["status"] = "applied"
        return Path("/tmp/final.tif"), Path("/tmp/prov.tif"), None, None, None, None

    def reproject_fn(cfg, final, report, sdb_raster, river_raster, fatal_errors):
        return Path("/tmp/final_epsg4269.tif")

    def write_bundle_fn(cfg, report, final_native, final_for_user, final_provenance):
        return Path("/tmp/report.json")

    def finalize_run_fn(cfg, report, args, final, final_for_user, final_provenance, fatal_errors):
        called["finalize"] += 1
        assert report["fusion"]["status"] == "skipped"
        assert report["final_dem_runtime"]["final_generation_route"] == "support_aware_direct_sources"
        return 0

    rc = execute_final_run_stage(
        cfg=DummyCfg(),
        args=DummyArgs(),
        report=report,
        log=type("L", (), {"info": lambda *a, **k: None})(),
        fatal_errors=[],
        sdb_raster=Path("/tmp/sdb.tif"),
        river_raster=Path("/tmp/river.tif"),
        river_for_fuse=Path("/tmp/river.tif"),
        river_excluded=None,
        fuse_fn=fuse_fn,
        condition_fn=condition_fn,
        reproject_fn=reproject_fn,
        write_bundle_fn=write_bundle_fn,
        finalize_run_fn=finalize_run_fn,
        write_io_manifest_fn=lambda *a, **k: (None, None),
        emit_artifacts_fn=lambda *a, **k: None,
        run_seam_comparisons_fn=lambda *a, **k: None,
        conditioned_gapfill_fn=lambda **kwargs: (_ for _ in ()).throw(AssertionError("gapfill should be skipped")),
    )

    assert rc == 0
    assert called["fuse"] == 0
    assert called["condition"] == 1
    assert called["finalize"] == 1
