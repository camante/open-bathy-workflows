from pathlib import Path

from final_route_inputs_stage import collect_final_route_inputs
from legacy_cleanup_stage import build_legacy_cleanup_summary


class _Cfg:
    def __init__(self, out_dir, authoritative_base):
        self.out_dir = Path(out_dir)
        self.authoritative_base = Path(authoritative_base)


def test_phase_k_template_selection_ignores_legacy_candidate_when_structural_exists(tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    auth = out_dir / "auth.tif"
    auth.write_bytes(b"x")
    sdb_dir = out_dir / "sdb"
    sdb_dir.mkdir()
    depth = sdb_dir / "depth.tif"
    depth.write_bytes(b"y")
    (sdb_dir / "artifacts_sdb.json").write_text('{"depth_raster": "depth.tif"}', encoding="utf-8")
    legacy = out_dir / "legacy_candidate.tif"
    legacy.write_bytes(b"z")
    cfg = _Cfg(out_dir=out_dir, authoritative_base=auth)
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=legacy, report={})
    assert paths is not None
    assert paths.template_path == auth


def test_phase_k_legacy_cleanup_detects_no_structural_leakage():
    report = {
        "fusion": {"outputs": {"depth": "/tmp/fusion.tif"}},
        "authoritative_base": {"outputs": {"source_aware_candidate": "/tmp/cand.tif", "source_aware_candidate_provenance": "/tmp/cand_prov.tif"}},
        "sdb": {"artifacts": {"depth_raster": "/tmp/sdb_depth.tif"}},
        "river": {"outputs": {"depth_terrain": "/tmp/river_depth.tif", "bottom_elevation": "/tmp/river_bed.tif"}},
        "final_dem_route": {
            "route_mode": "staged_final_route_single_source_of_truth",
            "single_authoritative_route_active": True,
            "legacy_parallel_route_retired": True,
            "structural_inputs": {"authoritative_base": "/tmp/auth.tif", "sdb_guide_points": "/tmp/sdb_points.gpkg"},
        },
    }
    summary = build_legacy_cleanup_summary(report)
    assert summary["legacy_parallel_route_retired"] is True
    assert summary["legacy_inputs_excluded_from_structural_route"] is True
    assert summary["deprecated_artifact_count"] >= 1
