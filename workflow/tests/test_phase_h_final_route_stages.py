from pathlib import Path
import json

from final_route_inputs_stage import resolve_existing_output_path


def test_resolve_existing_output_path_supports_relative_paths(tmp_path: Path):
    base = tmp_path / 'out'
    base.mkdir()
    target = base / 'river' / 'river_depth.tif'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'x')
    mapping = {'depth_terrain': 'river/river_depth.tif'}
    resolved = resolve_existing_output_path(mapping, 'depth_terrain', base_dir=base)
    assert resolved == target.resolve()


def test_resolve_existing_output_path_preserves_absolute_paths(tmp_path: Path):
    target = tmp_path / 'abs.tif'
    target.write_bytes(b'x')
    mapping = {'depth_terrain': str(target)}
    resolved = resolve_existing_output_path(mapping, 'depth_terrain', base_dir=tmp_path / 'ignored')
    assert resolved == target


from types import SimpleNamespace
from final_route_inputs_stage import collect_final_route_inputs


def test_collect_final_route_inputs_supports_string_out_dir(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    auth = tmp_path / "auth.tif"
    auth.write_bytes(b"x")
    sdb_dir = out_dir / "sdb"
    sdb_dir.mkdir()
    depth = sdb_dir / "depth.tif"
    depth.write_bytes(b"x")
    (sdb_dir / "artifacts_sdb.json").write_text(json.dumps({"depth_raster": "depth.tif"}), encoding="utf-8")
    cfg = SimpleNamespace(out_dir=str(out_dir), authoritative_base=str(auth))
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=None, report={})
    assert paths is not None
    assert paths.combined_dir == (out_dir / "combined")
    assert paths.auth_src == auth.resolve()
    assert paths.template_path == auth.resolve()


def test_collect_final_route_inputs_prefers_authoritative_baseline_grid_when_available(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    auth = tmp_path / "auth.tif"
    auth.write_bytes(b"x")
    baseline = tmp_path / "cudem_baseline_interpolation.tif"
    baseline.write_bytes(b"y")
    river_dir = out_dir / "river"
    river_dir.mkdir()
    river_depth = river_dir / "river_depth_terrain_patch.tif"
    river_depth.write_bytes(b"z")
    cfg = SimpleNamespace(out_dir=str(out_dir), authoritative_base=str(auth))
    report = {
        "authoritative_base_auto": {"baseline_cudem_interpolation": str(baseline)},
        "river": {"outputs": {"depth_terrain": str(river_depth)}},
    }
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=None, report=report)
    assert paths is not None
    assert paths.template_path == auth.resolve()
    assert paths.baseline_cudem_src == baseline.resolve()



def test_collect_final_route_inputs_receipt_matches_actual_template_selection_order(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    auth = tmp_path / "auth.tif"
    auth.write_bytes(b"x")
    baseline = tmp_path / "cudem_baseline_interpolation.tif"
    baseline.write_bytes(b"y")
    cfg = SimpleNamespace(out_dir=str(out_dir), authoritative_base=str(auth))
    report = {"authoritative_base_auto": {"baseline_cudem_interpolation": str(baseline)}}
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=None, report=report)
    payload = json.loads(paths.inputs_receipt_path.read_text(encoding="utf-8"))
    assert payload["template_role"] == "authoritative_base_canonical_grid"
    assert payload["template_selection_order"] == [
        "authoritative_base_canonical_grid",
        "baseline_cudem_interpolation_background_only",
        "dense_sdb_depth_raster_guidance_only",
        "dense_river_depth_raster_guidance_only",
        "legacy_candidate_diagnostic_only",
    ]
    assert payload["structural_inputs"]["canonical_final_grid_source"] == str(auth.resolve())
