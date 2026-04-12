from pathlib import Path

from river_guidance_contract import build_river_guidance_contract, validate_river_guidance_contract


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def test_build_river_guidance_contract_reports_missing_required_artifacts(tmp_path: Path):
    river_dir = tmp_path / 'river'
    for name in [
        'river_guidance_weight.tif',
        'river_admissibility.tif',
        'river_corridor_mask.tif',
        'river_guide_points.gpkg',
    ]:
        _touch(river_dir / name)
    contract = build_river_guidance_contract(out_root=tmp_path, river_dir=river_dir, report={'river': {'outputs': {}}})
    assert 'centerline_elevation' in contract.required_structural_artifacts
    assert 'centerline_elevation' in contract.missing_required_structural_artifacts


def test_validate_river_guidance_contract_returns_missing_required_artifact_error(tmp_path: Path):
    river_dir = tmp_path / 'river'
    for name in [
        'river_guidance_weight.tif',
        'river_admissibility.tif',
        'river_corridor_mask.tif',
        'river_guide_points.gpkg',
    ]:
        _touch(river_dir / name)
    contract = build_river_guidance_contract(out_root=tmp_path, river_dir=river_dir, report={'river': {'outputs': {}}})
    errors = validate_river_guidance_contract(contract)
    assert any('missing_required_river_guidance_artifacts' in e for e in errors)
