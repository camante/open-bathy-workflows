from __future__ import annotations

from pathlib import Path

from pipeline.final_dem_materialization import write_final_dem_from_aoi_export
from pipeline.river_workflow.river_workflow_execution_receipt import write_execution_mode_receipt
from pipeline.river_workflow.river_workflow_execution_contract import AOI_EXPORT_ONLY_ROLE, CANONICAL_BUILD_ROLE


def test_final_dem_materialization_is_byte_for_byte_copy(tmp_path: Path) -> None:
    export = tmp_path / "river_workflow" / "export" / "DEM_enhanced_export.tif"
    final = tmp_path / "combined" / "DEM_enhanced.tif"
    receipt = tmp_path / "river_workflow" / "receipts" / "final_dem_materialization_receipt.json"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"synthetic-dem-bytes")

    result = write_final_dem_from_aoi_export(aoi_export_dem=export, final_dem_path=final, receipt_path=receipt)

    assert result == final
    assert final.read_bytes() == export.read_bytes()
    payload = receipt.read_text(encoding="utf-8")
    assert '"source_destination_hash_match": true' in payload
    assert '"copy_only": true' in payload


def test_execution_receipt_allows_canonical_build_with_construction_receipts(tmp_path: Path) -> None:
    parent = tmp_path / "river_workflow" / "final" / "canonical_parent_dem.tif"
    export = tmp_path / "river_workflow" / "export" / "DEM_enhanced_export.tif"
    final = tmp_path / "combined" / "DEM_enhanced.tif"
    for path in (parent, export, final):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")

    result = write_execution_mode_receipt(
        out_dir=tmp_path,
        execution_role=CANONICAL_BUILD_ROLE,
        effective_mode="canonical_build_then_export",
        stage_receipts={"solve_domain": tmp_path / "solve.json", "aoi_export_identity": tmp_path / "export.json"},
        canonical_parent_dem=parent,
        aoi_export_dem=export,
        final_user_dem=final,
        canonical_manifest_path=None,
        source="synthetic_test",
    )

    assert result.passed is True
    assert result.receipt_path.is_file()


def test_execution_receipt_rejects_export_only_with_construction_receipts(tmp_path: Path) -> None:
    parent = tmp_path / "river_workflow" / "final" / "canonical_parent_dem.tif"
    export = tmp_path / "river_workflow" / "export" / "DEM_enhanced_export.tif"
    for path in (parent, export):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")

    try:
        write_execution_mode_receipt(
            out_dir=tmp_path,
            execution_role=AOI_EXPORT_ONLY_ROLE,
            effective_mode="aoi_export_only",
            stage_receipts={"solve_domain": tmp_path / "solve.json", "aoi_export_only": tmp_path / "export.json"},
            canonical_parent_dem=parent,
            aoi_export_dem=export,
            final_user_dem=None,
            canonical_manifest_path=None,
            source="synthetic_test",
        )
    except RuntimeError as exc:
        assert "river_execution_mode_contract_failed" in str(exc)
    else:
        raise AssertionError("export-only execution receipt accepted construction-stage receipts")
