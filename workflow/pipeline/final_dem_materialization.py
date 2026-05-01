from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pipeline.river_workflow.river_workflow_contract import (
    AOI_EXPORT_DEM_ROLE,
    FINAL_DEM_MATERIALIZER_ROLE,
    FINAL_USER_DEM_ROLE,
    assert_materialization_receipt_valid,
)
from pipeline.output_products import assert_only_materializer_writes_final_user_dem, role_for_final_path



def _sha256_file(path: Path) -> str:
    """Return a deterministic SHA-256 digest for a materialized raster/file.

    The final-route guard relies on this helper to prove that
    combined/DEM_enhanced.tif is a byte-for-byte copy of the named AOI export.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _append_final_dem_touch(final_dem_path: Path, *, source: Path, note: str) -> Path:
    """Append a single-writer trace entry for the final DEM materializer."""
    touch_log = Path(final_dem_path).parent / "dem_enhanced_touch_log.jsonl"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "action": "final_dem_materializer_write",
        "path": str(final_dem_path),
        "source": str(source),
        "writer_role": FINAL_DEM_MATERIALIZER_ROLE,
        "note": note,
    }
    touch_log.parent.mkdir(parents=True, exist_ok=True)
    with touch_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    return touch_log


def materialize_final_dem_from_aoi_export(
    *,
    aoi_export_dem: Path,
    final_dem_path: Path,
    receipt_path: Path,
    expected_source_role: str = AOI_EXPORT_DEM_ROLE,
) -> Path:
    """Materialize the stable final DEM from an AOI export artifact.

    This function is deliberately narrow: it copies the named AOI export to the
    stable user-facing final path. It refuses reprojection/resampling because
    display or comparison reprojections must be written as sidecars, not by
    mutating the scientific final DEM. It never runs terrain interpolation,
    blending, or authoritative locking.
    """
    src = Path(aoi_export_dem)
    dst = Path(final_dem_path)
    receipt = Path(receipt_path)
    if not src.is_file():
        raise FileNotFoundError(f"final_dem_aoi_export_missing:{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst_role = role_for_final_path(dst)
    if dst_role != FINAL_USER_DEM_ROLE:
        raise ValueError(f"final_dem_materialization_invalid_destination:{dst}")
    assert_only_materializer_writes_final_user_dem(role=dst_role, writer_role=FINAL_DEM_MATERIALIZER_ROLE)

    mode = "copy"
    pixel_values_modified = False
    resampled = False
    reprojected = False
    # The stable final DEM is copy-only: the AOI export stage is responsible
    # for producing the scientific grid. Any alternate-SRS product must be
    # created later as a display/comparison sidecar, never by mutating
    # combined/DEM_enhanced.tif.
    source_crs_matches_final = True
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
        touch_note = "copied_aoi_export_to_final_user_dem"
    else:
        touch_note = "aoi_export_already_at_final_user_dem_path"

    if not dst.exists():
        raise FileNotFoundError(f"final_dem_materialization_missing_destination:{dst}")

    touch_log = _append_final_dem_touch(dst, source=src, note=touch_note)

    payload: dict[str, Any] = {
        "stage": "final_dem_materialization",
        "writer_role": FINAL_DEM_MATERIALIZER_ROLE,
        "source_role": expected_source_role,
        "destination_role": FINAL_USER_DEM_ROLE,
        "source_path": str(src),
        "destination_path": str(dst),
        "receipt_path": str(receipt),
        "touch_log_path": str(touch_log),
        "mode": mode,
        "copy_only": True,
        "final_out_srs": None,
        "source_crs_matches_final": source_crs_matches_final,
        "pixel_values_modified": pixel_values_modified,
        "resampled": resampled,
        "reprojected": reprojected,
        "terrain_interpolation": False,
        "blending": False,
        "authoritative_locking": False,
        "aoi_local_solve": False,
        "source_size_bytes": src.stat().st_size,
        "destination_size_bytes": dst.stat().st_size,
        "source_sha256": _sha256_file(src),
        "destination_sha256": _sha256_file(dst),
        "source_destination_hash_match": _sha256_file(src) == _sha256_file(dst),
    }
    assert_materialization_receipt_valid(payload)
    _write_receipt(receipt, payload)
    return dst


def write_final_dem_from_aoi_export(*, aoi_export_dem: Path, final_dem_path: Path, receipt_path: Path) -> Path:
    """Named Bundle-E entry point: final DEM can only be written from AOI export."""
    return materialize_final_dem_from_aoi_export(
        aoi_export_dem=aoi_export_dem,
        final_dem_path=final_dem_path,
        receipt_path=receipt_path,
        expected_source_role=AOI_EXPORT_DEM_ROLE,
    )


__all__ = ["materialize_final_dem_from_aoi_export", "write_final_dem_from_aoi_export"]
