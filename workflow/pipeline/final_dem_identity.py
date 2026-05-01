"""Final DEM identity checks for the active parent/export/final route.

This module keeps the final DEM single-source-of-truth verification out of
``bathy_main.py``.  The check is intentionally verify-only: it records hashes,
validates raster/grid/value identity, and writes receipts without mutating the
final raster.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from core.json_io import write_json
from core.paths import ensure_dir

log = logging.getLogger(__name__)


def file_sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        log.debug("file_sha256: suppressed exception", exc_info=True)
        return None


def record_dem_enhanced_touch(
    out_dir: Path,
    *,
    action: str,
    path: Path,
    source: Optional[Path] = None,
    note: Optional[str] = None,
) -> None:
    try:
        combined_dir = ensure_dir(Path(out_dir) / "combined")
        touch_log = combined_dir / "dem_enhanced_touch_log.jsonl"
        payload = {
            "action": str(action),
            "path": str(Path(path)),
            "exists": bool(Path(path).exists()),
            "source": str(source) if source is not None else None,
            "source_exists": bool(Path(source).exists()) if source is not None else None,
            "path_sha256": file_sha256(Path(path)) if Path(path).exists() else None,
            "source_sha256": file_sha256(Path(source)) if source is not None and Path(source).exists() else None,
            "note": note,
            "is_symlink": bool(Path(path).is_symlink()),
        }
        with touch_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        log.info("[FINAL_OUTPUT][TRACE] %s: %s%s", action, path, f" <- {source}" if source is not None else "")
    except Exception:
        log.debug("record_dem_enhanced_touch: suppressed exception", exc_info=True)


def replace_with_symlink_or_copy(src: Path, dst: Path, *, out_dir: Optional[Path] = None, action: str = "replace_with_copy") -> None:
    """Replace a deliverable with a plain file copy.

    This helper intentionally avoids symlinks so final user-facing outputs and
    stable sidecars are straightforward on disk.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    try:
        if src == dst or (dst.exists() and os.path.samefile(src, dst)):
            if out_dir is not None and dst.name == "DEM_enhanced.tif":
                record_dem_enhanced_touch(out_dir, action=f"{action}_noop", path=dst, source=src)
            return
    except Exception:
        log.debug("replace_with_symlink_or_copy: samefile check suppressed exception", exc_info=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if out_dir is not None and dst.name == "DEM_enhanced.tif":
        record_dem_enhanced_touch(out_dir, action=f"{action}_copy", path=dst, source=src)


def _raster_identity_summary(src_path: Path, dst_path: Path) -> dict[str, Any]:
    import numpy as _np
    import rasterio as _rio

    summary: dict[str, Any] = {
        "same_shape": False,
        "same_transform": False,
        "same_crs": False,
        "same_nodata": False,
        "same_dtype": False,
        "same_values": False,
        "source_role": None,
        "final_role": None,
        "max_abs_diff_m": None,
        "value_diff_pixels": None,
    }
    with _rio.open(src_path) as src_ds, _rio.open(dst_path) as dst_ds:
        summary["source_role"] = src_ds.tags().get("ROLE")
        summary["final_role"] = dst_ds.tags().get("ROLE")
        summary["same_shape"] = src_ds.width == dst_ds.width and src_ds.height == dst_ds.height
        summary["same_transform"] = tuple(src_ds.transform) == tuple(dst_ds.transform)
        summary["same_crs"] = str(src_ds.crs) == str(dst_ds.crs)
        summary["same_nodata"] = float(src_ds.nodata) == float(dst_ds.nodata)
        summary["same_dtype"] = tuple(src_ds.dtypes) == tuple(dst_ds.dtypes)
        src_arr = src_ds.read(1)
        dst_arr = dst_ds.read(1)
        if src_arr.shape == dst_arr.shape:
            same = _np.array_equal(src_arr, dst_arr)
            summary["same_values"] = bool(same)
            if not same:
                diff = src_arr.astype(_np.float64) - dst_arr.astype(_np.float64)
                finite = _np.isfinite(diff)
                if finite.any():
                    summary["max_abs_diff_m"] = float(_np.max(_np.abs(diff[finite])))
                    summary["value_diff_pixels"] = int(_np.count_nonzero(diff[finite] != 0.0))
                else:
                    summary["max_abs_diff_m"] = None
                    summary["value_diff_pixels"] = 0
    return summary


def verify_dem_enhanced_single_source_of_truth(
    cfg: Any,
    report: dict[str, Any],
    *,
    final_native: Optional[Path | str],
    final_for_user: Optional[Path | str],
) -> Optional[str]:
    """Verify that ``combined/DEM_enhanced.tif`` remains the active final DEM.

    The active final-path rule is:
    1. the river workflow builds or reuses a canonical parent DEM;
    2. the AOI DEM is exported as an exact parent-grid subset;
    3. the final DEM materializer copies that AOI export to ``combined/DEM_enhanced.tif``;
    4. later stages verify identity only and must not mutate the final DEM.
    """
    combined_dir = ensure_dir(Path(cfg.out_dir) / "combined")
    final_path = combined_dir / "DEM_enhanced.tif"
    receipt = combined_dir / "dem_enhanced_identity_receipt.json"
    report.setdefault("outputs", {})["final_dem_user_stable"] = str(final_path)
    report.setdefault("outputs", {})["final_depth_user_stable"] = str(final_path)
    report.setdefault("outputs", {})["dem_enhanced_touch_log"] = str(combined_dir / "dem_enhanced_touch_log.jsonl")
    report.setdefault("outputs", {})["dem_enhanced_identity_receipt"] = str(receipt)

    try:
        auth_outputs = report.get("authoritative_base", {}).get("outputs", {}) if isinstance(report.get("authoritative_base", {}).get("outputs", {}), dict) else {}
        internal_source = Path(str(auth_outputs.get("conditioned_final_dem_internal"))) if auth_outputs.get("conditioned_final_dem_internal") else None
        debug_source = Path(str(auth_outputs.get("stage_debug_06_dem_enhanced_written"))) if auth_outputs.get("stage_debug_06_dem_enhanced_written") else None
        source = None
        source_kind = None
        final_for_user_path = Path(str(final_for_user)) if final_for_user is not None else None
        if final_for_user_path is not None and final_for_user_path.exists() and final_for_user_path.resolve() == final_path.resolve():
            source = final_for_user_path
            source_kind = "final_for_user"
        elif internal_source is not None and internal_source.exists():
            source = internal_source
            source_kind = "conditioned_final_dem_internal"
        elif debug_source is not None and debug_source.exists():
            source = debug_source
            source_kind = "stage_debug_06_dem_enhanced_written"
        elif final_native is not None and Path(str(final_native)).exists():
            source = Path(str(final_native))
            source_kind = "final_native"
        elif final_for_user_path is not None and final_for_user_path.exists():
            source = final_for_user_path
            source_kind = "final_for_user"
        if not final_path.exists():
            raise RuntimeError(f"DEM_enhanced missing at expected final path: {final_path}")
        if source is None:
            raise RuntimeError("No valid source available for DEM_enhanced identity verification")
        writer = "final_route_outputs_stage"
        if source_kind == "final_for_user" and source.resolve() == final_path.resolve():
            writer = "final_dem_materializer"
            record_dem_enhanced_touch(
                cfg.out_dir,
                action="active_river_existing_output_verify",
                path=final_path,
                source=source,
                note="existing_final_output_verified_without_rewrite",
            )
        record_dem_enhanced_touch(cfg.out_dir, action="identity_verification_start", path=final_path, source=source)
        if final_path.is_symlink():
            raise RuntimeError(f"DEM_enhanced must be a real raster, not a symlink: {final_path}")
        src_hash = file_sha256(source)
        dst_hash = file_sha256(final_path)
        identity = _raster_identity_summary(source, final_path)
        ok = bool(
            identity.get("same_shape")
            and identity.get("same_transform")
            and identity.get("same_crs")
            and identity.get("same_nodata")
            and identity.get("same_dtype")
            and identity.get("same_values")
        )
        write_json(receipt, {
            "final_path": str(final_path),
            "source": str(source),
            "source_kind": source_kind,
            "source_sha256": src_hash,
            "final_sha256": dst_hash,
            "match": ok,
            "writer": writer,
            "verification_mode": "raster_content_no_rewrite",
            "identity": identity,
        })
        if not ok:
            raise RuntimeError(f"DEM_enhanced mismatch: source={source} final={final_path}")
        record_dem_enhanced_touch(cfg.out_dir, action="identity_verification_ok", path=final_path, source=source)
        return str(final_path)
    except Exception:
        log.exception("[FINAL_OUTPUT] DEM_enhanced identity verification failed")
        raise


__all__ = [
    "file_sha256",
    "record_dem_enhanced_touch",
    "replace_with_symlink_or_copy",
    "verify_dem_enhanced_single_source_of_truth",
]
