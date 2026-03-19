from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional
import logging


def ensure_river_contribution(*, union_overlay_fn: Callable[[Path, Path, Path, Path, str], None],
                              gdal_union_overlay_fn: Callable[[Path, Path, Path, Path, str], None],
                              sdb_fuse_path: Path, river_fuse_path: Path, out_depth: Path,
                              out_prov: Optional[Path], template: Path, pri: str,
                              report: Dict[str, Any], logger: logging.Logger) -> None:
    """Ensure fused output contains some river contribution, with explicit fallback receipts."""
    import rasterio
    import numpy as np

    has_river = False
    if out_prov and Path(out_prov).exists():
        with rasterio.open(str(out_prov)) as dp:
            p = dp.read(1)
            has_river = bool(np.any((p == 2) | (p == 6)))
    if has_river:
        report.setdefault("fusion", {}).setdefault("guidance_controls", {})["river_present_after_fusion"] = True
        return

    try:
        union_overlay_fn(Path(sdb_fuse_path), Path(river_fuse_path), Path(out_depth), Path(template), pri)
        fallback = "simple_union_overlay"
    except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.debug("Simple union overlay fallback failed: %s", e, exc_info=True)
        gdal_union_overlay_fn(Path(sdb_fuse_path), Path(river_fuse_path), Path(out_depth), Path(template), pri)
        fallback = "gdal_union_overlay"

    report.setdefault("fusion", {}).setdefault("guidance_controls", {})["river_present_after_fusion"] = False
    report["fusion"]["guidance_controls"]["river_contribution_fallback"] = fallback
    report["fusion"]["note"] = "River contribution missing after fusion; applied union overlay fallback."



def burn_authoritative_xyz_into_final(*, out_depth: Path, xyz_paths, report: Dict[str, Any], logger: logging.Logger) -> None:
    """Burn authoritative XYZ depths exactly into the fused output where points fall on cells."""
    import rasterio
    import numpy as np
    from rasterio.transform import rowcol

    xyz_paths = [str(Path(x)) for x in (xyz_paths or []) if str(x).strip()]
    if not xyz_paths or not Path(out_depth).exists():
        return

    with rasterio.open(str(out_depth), 'r+') as ds:
        arr = ds.read(1).astype('float32')
        nodata = ds.nodata if ds.nodata is not None else -9999.0
        _ = nodata
        h, w = ds.height, ds.width
        pix_idx = []
        zvals = []
        for xp in xyz_paths:
            fp = Path(xp)
            if not fp.exists():
                continue
            dat = None
            for delim in (None, ','):
                try:
                    dat = np.genfromtxt(str(fp), dtype='float64', delimiter=delim)
                    if dat.ndim == 1:
                        dat = dat.reshape(1, -1)
                    break
                except (OSError, ValueError, TypeError):
                    dat = None
            if dat is None or dat.size == 0 or dat.shape[1] < 3:
                continue
            x = dat[:, 0]
            y = dat[:, 1]
            z = dat[:, 2]
            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if not np.any(m):
                continue
            x = x[m]; y = y[m]; z = z[m]
            rr, cc = rowcol(ds.transform, x, y)
            rr = np.asarray(rr); cc = np.asarray(cc)
            mm = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            if not np.any(mm):
                continue
            rr = rr[mm]; cc = cc[mm]; z = z[mm]
            idx = rr * w + cc
            pix_idx.append(idx)
            zvals.append(z)

        if not pix_idx:
            report.setdefault('fusion', {}).setdefault('constraints', {})['xyz_burned_into_final'] = False
            report['fusion']['constraints']['xyz_burn_reason'] = 'no_valid_xyz_points'
            return

        idx = np.concatenate(pix_idx)
        zv = np.concatenate(zvals).astype('float32')
        order = np.argsort(idx)
        idx = idx[order]
        zv = zv[order]
        uniq, start = np.unique(idx, return_index=True)
        med = np.empty_like(uniq, dtype='float32')
        for i, s0 in enumerate(start):
            s1 = start[i + 1] if i + 1 < len(start) else len(idx)
            med[i] = np.median(zv[s0:s1])
        rr = (uniq // w).astype('int64')
        cc = (uniq % w).astype('int64')
        arr[rr, cc] = med
        ds.write(arr.astype('float32'), 1)

    report.setdefault('fusion', {}).setdefault('constraints', {})['xyz_burned_into_final'] = True
    report['fusion']['constraints']['xyz_files'] = [str(Path(x).name) for x in xyz_paths]



def apply_residual_correction_with_reporting(*, cfg, out_depth: Path, report: Dict[str, Any], logger: logging.Logger,
                                             load_support_points_fn: Callable[[Any], Any],
                                             apply_residual_correction_fn: Callable[..., Dict[str, Any]]) -> None:
    """Apply smooth residual correction toward authoritative support where enough points exist."""
    support_points = load_support_points_fn(cfg)
    if support_points is None or len(support_points) < 20:
        report.setdefault("fusion", {}).setdefault("residual_correction", {})["status"] = "skipped"
        report["fusion"]["residual_correction"]["reason"] = "insufficient_support_points"
        return

    rc_stats = apply_residual_correction_fn(
        Path(out_depth),
        support_points,
        sigma_m=1000.0,
        max_correction_m=8.0,
        min_points=20,
        label="FUSED_RESIDUAL",
    )
    report.setdefault("fusion", {})["residual_correction"] = rc_stats
