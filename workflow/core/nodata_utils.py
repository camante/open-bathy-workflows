from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import rasterio

WORKFLOW_NODATA = -9999.0

def is_float_dtype(dtype: Any) -> bool:
    try:
        return np.issubdtype(np.dtype(dtype), np.floating)
    except Exception:
        return False


def should_sanitize_dataset_read(ds: Any, indexes: Any = None) -> bool:
    """Return True for continuous rasters that should be sanitized on read.

    We intentionally do *not* sanitize semantic mask/class rasters such as uint8
    WAFFLES/support/provenance products where values like 0/1 are meaningful.
    """
    try:
        dtypes = list(getattr(ds, 'dtypes', []) or [])
    except Exception:
        dtypes = []
    if indexes is None:
        idxs = list(range(len(dtypes)))
    elif isinstance(indexes, int):
        idxs = [int(indexes) - 1]
    elif isinstance(indexes, (list, tuple)):
        idxs = [int(i) - 1 for i in indexes]
    else:
        idxs = list(range(len(dtypes)))
    selected: list[Any] = []
    for i in idxs:
        if 0 <= i < len(dtypes):
            selected.append(dtypes[i])
    if any(is_float_dtype(dt) for dt in selected):
        return True
    try:
        nd = getattr(ds, 'nodata', None)
        if nd is not None:
            nd = float(nd)
            if np.isfinite(nd) and abs(nd) >= 9999.0 and not selected:
                return True
    except Exception:
        pass
    return False


def sanitize_read_result(arr: np.ndarray, nodata: Optional[float], *, extra: Optional[Iterable[float]] = None, extreme_abs: float = 1.0e20) -> np.ndarray:
    """Sanitize a read array while preserving masked-array behavior when possible."""
    masked = np.ma.isMaskedArray(arr)
    data = arr.filled(np.nan) if masked else np.asarray(arr)
    if not is_float_dtype(data.dtype):
        return arr
    out = sanitize_array(data, nodata, dtype=data.dtype, extra=extra, extreme_abs=extreme_abs)
    if masked:
        mask = np.ma.getmaskarray(arr) | ~np.isfinite(out)
        return np.ma.array(out, mask=mask, copy=False)
    return out

_DEFAULT_SENTINELS = (
    -999999.0,
    -99999.0,
    -9999.0,
    9999.0,
    99999.0,
    999999.0,
    -3.402823466e38,
    3.402823466e38,
)


def nodata_sentinels(extra: Optional[Iterable[float]] = None) -> tuple[float, ...]:
    vals = list(_DEFAULT_SENTINELS)
    if extra is not None:
        for v in extra:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            vals.append(fv)
    out: list[float] = []
    for v in vals:
        if np.isfinite(v) and v not in out:
            out.append(float(v))
    return tuple(out)


def nodata_mask(arr: np.ndarray, nodata: Optional[float] = None, *, extra: Optional[Iterable[float]] = None, extreme_abs: float = 1.0e20) -> np.ndarray:
    data = np.asarray(arr)
    mask = ~np.isfinite(data)
    if nodata is not None:
        try:
            nd = float(nodata)
        except (TypeError, ValueError):
            nd = None
        if nd is not None and np.isfinite(nd):
            mask |= np.isclose(data.astype(np.float64, copy=False), nd, rtol=0.0, atol=0.0)
    sentinels = nodata_sentinels(extra)
    if sentinels:
        sent_mask = np.isin(data.astype(np.float64, copy=False), np.asarray(sentinels, dtype=np.float64))
        mask |= sent_mask
    if extreme_abs is not None and extreme_abs > 0:
        with np.errstate(invalid='ignore'):
            mask |= np.abs(data.astype(np.float64, copy=False)) >= float(extreme_abs)
    return mask


def valid_mask(arr: np.ndarray, nodata: Optional[float] = None, *, extra: Optional[Iterable[float]] = None, extreme_abs: float = 1.0e20) -> np.ndarray:
    return ~nodata_mask(arr, nodata, extra=extra, extreme_abs=extreme_abs)


def sanitize_array(arr: np.ndarray, nodata: Optional[float] = None, *, dtype: str | np.dtype = 'float32', extra: Optional[Iterable[float]] = None, extreme_abs: float = 1.0e20) -> np.ndarray:
    out = np.asarray(arr, dtype=dtype).copy()
    out[nodata_mask(out, nodata, extra=extra, extreme_abs=extreme_abs)] = np.nan
    return out


def read_band_sanitized(src: str | Path | rasterio.io.DatasetReader, band: int = 1, *, dtype: str | np.dtype = 'float32', masked: bool = False, extra: Optional[Iterable[float]] = None, extreme_abs: float = 1.0e20) -> np.ndarray:
    if hasattr(src, 'read') and hasattr(src, 'nodata'):
        ds = src
        arr = ds.read(band, masked=masked)
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        return sanitize_array(arr, getattr(ds, 'nodata', None), dtype=dtype, extra=extra, extreme_abs=extreme_abs)
    with rasterio.open(src) as ds:
        arr = ds.read(band, masked=masked)
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        return sanitize_array(arr, ds.nodata, dtype=dtype, extra=extra, extreme_abs=extreme_abs)


def resolve_nodata_value(nodata: Optional[float], *, default: float = -9999.0, extra: Optional[Iterable[float]] = None) -> float:
    try:
        if nodata is not None:
            nd = float(nodata)
            if np.isfinite(nd) and nd not in nodata_sentinels(extra):
                return nd
    except (TypeError, ValueError):
        pass
    return float(default)


def prepare_array_for_reproject(arr: np.ndarray, nodata: Optional[float], *, dtype: str | np.dtype = 'float32', default_nodata: float = -9999.0, extra: Optional[Iterable[float]] = None, extreme_abs: float = 1.0e20) -> tuple[np.ndarray, float]:
    resolved = resolve_nodata_value(nodata, default=default_nodata, extra=extra)
    out = sanitize_array(arr, nodata, dtype=dtype, extra=extra, extreme_abs=extreme_abs)
    out = np.asarray(out, dtype=dtype)
    out[~np.isfinite(out)] = resolved
    return out, float(resolved)


def resolve_reproject_nodata_value(nodata: Optional[float], *, default: float = -9999.0) -> float:
    """Return the true source nodata value for warp-time masking when possible.

    Unlike resolve_nodata_value(), this preserves common finite sentinel nodata
    values such as -99999 / -999999 because rasterio.reproject() needs the
    actual source nodata sentinel in order to ignore those cells during warp.
    Fall back only when nodata is missing or non-finite.
    """
    try:
        if nodata is not None:
            nd = float(nodata)
            if np.isfinite(nd):
                return nd
    except (TypeError, ValueError):
        pass
    return float(default)


def sanitize_for_output(
    arr: np.ndarray,
    *,
    nodata: Optional[float] = None,
    dtype: str | np.dtype = "float32",
    extra: Optional[Iterable[float]] = None,
    extreme_abs: float = 1.0e20,
    min_allowed: Optional[float] = None,
    max_allowed: Optional[float] = None,
) -> np.ndarray:
    """Return float array with every invalid/sentinel/out-of-range cell converted to NaN.

    This is the workflow-wide write-boundary sanitizer.  It must be applied before
    nodata replacement so no alternate finite sentinels (for example -999999) can
    survive into output rasters as fake valid data.
    """
    out = sanitize_array(arr, nodata, dtype=dtype, extra=extra, extreme_abs=extreme_abs)
    if min_allowed is not None:
        with np.errstate(invalid='ignore'):
            out[out < float(min_allowed)] = np.nan
    if max_allowed is not None:
        with np.errstate(invalid='ignore'):
            out[out > float(max_allowed)] = np.nan
    return out


def fill_output_nodata(
    arr: np.ndarray,
    *,
    nodata: float,
    dtype: str | np.dtype = "float32",
    extra: Optional[Iterable[float]] = None,
    extreme_abs: float = 1.0e20,
    min_allowed: Optional[float] = None,
    max_allowed: Optional[float] = None,
) -> np.ndarray:
    out = sanitize_for_output(
        arr,
        nodata=nodata,
        dtype=dtype,
        extra=extra,
        extreme_abs=extreme_abs,
        min_allowed=min_allowed,
        max_allowed=max_allowed,
    )
    resolved = resolve_nodata_value(nodata, default=-9999.0, extra=extra)
    out = np.asarray(out, dtype=dtype)
    out[~np.isfinite(out)] = resolved
    return out


def array_valid_mask(
    arr: np.ndarray,
    *,
    nodata: Optional[float] = None,
    extra: Optional[Iterable[float]] = None,
    extreme_abs: float = 1.0e20,
    min_allowed: Optional[float] = None,
    max_allowed: Optional[float] = None,
) -> np.ndarray:
    data = sanitize_for_output(
        arr,
        nodata=nodata,
        dtype='float32',
        extra=extra,
        extreme_abs=extreme_abs,
        min_allowed=min_allowed,
        max_allowed=max_allowed,
    )
    return np.isfinite(data)


def collect_array_nodata_audit(
    arr: np.ndarray,
    *,
    nodata: Optional[float] = None,
    extra: Optional[Iterable[float]] = None,
) -> dict[str, Any]:
    data = np.asarray(arr, dtype=np.float64)
    sentinel_vals = nodata_sentinels(extra)
    audit = {
        'shape': tuple(int(v) for v in data.shape),
        'declared_nodata': (None if nodata is None else float(nodata)),
        'nonfinite_count': int(np.count_nonzero(~np.isfinite(data))),
        'finite_min': None,
        'finite_max': None,
        'sentinel_counts': {},
    }
    finite = data[np.isfinite(data)]
    if finite.size:
        audit['finite_min'] = float(np.min(finite))
        audit['finite_max'] = float(np.max(finite))
    for s in sentinel_vals:
        audit['sentinel_counts'][str(float(s))] = int(np.count_nonzero(np.isclose(data, float(s), rtol=0.0, atol=0.0)))
    return audit
