from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from nodata_utils import nodata_mask, nodata_sentinels

COMMON_SENTINELS = tuple(nodata_sentinels())


class RasterContractError(RuntimeError):
    pass


@dataclass
class RasterSpec:
    path: str
    driver: Optional[str]
    width: int
    height: int
    count: int
    dtype: str
    nodata: Optional[float]
    epsg: Optional[int]
    transform: tuple[float, float, float, float, float, float]
    finite_min: Optional[float]
    finite_max: Optional[float]
    finite_pixels: int
    sentinel_hits: int
    raw_nonfinite_pixels: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _epsg(crs: Any) -> Optional[int]:
    try:
        return crs.to_epsg() if crs is not None else None
    except Exception:
        return None


def _transform_tuple(transform: Any) -> tuple[float, float, float, float, float, float]:
    try:
        return (
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        )
    except Exception:
        vals = tuple(transform) if transform is not None else (0, 0, 0, 0, 0, 0)
        if len(vals) >= 6:
            return tuple(float(v) for v in vals[:6])
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _iter_valid_values(ds) -> Iterable[np.ndarray]:
    nodata = ds.nodata
    extra = [s for s in COMMON_SENTINELS if nodata is None or not math.isclose(float(s), float(nodata), rel_tol=0.0, abs_tol=0.0)]
    for bidx in range(1, ds.count + 1):
        for _, window in ds.block_windows(bidx):
            arr = ds.read(bidx, window=window, masked=False)
            arr = np.asarray(arr)
            valid = ~nodata_mask(arr, nodata, extra=extra)
            if np.any(valid):
                yield arr[valid].astype("float64", copy=False)


def collect_raster_spec(path: os.PathLike[str] | str) -> RasterSpec:
    import rasterio

    p = Path(path)
    with rasterio.open(p) as ds:
        finite_min: Optional[float] = None
        finite_max: Optional[float] = None
        finite_pixels = 0
        sentinel_hits = 0
        raw_nonfinite_pixels = 0
        nodata = float(ds.nodata) if ds.nodata is not None and np.isfinite(ds.nodata) else None
        exclude = {float(nodata)} if nodata is not None else set()
        for bidx in range(1, ds.count + 1):
            for _, window in ds.block_windows(bidx):
                raw = np.asarray(ds.read(bidx, window=window, masked=False))
                raw_nonfinite_pixels += int(np.count_nonzero(~np.isfinite(raw)))
                for s in COMMON_SENTINELS:
                    if float(s) in exclude:
                        continue
                    sentinel_hits += int(np.count_nonzero(np.isclose(raw, s, rtol=0.0, atol=max(1e-6, abs(float(s)) * 1e-6))))
            
        for vals in _iter_valid_values(ds):
            if vals.size == 0:
                continue
            finite_pixels += int(vals.size)
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            finite_min = vmin if finite_min is None else min(finite_min, vmin)
            finite_max = vmax if finite_max is None else max(finite_max, vmax)
        return RasterSpec(
            path=str(p),
            driver=ds.driver,
            width=int(ds.width),
            height=int(ds.height),
            count=int(ds.count),
            dtype=str(ds.dtypes[0]) if ds.count else "",
            nodata=nodata,
            epsg=_epsg(ds.crs),
            transform=_transform_tuple(ds.transform),
            finite_min=finite_min,
            finite_max=finite_max,
            finite_pixels=finite_pixels,
            sentinel_hits=sentinel_hits,
            raw_nonfinite_pixels=raw_nonfinite_pixels,
        )


def _profile_expected(profile: Dict[str, Any], path: os.PathLike[str] | str) -> RasterSpec:
    from rasterio.crs import CRS
    import rasterio

    crs = profile.get("crs")
    if crs is not None and not hasattr(crs, "to_epsg"):
        try:
            crs = CRS.from_user_input(crs)
        except Exception:
            crs = None
    dtype = profile.get("dtype")
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    nodata = profile.get("nodata")
    try:
        nodata = float(nodata) if nodata is not None else None
    except Exception:
        nodata = None
    transform = profile.get("transform", rasterio.Affine.identity())
    return RasterSpec(
        path=str(path),
        driver=profile.get("driver"),
        width=int(profile.get("width") or 0),
        height=int(profile.get("height") or 0),
        count=int(profile.get("count") or 1),
        dtype=str(dtype or ""),
        nodata=nodata,
        epsg=_epsg(crs),
        transform=_transform_tuple(transform),
        finite_min=None,
        finite_max=None,
        finite_pixels=0,
        sentinel_hits=0,
        raw_nonfinite_pixels=0,
    )


def _semantic_problems(actual: RasterSpec, *, min_allowed: float | None = None, max_allowed: float | None = None) -> list[str]:
    problems: list[str] = []
    if actual.raw_nonfinite_pixels > 0:
        problems.append(f"found {actual.raw_nonfinite_pixels} raw non-finite pixels; invalid cells must be written as declared nodata")
    if actual.sentinel_hits > 0:
        problems.append(f"found {actual.sentinel_hits} unexpected sentinel-valued pixels in raster payload")
    if min_allowed is not None and actual.finite_min is not None and actual.finite_min < min_allowed:
        problems.append(f"min {actual.finite_min} below allowed {min_allowed}")
    if max_allowed is not None and actual.finite_max is not None and actual.finite_max > max_allowed:
        problems.append(f"max {actual.finite_max} above allowed {max_allowed}")
    return problems


def validate_raster_against_profile(
    path: os.PathLike[str] | str,
    profile: Dict[str, Any],
    *,
    operation: str,
    source_path: os.PathLike[str] | str | None = None,
    min_allowed: float | None = None,
    max_allowed: float | None = None,
) -> RasterSpec:
    actual = collect_raster_spec(path)
    expected = _profile_expected(profile, path)
    problems: list[str] = []
    if expected.driver and actual.driver != expected.driver:
        problems.append(f"driver expected {expected.driver} got {actual.driver}")
    for name in ("width", "height", "count", "dtype", "epsg"):
        exp = getattr(expected, name)
        if exp not in (None, "", 0) and getattr(actual, name) != exp:
            problems.append(f"{name} expected {exp} got {getattr(actual, name)}")
    if expected.nodata is None:
        if actual.nodata is not None:
            problems.append(f"nodata expected None got {actual.nodata}")
    else:
        if actual.nodata is None or not math.isclose(actual.nodata, expected.nodata, rel_tol=0.0, abs_tol=1e-9):
            problems.append(f"nodata expected {expected.nodata} got {actual.nodata}")
    if expected.transform != (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) and actual.transform != expected.transform:
        problems.append(f"transform expected {expected.transform} got {actual.transform}")
    problems.extend(_semantic_problems(actual, min_allowed=min_allowed, max_allowed=max_allowed))
    if problems:
        src_spec = None
        if source_path and Path(source_path).exists():
            try:
                src_spec = collect_raster_spec(source_path).to_dict()
            except Exception as exc:
                src_spec = {"path": str(source_path), "error": repr(exc)}
        raise RasterContractError(
            f"Raster contract violation after {operation} for {path}: "
            + "; ".join(problems)
            + f" | expected={expected.to_dict()} actual={actual.to_dict()}"
            + (f" source={src_spec}" if src_spec is not None else "")
        )
    return actual


def validate_raster_mask_values(path: os.PathLike[str] | str, *, operation: str, allowed_values: set[int], nodata: int | None = None) -> RasterSpec:
    import rasterio

    actual = collect_raster_spec(path)
    with rasterio.open(path) as ds:
        bad = 0
        for bidx in range(1, ds.count + 1):
            for _, window in ds.block_windows(bidx):
                arr = ds.read(bidx, window=window, masked=False)
                vals = np.unique(arr)
                for v in vals.tolist():
                    if nodata is not None and int(v) == int(nodata):
                        continue
                    if int(v) not in allowed_values:
                        bad += 1
        if bad:
            raise RasterContractError(
                f"Raster contract violation after {operation} for {path}: found values outside allowed mask set {sorted(allowed_values)} with nodata={nodata}; actual={actual.to_dict()}"
            )
    return actual


def validate_gdal_output(
    path: os.PathLike[str] | str,
    *,
    operation: str,
    expected_crs: Any | None = None,
    expected_nodata: float | None = None,
    expected_dtype: str | None = None,
    min_allowed: float | None = None,
    max_allowed: float | None = None,
    source_path: os.PathLike[str] | str | None = None,
) -> RasterSpec:
    actual = collect_raster_spec(path)
    problems: list[str] = []
    exp_epsg = None
    try:
        if expected_crs is not None:
            from rasterio.crs import CRS
            exp_epsg = CRS.from_user_input(expected_crs).to_epsg()
    except Exception:
        exp_epsg = None
    if exp_epsg is not None and actual.epsg != exp_epsg:
        problems.append(f"epsg expected {exp_epsg} got {actual.epsg}")
    if expected_nodata is None:
        if actual.nodata is not None:
            problems.append(f"nodata expected None got {actual.nodata}")
    elif actual.nodata is None or not math.isclose(actual.nodata, expected_nodata, rel_tol=0.0, abs_tol=1e-9):
        problems.append(f"nodata expected {expected_nodata} got {actual.nodata}")
    if expected_dtype is not None and actual.dtype != expected_dtype:
        problems.append(f"dtype expected {expected_dtype} got {actual.dtype}")
    problems.extend(_semantic_problems(actual, min_allowed=min_allowed, max_allowed=max_allowed))
    if problems:
        src_spec = None
        if source_path and Path(source_path).exists():
            try:
                src_spec = collect_raster_spec(source_path).to_dict()
            except Exception as exc:
                src_spec = {"path": str(source_path), "error": repr(exc)}
        raise RasterContractError(
            f"Raster contract violation after {operation} for {path}: "
            + "; ".join(problems)
            + f" | actual={actual.to_dict()}"
            + (f" source={src_spec}" if src_spec is not None else "")
        )
    return actual


def cached_raster_semantics_valid(
    path: os.PathLike[str] | str,
    *,
    expected_nodata: float | None = None,
    expected_crs: Any | None = None,
    expected_dtype: str | None = None,
    min_allowed: float | None = None,
    max_allowed: float | None = None,
) -> tuple[bool, str, Optional[RasterSpec]]:
    try:
        spec = validate_gdal_output(
            path,
            operation="cache_hit_validation",
            expected_crs=expected_crs,
            expected_nodata=expected_nodata,
            expected_dtype=expected_dtype,
            min_allowed=min_allowed,
            max_allowed=max_allowed,
        )
        return True, "ok", spec
    except RasterContractError as exc:
        return False, str(exc), None
    except Exception as exc:
        return False, f"cache_hit_validation could not open/inspect raster: {exc}", None
