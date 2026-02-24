"""river_wse.py – Fit a smoothed longitudinal water-surface elevation (WSE) profile along river stationing

Longitudinal Water Surface Elevation (WSE) profile fitting.

Why this exists
--------------
River bathymetry inference often starts from a DEM-derived proxy for water
surface elevation at each cross-section (e.g., a low quantile near the XS
center). Those per-XS estimates are noisy. Using them directly to compute slope
can inject large errors into Manning inversion and multivariate priors.

This module fits a *reach-consistent* WSE profile per river_id:

1) Smooth noisy WSE along stationing (rolling median/mean)
2) Optionally enforce monotonicity (pool-adjacent-violators / isotonic)
3) Provide a stabilized slope estimate from the fitted profile

The output is designed to be merged back into the cross-section parameter
table (xs_param) and used as:

* A replacement for the "slope proxy" derived from unsmoothed WSE
* A physically plausible WSE curve for diagnostics and future SWOT/ICESat-2
  constraints
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class WSEFitConfig:
    """Configuration for WSE profile fitting."""

    enabled: bool = True
    window: int = 9
    min_n: int = 7
    enforce_monotonic: bool = True
    slope_min: float = 1e-5
    slope_max: float = 0.05


def _rolling_smooth(vals: np.ndarray, window: int) -> np.ndarray:
    w = int(max(3, window))
    if w % 2 == 0:
        w += 1
    s = pd.Series(vals)
    med = s.rolling(window=w, center=True, min_periods=max(3, w // 2)).median()
    sm = med.rolling(window=w, center=True, min_periods=max(3, w // 2)).mean()
    return sm.to_numpy(dtype="float64")


def _pava_isotonic(y: np.ndarray, increasing: bool = True) -> np.ndarray:
    """Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.

    Returns the closest (L2) monotone sequence to y.
    """
    y = np.asarray(y, dtype="float64")
    n = y.size
    if n <= 1:
        return y.copy()

    # If decreasing, flip sign and solve increasing.
    if not increasing:
        y = -y

    # Each block has (start, end, mean, weight)
    means = y.copy()
    wts = np.ones(n, dtype="float64")
    starts = np.arange(n, dtype=int)
    ends = np.arange(n, dtype=int)

    m = 0  # number of active blocks-1
    for i in range(n):
        starts[m] = i
        ends[m] = i
        means[m] = y[i]
        wts[m] = 1.0
        # Merge backward while violating monotonicity
        while m > 0 and means[m - 1] > means[m]:
            tot_w = wts[m - 1] + wts[m]
            new_mean = (means[m - 1] * wts[m - 1] + means[m] * wts[m]) / tot_w
            means[m - 1] = new_mean
            wts[m - 1] = tot_w
            ends[m - 1] = ends[m]
            m -= 1
        m += 1

    # Expand blocks
    out = np.empty(n, dtype="float64")
    for b in range(m):
        out[starts[b] : ends[b] + 1] = means[b]

    if not increasing:
        out = -out
    return out


def fit_wse_profile(
    xs_param: pd.DataFrame,
    cfg: Optional[WSEFitConfig] = None,
    river_id_field: str = "river_id",
    s_field: str = "s_center_m",
    wse_field: str = "wse_m",
) -> Tuple[pd.Series, pd.Series]:
    """Fit a stabilized WSE profile and slope per XS.

    Parameters
    ----------
    xs_param:
        Cross-section parameter table containing river_id, stationing, and WSE.
    cfg:
        WSEFitConfig.
    river_id_field, s_field, wse_field:
        Column names.

    Returns
    -------
    wse_fit_m:
        Smoothed/monotone WSE estimate per XS (NaN if unavailable)
    slope_wse_mpm:
        Stabilized water-surface slope per XS (m/m), NaN if unavailable.
    """
    if cfg is None:
        cfg = WSEFitConfig()

    wse_fit = pd.Series(np.nan, index=xs_param.index, dtype="float64")
    slope = pd.Series(np.nan, index=xs_param.index, dtype="float64")

    if (not cfg.enabled) or xs_param is None or xs_param.empty:
        return wse_fit, slope

    req = {river_id_field, s_field, wse_field}
    if not req.issubset(set(xs_param.columns)):
        return wse_fit, slope

    for _, g in xs_param.groupby(river_id_field, dropna=False):
        gg = g.copy()
        gg[s_field] = pd.to_numeric(gg[s_field], errors="coerce")
        gg[wse_field] = pd.to_numeric(gg[wse_field], errors="coerce")
        gg = gg.dropna(subset=[s_field, wse_field]).sort_values(s_field)
        if len(gg) < int(cfg.min_n):
            continue

        s = gg[s_field].to_numpy(dtype="float64")
        w = gg[wse_field].to_numpy(dtype="float64")

        w_smooth = _rolling_smooth(w, int(cfg.window))

        if bool(cfg.enforce_monotonic):
            # Choose direction based on correlation; WSE should usually decrease downstream.
            corr = np.corrcoef(s, w_smooth)[0, 1] if (np.std(s) > 0 and np.std(w_smooth) > 0) else np.nan
            decreasing = True
            if np.isfinite(corr) and corr > 0:
                # WSE appears to increase with s -> likely stationing reversed
                decreasing = False
            w_fit = _pava_isotonic(w_smooth, increasing=(not decreasing))
        else:
            w_fit = w_smooth

        # Slope from central differences of fitted WSE
        if len(s) >= 3:
            ds = s[2:] - s[:-2]
            dw = w_fit[2:] - w_fit[:-2]
            with np.errstate(divide="ignore", invalid="ignore"):
                sl = np.abs(dw / ds)
            sl = np.concatenate([[np.nan], sl, [np.nan]])
            sl = np.clip(sl, float(cfg.slope_min), float(cfg.slope_max))
        else:
            sl = np.full_like(s, np.nan, dtype="float64")

        wse_fit.loc[gg.index] = w_fit
        slope.loc[gg.index] = sl

    return wse_fit, slope
