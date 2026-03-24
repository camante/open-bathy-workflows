from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("xs_infer_bathy")

try:  # pragma: no cover - optional dependency
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    log.debug("xs_interpolation: suppressed exception", exc_info=True)
    cKDTree = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.neighbors import KDTree  # type: ignore
except Exception:  # pragma: no cover
    log.debug("xs_interpolation: suppressed exception", exc_info=True)
    KDTree = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from shapely.geometry import Point  # type: ignore
except Exception:  # pragma: no cover
    log.debug("xs_interpolation: suppressed exception", exc_info=True)
    Point = None  # type: ignore


def _kd_tree():
    """Return a (TreeClass, name) using available libs."""
    if cKDTree is not None:
        return cKDTree, "scipy"
    if KDTree is not None:
        return KDTree, "sklearn"
    return None, "none"


def _idw_interpolate_on_mask(
    pts_xy: np.ndarray,
    pts_val: np.ndarray,
    q_xy: np.ndarray,
    k: int = 12,
    power: float = 2.0,
    adaptive: bool = False,
    eps: float = 1e-6,
    pts_weight: Optional[np.ndarray] = None,
    pts_group: Optional[np.ndarray] = None,
    pts_priority: Optional[np.ndarray] = None,
    priority_delta: float = 0.0,
    priority_min_spread: float = 1.0,
) -> np.ndarray:
    """IDW / adaptive IDW interpolation for query coordinates.

    Memory-stable implementation: computes kNN + weights in chunks and writes results
    directly into the output vector (avoids allocating full n_query×k arrays).
    """
    pts_xy = np.asarray(pts_xy, dtype="float64")
    pts_val = np.asarray(pts_val, dtype="float64").reshape(-1)
    q_xy = np.asarray(q_xy, dtype="float64")

    if pts_weight is not None:
        pts_weight = np.asarray(pts_weight, dtype="float64").reshape(-1)
        if pts_weight.shape[0] != pts_val.shape[0]:
            raise ValueError("pts_weight must have same length as pts_val")
        pts_weight = np.maximum(pts_weight, 0.0)

    Tree, which = _kd_tree()
    if Tree is None:
        which = "numpy"

    n_pts = int(len(pts_xy))
    n_q = int(len(q_xy))
    if n_pts == 0 or n_q == 0:
        return np.full((n_q,), np.nan, dtype="float64")

    k_eff = int(min(max(1, int(k)), n_pts))
    out = np.full((n_q,), np.nan, dtype="float64")

    def _compute_vals_from_knn(d: np.ndarray, idx: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype="float64")
        idx = np.asarray(idx, dtype="int64")
        if d.ndim == 1:
            d = d[:, None]
            idx = idx[:, None]

        if adaptive:
            d1 = d[:, 0]
            dK = d[:, -1]
            ratio = np.clip(dK / np.maximum(d1, eps), 1.0, 10.0)
            p = 1.5 + (np.log(ratio) / np.log(10.0)) * (4.0 - 1.5)
            p = p[:, None]
            w = 1.0 / (np.power(d + eps, p))
        else:
            w = 1.0 / (np.power(d + eps, float(power)))

        if pts_weight is not None:
            w = w * pts_weight[idx]

        if pts_group is not None and pts_priority is not None:
            pg = np.asarray(pts_group)
            pp = np.asarray(pts_priority, dtype="float64").reshape(-1)
            if pg.shape[0] == pts_val.shape[0] and pp.shape[0] == pts_val.shape[0]:
                try:
                    g = pg[idx]
                    gv = g.astype("float64", copy=False)
                    gv[gv < 0] = np.nan
                    gmin = np.nanmin(gv, axis=1)
                    gmax = np.nanmax(gv, axis=1)
                    multi = np.isfinite(gmin) & np.isfinite(gmax) & (gmin != gmax)
                    p = pp[idx]
                    pmax = np.nanmax(p, axis=1)
                    pmin = np.nanmin(p, axis=1)
                    spread = pmax - pmin
                    apply = multi & np.isfinite(pmax) & np.isfinite(spread) & (spread >= float(priority_min_spread))
                    if np.any(apply):
                        keep = p >= (pmax[:, None] - float(priority_delta))
                        mask_keep = np.ones_like(w, dtype=bool)
                        mask_keep[apply, :] = keep[apply, :]
                        w_f = w * mask_keep
                        den_f = np.sum(w_f, axis=1)
                        bad = den_f <= eps
                        if np.any(bad):
                            w_f[bad, :] = w[bad, :]
                        w = w_f
                except (TypeError, ValueError, IndexError, FloatingPointError):
                    log.debug("ignored", exc_info=True)

        v = pts_val[idx]
        sw = np.sum(w, axis=1)
        good = np.isfinite(sw) & (sw > 0)
        outv = np.full((idx.shape[0],), np.nan, dtype="float64")
        outv[good] = np.sum(w[good] * v[good], axis=1) / sw[good]
        return outv

    chunk_q = 200000 if which in ("scipy", "sklearn") else 20000

    if which == "scipy":
        tree = Tree(pts_xy)
        for i0 in range(0, n_q, chunk_q):
            q = q_xy[i0 : i0 + chunk_q]
            try:
                d_blk, idx_blk = tree.query(q, k=k_eff, workers=-1)
            except TypeError:
                d_blk, idx_blk = tree.query(q, k=k_eff)
            if k_eff == 1:
                d_blk = np.asarray(d_blk).reshape(-1, 1)
                idx_blk = np.asarray(idx_blk).reshape(-1, 1)
            out[i0 : i0 + d_blk.shape[0]] = _compute_vals_from_knn(d_blk, idx_blk)
    elif which == "sklearn":
        tree = Tree(pts_xy)
        for i0 in range(0, n_q, chunk_q):
            q = q_xy[i0 : i0 + chunk_q]
            d_blk, idx_blk = tree.query(q, k=k_eff, return_distance=True)
            if k_eff == 1:
                d_blk = np.asarray(d_blk).reshape(-1, 1)
                idx_blk = np.asarray(idx_blk).reshape(-1, 1)
            out[i0 : i0 + d_blk.shape[0]] = _compute_vals_from_knn(d_blk, idx_blk)
    else:
        for i0 in range(0, n_q, chunk_q):
            q = q_xy[i0 : i0 + chunk_q]
            dx = q[:, None, 0] - pts_xy[None, :, 0]
            dy = q[:, None, 1] - pts_xy[None, :, 1]
            dist2 = dx * dx + dy * dy
            idx_k = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
            row = np.arange(idx_k.shape[0])[:, None]
            dist2_k = dist2[row, idx_k]
            ord_k = np.argsort(dist2_k, axis=1)
            idx_sorted = idx_k[row, ord_k]
            d_sorted = np.sqrt(dist2[row, idx_sorted])
            out[i0 : i0 + idx_sorted.shape[0]] = _compute_vals_from_knn(d_sorted, idx_sorted)

    return out


def _aniso_idw_interpolate_on_mask(
    pts_xy: np.ndarray,
    pts_val: np.ndarray,
    q_xy: np.ndarray,
    centerline,
    k: int = 12,
    power: float = 2.0,
    along_scale_m: float = 500.0,
    cross_scale_m: float = 30.0,
    eps: float = 1e-6,
    pts_weight: Optional[np.ndarray] = None,
    pts_group: Optional[np.ndarray] = None,
    pts_priority: Optional[np.ndarray] = None,
    priority_delta: float = 0.0,
    priority_min_spread: float = 1.0,
) -> np.ndarray:
    """Anisotropic IDW using a centerline as the along-channel axis."""
    if pts_xy.size == 0 or q_xy.size == 0:
        return np.full((q_xy.shape[0],), np.nan, dtype="float64")

    if pts_weight is not None:
        pts_weight = np.asarray(pts_weight, dtype="float64").reshape(-1)
        if pts_weight.shape[0] != len(pts_val):
            raise ValueError("pts_weight must have same length as pts_val")
        pts_weight = np.maximum(pts_weight, 0.0)

    along_scale_m = float(max(1e-3, along_scale_m))
    cross_scale_m = float(max(1e-3, cross_scale_m))

    try:
        s_pts = np.array([centerline.project(Point(float(x), float(y))) for x, y in pts_xy], dtype="float64")
    except (AttributeError, TypeError, ValueError):
        return _idw_interpolate_on_mask(
            pts_xy, pts_val, q_xy, k=int(k), power=float(power), adaptive=False, eps=eps, pts_weight=pts_weight,
            pts_group=pts_group, pts_priority=pts_priority, priority_delta=0.0, priority_min_spread=1.0,
        )

    Tree, which = _kd_tree()
    if Tree is None:
        return _idw_interpolate_on_mask(
            pts_xy, pts_val, q_xy, k=int(k), power=float(power), adaptive=False, eps=eps, pts_weight=pts_weight,
            pts_group=pts_group, pts_priority=pts_priority, priority_delta=0.0, priority_min_spread=1.0,
        )

    tree = Tree(pts_xy)
    n = pts_xy.shape[0]
    k = int(min(max(1, int(k)), n))
    cand = int(min(n, max(k, k * 5)))
    out = np.full((q_xy.shape[0],), np.nan, dtype="float64")
    chunk = 200000
    for i0 in range(0, q_xy.shape[0], chunk):
        q_blk = q_xy[i0 : i0 + chunk]
        try:
            d_eu, idx = tree.query(q_blk, k=cand, workers=-1)
        except TypeError:
            d_eu, idx = tree.query(q_blk, k=cand)
        if cand == 1:
            d_eu = np.asarray(d_eu).reshape(-1, 1)
            idx = np.asarray(idx).reshape(-1, 1)

        for j in range(q_blk.shape[0]):
            i = i0 + j
            ids = idx[j]
            de = d_eu[j].astype("float64")
            try:
                s_q = centerline.project(Point(float(q_xy[i, 0]), float(q_xy[i, 1])))
            except (AttributeError, TypeError, ValueError):
                ids2 = ids[:k]
                de2 = de[:k]
                w = 1.0 / (np.maximum(de2, eps) ** float(power))
                if pts_weight is not None:
                    w = w * pts_weight[ids2]
                vv = pts_val[ids2]
                sw = np.sum(w)
                out[i] = float(np.sum(w * vv) / sw) if np.isfinite(sw) and sw > 0 else float(np.nan)
                continue

            s_p = s_pts[ids]
            d_along = np.abs(s_q - s_p)
            rad = de * de - d_along * d_along
            valid = rad >= 0
            d_cross = np.zeros_like(de)
            d_cross[valid] = np.sqrt(rad[valid])
            d_cross[~valid] = de[~valid]
            d_eff = np.sqrt((d_along / along_scale_m) ** 2 + (d_cross / cross_scale_m) ** 2) + eps
            order = np.argsort(d_eff)[:k]
            ids2 = ids[order]
            d2 = d_eff[order]

            w = 1.0 / (d2 ** float(power))
            if pts_weight is not None:
                w = w * pts_weight[ids2]
            if pts_group is not None and pts_priority is not None:
                try:
                    g = np.asarray(pts_group)[ids2]
                    gv = g.astype('float64', copy=False)
                    gv[gv < 0] = np.nan
                    gmin = np.nanmin(gv)
                    gmax = np.nanmax(gv)
                    if np.isfinite(gmin) and np.isfinite(gmax) and (gmin != gmax):
                        p = np.asarray(pts_priority, dtype='float64').reshape(-1)[ids2]
                        pmax = np.nanmax(p)
                        pmin = np.nanmin(p)
                        if np.isfinite(pmax) and np.isfinite(pmin) and (pmax - pmin) >= float(priority_min_spread):
                            keep = p >= (pmax - float(priority_delta))
                            w_f = w * keep
                            if np.sum(w_f) > 0:
                                w = w_f
                except (TypeError, ValueError, IndexError, FloatingPointError):
                    log.debug("ignored", exc_info=True)
            vv = pts_val[ids2]
            sw = np.sum(w)
            out[i] = float(np.sum(w * vv) / sw) if np.isfinite(sw) and sw > 0 else float(np.nan)
    return out
