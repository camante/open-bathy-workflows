from __future__ import annotations
import numpy as np


def edge_interior_mask(shape: tuple[int, int], edge_buffer_px: int) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    if edge_buffer_px <= 0:
        return mask
    h, w = shape
    if edge_buffer_px * 2 >= h or edge_buffer_px * 2 >= w:
        return np.zeros(shape, dtype=bool)
    mask[:edge_buffer_px, :] = False
    mask[-edge_buffer_px:, :] = False
    mask[:, :edge_buffer_px] = False
    mask[:, -edge_buffer_px:] = False
    return mask


def build_authoritative_anchor_support(*, authoritative_support: np.ndarray | None, channel: np.ndarray) -> np.ndarray:
    if authoritative_support is None:
        return np.zeros_like(channel, dtype='uint8')
    return (((np.asarray(authoritative_support) > 0) & (np.asarray(channel) > 0)).astype('uint8'))


def build_trusted_export_region(*, channel: np.ndarray, estuary_transition: np.ndarray | None = None, edge_buffer_px: int = 0) -> np.ndarray:
    trusted = (np.asarray(channel) > 0) & edge_interior_mask(np.asarray(channel).shape, edge_buffer_px)
    if estuary_transition is not None:
        trusted &= ~(np.asarray(estuary_transition) > 0)
    return trusted.astype('uint8')


def build_soft_guidance_domain(*, trusted_export_region: np.ndarray, valid_depth: np.ndarray) -> np.ndarray:
    """Return the broader soft-guidance domain inside the trusted export region.

    This is intentionally broader than admissibility: it marks pixels where river
    guidance could conceptually act, before removing exact authoritative anchor
    pixels that should remain hard control.
    """
    soft = (np.asarray(trusted_export_region) > 0) & np.asarray(valid_depth, dtype=bool)
    return soft.astype('uint8')


def build_river_admissibility(*, soft_guidance_domain: np.ndarray, authoritative_anchor_support: np.ndarray | None = None) -> np.ndarray:
    """Return the soft-guidance subset that is admissible for non-anchor influence."""
    admissible = np.asarray(soft_guidance_domain) > 0
    if authoritative_anchor_support is not None:
        admissible &= ~(np.asarray(authoritative_anchor_support) > 0)
    return admissible.astype('uint8')


def build_river_trusted_interior(*, channel: np.ndarray, estuary_transition: np.ndarray | None = None, edge_buffer_px: int = 0) -> np.ndarray:
    """Backward-compatible wrapper for trusted export region."""
    return build_trusted_export_region(channel=channel, estuary_transition=estuary_transition, edge_buffer_px=edge_buffer_px)


def restrict_river_admissibility(*, trusted_interior: np.ndarray, valid_depth: np.ndarray, authoritative_support: np.ndarray | None = None) -> np.ndarray:
    """Backward-compatible wrapper for admissibility inside the trusted export region."""
    soft = build_soft_guidance_domain(trusted_export_region=trusted_interior, valid_depth=valid_depth)
    anchor = None if authoritative_support is None else build_authoritative_anchor_support(authoritative_support=authoritative_support, channel=np.asarray(trusted_interior) > 0)
    return build_river_admissibility(soft_guidance_domain=soft, authoritative_anchor_support=anchor)



def summarize_trusted_export_region(*, channel: np.ndarray, trusted_export_region: np.ndarray, estuary_transition: np.ndarray | None = None, edge_buffer_px: int = 0) -> dict:
    """Summarize the trusted-export contract for reporting and regression receipts."""
    channel_arr = np.asarray(channel) > 0
    trusted_arr = np.asarray(trusted_export_region) > 0
    estuary_arr = np.asarray(estuary_transition) > 0 if estuary_transition is not None else np.zeros_like(channel_arr, dtype=bool)
    return {
        'channel_pixels': int(channel_arr.sum()),
        'trusted_export_pixels': int(trusted_arr.sum()),
        'estuary_transition_pixels': int(estuary_arr.sum()),
        'edge_buffer_px': int(edge_buffer_px),
        'trusted_fraction_of_channel': float(trusted_arr.sum() / max(int(channel_arr.sum()), 1)),
        'edge_pixels_excluded': int(np.count_nonzero(channel_arr & ~edge_interior_mask(channel_arr.shape, int(edge_buffer_px)))),
        'trusted_excludes_estuary_transition': bool(not np.any(trusted_arr & estuary_arr)),
        'trusted_subset_of_channel': bool(not np.any(trusted_arr & ~channel_arr)),
    }


def trusted_overlap_identity(*, trusted_a: np.ndarray, trusted_b: np.ndarray) -> dict:
    """Return identity metrics for two trusted-interior masks over their common extent."""
    a = np.asarray(trusted_a) > 0
    b = np.asarray(trusted_b) > 0
    ny = min(a.shape[0], b.shape[0])
    nx = min(a.shape[1], b.shape[1])
    if ny <= 0 or nx <= 0:
        return {'overlap_pixels': 0, 'jaccard': None, 'exact_identity_fraction': None}
    a = a[:ny, :nx]
    b = b[:ny, :nx]
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    exact = int(np.count_nonzero(a == b))
    total = int(a.size)
    return {
        'overlap_pixels': inter,
        'union_pixels': union,
        'jaccard': float(inter / max(union, 1)),
        'exact_identity_fraction': float(exact / max(total, 1)),
    }
