from __future__ import annotations

from typing import Tuple

import math


def classify_section_tendency_family(
    width_m: float,
    support_class: str,
    confinement_ratio: float | None = None,
    local_authoritative_density: float | None = None,
) -> str:
    support = str(support_class or "unsupported").strip().lower()
    width = float(width_m) if math.isfinite(width_m) else float("nan")
    if support in {"bed_supported", "authoritative_locked"}:
        return "authoritative_section_derived"
    if math.isfinite(width):
        if width <= 12.0:
            return "narrow_v"
        if width >= 40.0:
            return "compound_lowflow"
    if confinement_ratio is not None and math.isfinite(confinement_ratio) and confinement_ratio >= 2.5:
        return "narrow_v"
    if local_authoritative_density is not None and math.isfinite(local_authoritative_density) and local_authoritative_density >= 0.5:
        return "authoritative_section_derived"
    return "flat_u"


def compute_tendency_depth_fraction(
    width_m: float,
    family: str,
    support_class: str,
    component_class: str | None = None,
    reconciliation_confidence: float | None = None,
    support_distance_m: float | None = None,
) -> float:
    width = float(width_m) if math.isfinite(width_m) else 20.0
    width = min(max(width, 4.0), 120.0)
    width_scale = min(max((width - 10.0) / 42.0, 0.0), 1.0)
    base = 0.17 + 0.05 * width_scale
    fam = str(family or "flat_u")
    if fam == "narrow_v":
        base += 0.05
    elif fam == "compound_lowflow":
        base -= 0.025
    elif fam == "authoritative_section_derived":
        base += 0.015
    support = str(support_class or "unsupported").strip().lower()
    if support in {"bed_supported", "authoritative_locked"}:
        base += 0.025
    elif support == "bank_margin_only":
        base -= 0.045
    elif support in {"none", "unsupported"}:
        base -= 0.015
    cc = str(component_class or "unknown").strip().lower()
    rc = float(reconciliation_confidence) if reconciliation_confidence is not None and math.isfinite(reconciliation_confidence) else float("nan")
    sd = float(support_distance_m) if support_distance_m is not None and math.isfinite(support_distance_m) else float("nan")
    weak_recon = (not math.isfinite(rc)) or rc < 0.35
    far_support = (not math.isfinite(sd)) or sd >= 200.0
    moderate_support = math.isfinite(sd) and sd <= 175.0
    moderate_recon = math.isfinite(rc) and rc >= 0.35
    strong_recon = math.isfinite(rc) and rc >= 0.60
    if cc == "unsupported_mainstem":
        if moderate_support or moderate_recon:
            base += 0.015
        if strong_recon and moderate_support:
            base += 0.010
        if weak_recon and far_support:
            base -= 0.010
    elif cc == "unsupported_side_component":
        base -= 0.040
        if moderate_recon and moderate_support:
            base += 0.005
        if weak_recon:
            base -= 0.020
        if far_support:
            base -= 0.015
    elif cc == "tiny_detached_component":
        base -= 0.070
        if weak_recon:
            base -= 0.025
        if far_support:
            base -= 0.020
    return float(min(max(base, 0.06), 0.30))


def compute_tendency_confidence(
    width_m: float,
    family: str,
    support_class: str,
    component_class: str | None = None,
    reconciliation_confidence: float | None = None,
    support_distance_m: float | None = None,
) -> float:
    width = float(width_m) if math.isfinite(width_m) else 20.0
    width = min(max(width, 4.0), 120.0)
    conf = 0.58
    fam = str(family or "flat_u")
    if fam == "authoritative_section_derived":
        conf += 0.15
    elif fam == "narrow_v":
        conf += 0.03
    support = str(support_class or "unsupported").strip().lower()
    if support in {"bed_supported", "authoritative_locked"}:
        conf += 0.12
    elif support == "bank_margin_only":
        conf -= 0.08
    elif support in {"none", "unsupported"}:
        conf -= 0.03
    if width >= 60.0:
        conf -= 0.03
    cc = str(component_class or "unknown").strip().lower()
    rc = float(reconciliation_confidence) if reconciliation_confidence is not None and math.isfinite(reconciliation_confidence) else float("nan")
    sd = float(support_distance_m) if support_distance_m is not None and math.isfinite(support_distance_m) else float("nan")
    weak_recon = (not math.isfinite(rc)) or rc < 0.35
    far_support = (not math.isfinite(sd)) or sd >= 200.0
    moderate_support = math.isfinite(sd) and sd <= 175.0
    moderate_recon = math.isfinite(rc) and rc >= 0.35
    strong_recon = math.isfinite(rc) and rc >= 0.60
    if cc == "unsupported_mainstem":
        if moderate_support or moderate_recon:
            conf += 0.04
        if strong_recon and moderate_support:
            conf += 0.02
        if weak_recon and far_support:
            conf -= 0.03
    elif cc == "unsupported_side_component":
        conf -= 0.10
        if moderate_recon and moderate_support:
            conf += 0.02
        if weak_recon:
            conf -= 0.05
        if far_support:
            conf -= 0.03
    elif cc == "tiny_detached_component":
        conf -= 0.18
        if weak_recon:
            conf -= 0.07
        if far_support:
            conf -= 0.04
    if math.isfinite(rc):
        conf += 0.06 * max(min(rc, 1.0), 0.0)
    return float(min(max(conf, 0.15), 0.92))


def compute_inner_node_positions(width_m: float, family: str) -> Tuple[float, float]:
    fam = str(family or "flat_u")
    if fam == "narrow_v":
        return (0.33, 0.67)
    if fam == "compound_lowflow":
        return (0.22, 0.78)
    return (0.28, 0.72)


def build_section_from_thalweg(
    thalweg_z: float,
    width_m: float,
    family: str,
    tendency_depth_fraction: float,
    bank_caps: tuple[float, float] | None = None,
) -> tuple[float, float, float]:
    if not math.isfinite(thalweg_z):
        return (float("nan"), float("nan"), float("nan"))
    width = float(width_m) if math.isfinite(width_m) else 20.0
    width = min(max(width, 4.0), 120.0)
    fam = str(family or "flat_u")
    amplitude = 0.28 + 0.022 * math.sqrt(width)
    if fam == "narrow_v":
        amplitude += 0.18
    elif fam == "compound_lowflow":
        amplitude += 0.05
    elif fam == "authoritative_section_derived":
        amplitude += 0.10
    amplitude *= min(max(float(tendency_depth_fraction), 0.10), 0.40) / 0.20
    amplitude = min(max(amplitude, 0.22), 1.40)
    left_inner = float(thalweg_z) + amplitude
    right_inner = float(thalweg_z) + amplitude
    if bank_caps is not None:
        left_cap, right_cap = bank_caps
        if math.isfinite(left_cap):
            left_inner = min(left_inner, float(left_cap) - 0.05)
        if math.isfinite(right_cap):
            right_inner = min(right_inner, float(right_cap) - 0.05)
    left_inner = max(left_inner, float(thalweg_z)) if math.isfinite(left_inner) else float("nan")
    right_inner = max(right_inner, float(thalweg_z)) if math.isfinite(right_inner) else float("nan")
    return (left_inner, right_inner, amplitude)
