# Coastal + River Seam Stability Framework
(Operational Consistency & Convergence Doctrine)

Purpose:
Ensure seamless bathymetry across:
- Adjacent tiles
- Coastal-to-river transitions
- Independent AOI runs
- Incremental model bank updates

This framework defines the quantitative criteria and invariants required
to guarantee operational seam stability across the full bathymetry system.

---

## 1. Core Seam Stability Principle

A bathymetry workflow is seam-stable if:

1) Adjacent AOIs produce statistically indistinguishable predictions
   within shared boundary regions.

2) River longitudinal profiles remain hydraulically monotonic
   across AOI boundaries.

3) Fusion outputs introduce no additional discontinuity beyond
   input method differences.

4) Model bank updates converge (do not oscillate).

---

## 2. Seam Types (Three Domains)

Seam stability must be evaluated in three domains:

### 2.1 Coastal–Coastal (SDB ↔ SDB)
Adjacent coastal AOIs.

### 2.2 River–River
Adjacent inland AOIs along same watershed.

### 2.3 Coastal–River (Estuarine Transition)
Where SDB and River predictions overlap or merge.

---

## 3. Universal Seam Metric Definition

For any two rasters A and B sharing a boundary:

Define boundary buffer width W (default 200 m).

Within buffer:
- restrict to valid overlapping pixels

Compute:

seam_bias = median(A - B)

seam_rmse = sqrt(mean((A - B)^2))

seam_p95_abs = percentile(|A - B|, 95)

overlap_valid_frac = valid_overlap_pixels / total_buffer_pixels

---

## 4. Coastal–Coastal Seam Stability (SDB)

Targets:

| metric | operational target |
|--------|-------------------|
| median seam bias | ≤ 0.15 m |
| seam RMSE | ≤ 0.50 m |
| seam p95 | ≤ 1.0 m |

Additional:
Difference in DOA pass fraction across seam ≤ 10%.

---

## 5. River–River Seam Stability

Same seam metrics, restricted to river mask.

Additional:
median bed elevation difference along centerline across seam ≤ 0.10 m.

---

## 6. Coastal–River Transition Stability

Within transition zone Ω:
Δ = SDB_depth - River_depth

Targets:
|median(Δ)| ≤ 0.25 m
RMSE(Δ) ≤ 0.75 m

Fusion must not introduce artificial steps > 0.5 m across mask boundary.

---

## 7. Model Bank Convergence

Probe set P fixed in space.

After update k:
Δ_k = P_depth_k - P_depth_{k-1}

Convergence when:
p95(|Δ_k|) ≤ 0.25 m for ≥ 3 consecutive updates.

---

## 8. Fusion Stability Rule

Fusion output F must satisfy:
seam_rmse(F) ≤ max(seam_rmse(SDB), seam_rmse(River)) + 0.10 m

If weighted overlap fails and priority copy is used:
flag seam stability as degraded.

---

## 9. Coverage Consistency Across Seams

|pred_frac_A - pred_frac_B| ≤ 0.20

Prevents coverage cliffs at tile edges.

---

## 10. Required Outputs

Per AOI pair:
- seam_metrics.csv (coastal, river, fusion)
- transition_metrics.csv
- longitudinal metrics (centerline)
- seam_stability_summary.json

---
End of framework.
