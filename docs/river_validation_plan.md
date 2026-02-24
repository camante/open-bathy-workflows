# River Bathymetry Validation Plan

Required:
- river_report.json (mask area, predicted area, datum, connected-water usage)
- River seam metrics (restricted to river mask)
- Longitudinal plausibility along centerlines (median slope, uphill_fraction, slope spikes)
- Transition stability vs SDB in estuary zone (if applicable)
- Coverage accounting within river mask

Targets:
- River seam: |bias|≤0.15 m, rmse≤0.50 m
- Longitudinal: uphill_fraction < 0.10 (except tidal reach)

---
End of plan.
