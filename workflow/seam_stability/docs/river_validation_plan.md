# River Bathymetry Validation Plan

Purpose:
check that river outputs are hydraulically plausible, domain-restricted, and tile-stable.

## Minimum evidence to retain per evaluation run

- `bathy_report.json`
- `unified_bathy_report.json`
- `io_manifest.json`
- river-domain mask used by the run
- any river-specific receipts written by the workflow
- seam or transition metrics if comparing neighboring AOIs or river/coastal transitions

## What to evaluate

### Domain restriction

Confirm that the final river deliverables are nodata outside the river/channel domain.

### Longitudinal plausibility

Inspect centerline or profile-based diagnostics for uphill fractions, slope spikes, or abrupt bed steps that are inconsistent with the expected reach behavior.

### Transition behavior

Where river and coastal products meet, evaluate whether fusion or priority rules introduce an artificial step.
