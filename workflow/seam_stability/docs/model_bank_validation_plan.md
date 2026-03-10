# Model Bank Validation Plan

Purpose:
validate that the bounded SDB model-bank behavior improves cross-tile stability without drifting silently or hiding coverage failures.

## Minimum evidence to retain per evaluation run

- `unified_bathy_report.json`
- `io_manifest.json`
- `artifacts_sdb.json`
- seam metrics for at least one neighboring AOI comparison
- training/diversity and validation outputs when those were produced by the run
- any bank-drift or probe-set comparison artifact used in the assessment

## What to evaluate

### Seam behavior

Confirm that adjacent AOIs using the bank do not develop a larger edge mismatch than the non-bank baseline.

### Convergence behavior

If you are running repeated updates against a fixed probe set, track the probe-set deltas across updates rather than relying on a single final accuracy number.

### Coverage behavior

Document where predictions were possible versus where water, clarity, mask, or DOA gates prevented output.
A model bank that smooths seams but silently loses valid coverage is not a pass.
