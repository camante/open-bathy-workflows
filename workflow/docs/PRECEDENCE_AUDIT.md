# Precedence Audit

The precedence audit is a machine-readable JSON artifact emitted by the authoritative conditioning stage.

## Purpose
It proves the final DEM respected the authoritative-first contract.

## Required checks
- locked authoritative cell count
- count of locked cells changed by conditioning
- count of locked cells with non-zero guidance influence
- authoritative gap cell count
- finite conditioned gap cell count
- remaining nodata gap count
- support-class counts
- provenance-class counts

## Pass conditions
- `lock_preserved = true`
- `changed_locked_cell_count = 0`
- `guidance_nonzero_on_locked_count = 0`
- support/provenance classes match the final DEM contract

The audit should be written next to the conditioned DEM outputs and also included in the run report.
