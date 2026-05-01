# River workflow final freeze for SDB template

This pass freezes the active river workflow as the clean template for the SDB cleanup.

## Frozen active architecture

```text
canonical parent river solution
-> exact AOI export
-> combined/DEM_enhanced.tif written once
-> verify-only receipts and comparison reports
```

The active river route must keep these invariants:

- AOI runs do not rebuild local river science when a canonical parent exists.
- AOI exports are exact windows/subsets of the canonical parent.
- `combined/DEM_enhanced.tif` is materialized from the named AOI export artifact.
- The final DEM has one writer and is verified against the AOI export.
- Final-folder products are review products only; they do not drive construction.
- Science diagnostics are read-only summaries from retained parent/stage receipts.

## Backbone diagnostic interpretation

The bed backbone is guidance before authoritative lock, not a hard observed surface. The WSE profile is the hydraulic monotonicity invariant. The bed-backbone diagnostic therefore separates two ideas:

- `large_downstream_rise_count_gt_allowed`: pass/fail metric. This must be zero.
- `positive_downstream_step_count` / `downstream_trend_violation_count`: diagnostic count of small allowed local positive steps.

A passing backbone check means no downstream rise exceeds the stage allowance recorded in the backbone receipt. Small positive local steps may remain for continuity, smoothing, or local support transitions, and should not be interpreted as a route failure when the large-rise count is zero.

`largest_step_m` may include downstream-deepening drops; `max_downstream_rise_m` is the downstream-rise metric.

## What should transfer to SDB

Use the same pattern for SDB:

```text
canonical/source-domain SDB guidance product
-> AOI export/subset
-> final materialization from named export artifact
-> identity receipts
-> parent science summary
-> single-writer final contract
```

Do not let SDB become another hidden final writer. It should produce retained guidance artifacts and receipts first, then participate in the final route only through explicit named products.
