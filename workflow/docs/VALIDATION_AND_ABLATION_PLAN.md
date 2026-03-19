Validation and Ablation Plan
===========================

Purpose
-------
Validate the authoritative-first guidance-conditioned workflow by comparing multiple final-surface cases against common truth and reporting metrics by support and provenance class.

Minimum comparisons
-------------------
- authoritative-only interpolation
- authoritative + SDB guidance
- authoritative + river guidance
- authoritative + combined guidance
- legacy final path versus terrain_interpolator path

Minimum metrics
---------------
- overall mean error, MAE, RMSE
- support-class metrics
- provenance-class metrics
- nested-AOI overlap identity deltas
- adjacent-tile seam deltas

Implementation notes
--------------------
The lightweight validation layer lives in validation_runner.py and is intended to support support-class and provenance-class ablation summaries without forcing a heavy benchmark framework into the main runtime path.
