# Test suite classification

This document classifies the workflow test suite into three buckets that match the current bathymetry architecture.

## Core
Core tests protect the intended end-state workflow:
- authoritative hard control
- support-aware classification
- canonical river scaffold and trusted interior
- guidance-first river/SDB artifacts
- deterministic terrain interpolation
- seam stability, reproducibility, provenance, and validation contracts

In `pytest`, any test file not explicitly listed as `transitional` or `legacy` is treated as `core`.

Run them with:

```bash
pytest -m core
```

## Transitional
Transitional tests protect still-supported routes that are useful during migration but are not the architectural center of the final workflow.

Current transitional files:
- `test_bathy_main.py`
- `test_bathy_main_authoritative_river_fallback.py`
- `test_fusion_atl.py`
- `test_pipeline.py`
- `test_pipeline_aoi.py`
- `test_river_guidance_controls.py`
- `test_run_summary_scientific.py`
- `test_source_aware_candidate.py`
- `test_source_aware_candidate_river_corridor.py`
- `test_tier2_path.py`

Why these are transitional:
- they exercise orchestration or fallback behavior that is still present but should keep shrinking as the terrain interpolator and guidance-first route take over
- some are framed in older fusion/candidate terminology even though the behaviors they protect still matter

Run them with:

```bash
pytest -m transitional
```

## Legacy
Legacy tests protect older fusion-centered behavior that is no longer the scientific center of the workflow but is still worth preserving during the migration period.

Current legacy files:
- `test_bathy_fusion_numerical.py`

Why this is legacy:
- it focuses on standalone blending behavior from `bathy_fusion.py`
- your intended end-state is authoritative control plus support-aware terrain generation, not peer-surface blending as the central product model

Run them with:

```bash
pytest -m legacy
```

## Practical CI recommendation
For routine confidence in the current workflow, prioritize:

```bash
pytest -m core
```

Then keep `transitional` and `legacy` coverage available for migration safety and regression review.
