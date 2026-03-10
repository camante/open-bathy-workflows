# Open Bathy Workflows — Technical Guide

> **Authoritative file list:** each run writes `io_manifest.json` and `io_manifest.md` into `--out-dir`.
> Treat those manifests, plus `bathy_report.json` and `unified_bathy_report.json`, as the authoritative record of what actually happened.

This guide summarizes the key invariants and current technical behavior of the workflow.

## 1) Orchestrator and defaults

`bathy_main.py` is the top-level orchestrator.
Verified current defaults in code include:

- `--methods=sdb,river,fuse`
- `--priority=sdb`
- `--river-method=hybrid`
- bounded SDB model-bank support enabled by default
- final-output retention that favors deliverables and run metadata over keeping every intermediate file in `--out-dir`

## 2) Depth sign and elevation interpretation

Current documentation should assume:

- deliverable depth rasters are generally **negative-down**
- bed rasters are elevations referenced to the working DEM vertical reference unless an explicit conversion step is requested
- optional SDB conversion to NAVD88 is explicit, opt-in, and recorded in reports/manifests

Do not mix the internal sign handling of input soundings with the final deliverable sign convention.
Several river and calibration routines normalize input soundings internally before writing final products.

## 3) Explicit-artifact doctrine

The workflow is intentionally moving away from filename guessing.
Important examples:

- `bathy_main.py` writes `io_manifest.json` and `io_manifest.md`
- `sdb_main.py` writes `artifacts_sdb.json`
- `river_diagnostics.py` writes `unified_bathy_report.json` and `unified_bathy_report.md`
- `run_summary.py` writes machine, technical, scientific, and human summaries into `run_logs/`

When downstream logic needs an output path, the code increasingly discovers it from these manifests or reports rather than assuming a historical filename.

## 4) Domain policy

The workflow maintains distinct coastal and river domains.
The final domain policy in `bathy_main.py` is a last-resort guard that clips deliverables according to the run mode.
Conceptually:

- SDB-only runs are clipped to the coastal water domain
- river-only runs are clipped to the river/channel domain and associated coastal safety masks where required by the implementation
- combined runs are clipped to the appropriate combined allowed-water domain

This protects against lingering bathymetry on land or outside the intended hydrologic domain.

## 5) River methods

### Hybrid

This is the current default and should be treated as the main operational river path.
It runs mainstem-focused cross-section inference and combines it with skeleton-based river bathymetry elsewhere.
The goal is better continuity on larger channels without forcing cross-sections across every tributary or junction.

### Skeleton

`river_skeleton_bathy.py` is the no-cross-section path.
It is useful where XS generation is unstable or geometrically awkward.
It can incorporate soundings, optional authoritative bed blending, WSE smoothing, and SWOT-based residual correction.

### XS

`xs_builder.py` plus `xs_infer_bathy_raster.py` is the explicit cross-section path.
It remains important for mainstem structure and hydraulic priors, but it is also the path most sensitive to overlap, junction geometry, and deconfliction settings.

## 6) Hydraulic constraints

The current river stack can incorporate several optional constraints:

- external soundings
- drainage-area-based and regional priors
- Manning-based inversion support
- optional 1D energy-solver prioring in `xs_infer_bathy_raster.py`
- optional SWOT RiverSP water-surface elevation anchoring

SWOT is used here as a WSE/stage/slope constraint, not as a direct bed raster.

## 7) SDB model bank and cache behavior

The SDB side supports a bounded model bank so neighboring AOIs do not behave like fully isolated training problems.
This is intended to improve seam stability without keeping unbounded training state.
The workflow also uses explicit cache and fingerprint helpers so cache hits can be tied to parameter/input/code state rather than guessed.

## 8) Seam comparison support

`bathy_main.py` can compare a run against one or more neighboring `io_manifest.json` files.
When requested, it writes `seam_comparisons.json` and records those results in the main report.
Supporting tools also exist under `seam_stability/` for more detailed seam metrics and gating.

## 9) Verification artifacts and receipts

Depending on the path taken, you may see receipts such as:

- `input_receipt.json`
- `soundings_channel_receipt.json`
- `energy_solver_receipt.json`
- `*.reproject_receipt.json`

These are intended to make specific decisions auditable rather than inferred from logs alone.

## 10) Verification commands

```bash
./verify_repo.sh
./ci_smoke.sh
```

`verify_repo.sh` is the standard offline repo check.
`ci_smoke.sh` is stricter and also fails if committed cache artifacts such as `__pycache__`, `.pytest_cache`, or `*.pyc` are present in the repo tree.

## 11) Recommended debugging order

When behavior is unclear, inspect:

1. `bathy_report.json`
2. `unified_bathy_report.json`
3. `io_manifest.json`
4. `run_logs/`
5. component-specific receipts and diagnostics in `sdb/`, `river/`, or the cache tree
