# Open Bathy Workflows

This repository is the current March 2026 workflow snapshot for unified coastal and river bathymetry production.
The main entrypoint is `bathy_main.py`, which orchestrates:

- Satellite-Derived Bathymetry (SDB) from Sentinel-2 plus ICESat-2 and optional external XYZ
- River bathymetry using a hybrid default that injects cross-section inference on the mainstem and skeleton-based interpolation elsewhere
- Fusion into a single deliverable raster, with explicit domain clipping and run metadata

## Current workflow state

The codebase is organized around a few operational invariants:

- `bathy_main.py` defaults to `--methods=sdb,river,fuse`
- `--river-method` defaults to `hybrid`
- final deliverable paths are treated as explicit artifacts, not guessed from canonical filenames
- each run writes `io_manifest.json` and `io_manifest.md` in `--out-dir`
- the workflow also writes `bathy_report.json`, `unified_bathy_report.json`, and run summaries under `run_logs/`
- final depth rasters are documented as **negative-down** unless a product explicitly states otherwise
- final outputs are clipped to the intended coastal and/or river water domain as a last-resort safety guard

## Quick start

Run from the workflow directory:

```bash
python bathy_main.py \
  --aoi="-71.14/-71.10/42.75/42.78" \
  --start=2025-01-01 \
  --end=2026-01-01 \
  --methods=sdb,river,fuse \
  --out-dir=output/example_run \
  --cache-root=cache \
  --extra-xyz-cudem=hydronos,ehydro
```

## Where to look after a run

Treat these as the first files to inspect:

- `io_manifest.json` and `io_manifest.md` for the exact inputs and outputs used by the run
- `bathy_report.json` for step-level status, commands, and recorded outputs
- `unified_bathy_report.json` for a compact whole-run summary
- `run_logs/` for flight recorder logs and human/technical/scientific summaries

## Verification

Offline verification:

```bash
./verify_repo.sh
```

Stricter repository hygiene check:

```bash
./ci_smoke.sh
```

These checks are intended to catch syntax, CLI wiring, lightweight unit regressions, and committed cache artifacts before you package or commit changes.

## Documentation map

- `README_GENERAL_WORKFLOW.md` — operator-focused overview and quickstart
- `README_WORKFLOW_PLAIN.md` — plain-language explanation of what the workflow is doing
- `README_WORKFLOW_TECHNICAL.md` — key invariants, domain policies, reports, and troubleshooting
- `README_WORKFLOW_DETAILED.md` — stage-by-stage walkthrough of the full pipeline
- `README_SCRIPTS_DETAILED.md` — grouped module and script index
- `CHECKLIST_A_GRADE.md` — short enforceable repo checklist
- `seam_stability/` — seam comparison docs, metrics, and gating tools

## Important documentation rule

Do not rely on historical "canonical" filenames in older notes.
Use the manifests and run reports written by the current run as the authoritative record of what was actually produced.
