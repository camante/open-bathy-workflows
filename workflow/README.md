# Bathymetry Workflow

This repository contains the current bathymetry workflow for authoritative-first coastal and river terrain generation.
The main entrypoint is `bathy_main.py`.

## Active workflow story

The repo supports SDB, river, and fused runs, but the **recommended active river workflow** is the explicit linear path:

`entry bundle -> solve_domain -> grids -> authoritative -> centerline -> wse_proxy -> authoritative_bed -> observed_offset -> modeled_offset -> backbone -> corridor -> surface -> lock -> export -> final_dem`

That is the river story this repo should be read through first.
The active river workflow is one built-in AOI-independent route. Historical river names remain only as compatibility aliases and normalize to the same route; they should not be read as selectable methods. Secondary validation/benchmark helpers live under `validation/`, run/report writers live under `reporting/`, shared runtime infrastructure lives under `core/`, and curated diagnostics helpers live under `tools/debug/`. River processing is controlled by `--aoi`, optional `--solve-domain`, and explicit science/runtime knobs.

## Recommended run patterns

River-focused run using the active linear path:

```bash
python bathy_main.py \
  --aoi="-71.15/-71.04/42.70/42.75" \
  --start=2025-01-01 \
  --end=2026-01-01 \
  --out-dir=output/example_linear_run
```

Full run with SDB, river, and fusion:

```bash
python bathy_main.py \
  --aoi="-71.15/-71.04/42.70/42.75" \
  --start=2025-01-01 \
  --end=2026-01-01 \
  --methods=sdb,river,fuse \
  --out-dir=output/example_full_run
```

## Normal run outputs

For a normal run, the first files to inspect are:

- `RUN_OVERVIEW.txt`
- `bathy_report.json`
- `io_manifest.json`
- `run_logs/screen_*.log` if the run failed or looks wrong

A normal run should be read through those files first, not through extra debug trees.
Additional receipts, manifests, and stage artifacts are for intermediate/debug runs only.

## Verification

Repository verification:

```bash
./verify_repo.sh
```

Stricter hygiene check:

```bash
./ci_smoke.sh
```

## Documentation map

Start here, in order:

1. `docs/ACTIVE_WORKFLOW.md` — the active workflow path and stage contract
2. `docs/RUN_OUTPUTS.md` — what a normal run writes and what to inspect first
3. `docs/REPO_MAP.md` — how the repo is organized
4. `docs/LEGACY_WORKFLOWS.md` — what older paths still exist and how to think about them

Repo-level runtime/output language is pinned in `repo_runtime_modes.py` and checked by `repo_contract_checks.py`.

Historical README variants, patch notes, and older design notes now live under `archive/` and are no longer part of the top-level workflow story.

## Compatibility note

The repo now uses one built-in river workflow. User control comes from `--aoi`, optional `--solve-domain`, and explicit science/runtime knobs rather than river method selection.
