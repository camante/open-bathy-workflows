# Seam Stability Docs and Tools

This directory contains the seam-stability framework docs plus the standalone scripts used to quantify adjacent-tile and transition behavior.

## How this relates to the main workflow

`bathy_main.py` can already perform a lightweight adjacent-tile comparison when you pass neighboring `io_manifest.json` files with:

- `--seam-compare-with-io`
- `--seam-compare-with-io-list`

That in-run comparison writes `seam_comparisons.json` into the run output.
The material in this directory is for deeper and more explicit seam analysis, validation, and gating.

## Contents

### Docs

- `seam_stability/docs/seam_stability_framework.md`
- `seam_stability/docs/model_bank_validation_plan.md`
- `seam_stability/docs/model_bank_validation_runbook.md`
- `seam_stability/docs/river_validation_plan.md`
- `seam_stability/docs/river_validation_runbook.md`
- `seam_stability/docs/seam_stability_summary_tool.md`

### Tools

- `seam_stability/seam_stability_tools/README.md`
- `seam_stability/seam_stability_tools/seam_stability_architecture.svg`
- `seam_stability/seam_stability_tools/seam_stability_architecture.png`
- `seam_stability/seam_stability_tools/scripts/`

## Recommended usage pattern

Use the run-level seam comparison in `bathy_main.py` when you want a quick adjacent-tile check tied directly to a specific run.
Use the scripts in `seam_stability/seam_stability_tools/scripts/` when you want a more formal seam analysis, transition metrics, longitudinal checks, or readiness gating.
