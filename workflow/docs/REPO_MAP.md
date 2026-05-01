# Repo Map

This document gives the intended mental model for the repository.

## Primary entrypoint

- `bathy_main.py` — main workflow entrypoint

## Active workflow code

The active river workflow is the built-in shared-solve path, centered on modules such as:
- `pipeline/river_shared_solve/river_context.py`
- `pipeline/river_shared_solve/river_contract.py`
- `pipeline/river_shared_solve/river_paths.py`
- `pipeline/river_shared_solve/river_pipeline.py`
- `pipeline/river_shared_solve/river_canonical_domain.py`

During the transition, those active package modules delegate to the mature stage implementation under:
- `pipeline/river_workflow/river_workflow_stage_*`
- `pipeline/river_workflow/river_workflow_*_logic.py`

These active-package files are the first place to look for the built-in river workflow.

## Shared workflow infrastructure

Examples:
- authoritative support/materialization helpers
- guidance-domain helpers
- output/final-route contract helpers

These support the active path but are not themselves the primary workflow story.

## Secondary runtime namespaces

Secondary helpers that are still part of the supported runtime now live in clearer subpackages:

- `validation/` — benchmark, seam, nested-AOI, postrun regression, and scientific-validation helpers
- `reporting/` — run-summary, final-reporting, and provenance-reporting writers

These are real runtime modules, but they are intentionally grouped away from the active linear river root so the top-level code tree stays simpler to read.

## Legacy and comparison code

Older river implementations that are not part of the active linear path now live under `legacy/river/`, including:
- `river_v1_*`
- `river_v2_*`
- `simple_river_*`

Older structured / skeleton / XS-era helpers still exist where the runtime compatibility surface still needs them, but the main legacy river tree is now visually separated from the active path.

## Optional debug and report helpers

Active optional diagnostic/report helpers now live under `tools/debug/`, including:
- workflow trace helpers
- connected-diagnostics helpers
- reports-hub helpers
- active river summary/report helpers

Older debug-bundle and consolidated river-debug helpers now live under `legacy/debug/` because they are no longer part of the active diagnostics contract.

These are optional tooling, not the main workflow entry path.

Repo-wide active/legacy/debug language is defined in `repo_runtime_modes.py` and enforced by `repo_contract_checks.py`.

## Documentation spine

Use docs in this order:
1. `README.md`
2. `docs/ACTIVE_WORKFLOW.md`
3. `docs/RUN_OUTPUTS.md`
4. `docs/LEGACY_WORKFLOWS.md`

## Archive

Historical patch notes, older README variants, historical contracts, and older design notes live under `archive/`.
They are preserved for context but are intentionally kept out of the main repo entry path.


## Final-route and final-DEM packaging

The single-writer final DEM path now lives under:

- `pipeline/final_route/final_route_inputs_stage.py`
- `pipeline/final_route/final_route_outputs_stage.py`
- `pipeline/final_route/final_route_contract.py`
- `pipeline/final_dem/final_dem_contract.py`
- `pipeline/final_dem/final_dem_policy.py`
- `pipeline/final_dem/final_dem_contract_validator.py`
