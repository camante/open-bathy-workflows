# REVIEW_NOTES_v894

## Scope

This pass reviewed Bundles A, B, and C as delivered in `workflow_v893.zip`, then implemented Bundle D on top of that state.

## Bundle A review

Status: retained with one documentation correction.

The active public entrypoint now uses the river workflow names:

- `run_active_river_workflow(ctx)`
- `finalize_active_river_workflow(ctx, workflow_result)`
- `river_workflow_entry.execute_river_workflow_entry(...)`

Issue found and fixed: `docs/ACTIVE_LINEAR_WORKFLOW_CONTRACT.md` still described the old active-linear names. I added `docs/ACTIVE_RIVER_WORKFLOW_CONTRACT.md` as the current source of truth and converted the old document into a deprecated pointer.

## Bundle B review

Status: retained.

The stage artifact contract layer is present and wired through `active_pipeline.py`:

- `reports/river_stage_chain_summary.json`
- `reports/river_stage_artifact_contract.json`
- `reports/river_stage_artifact_contract.txt`

The contract modules parse cleanly:

- `river_workflow_contract.py`
- `river_workflow_receipts.py`
- `river_workflow_validation.py`

## Bundle C review

Status: retained with one metadata correction.

The active wrappers no longer export old active-linear function aliases. Old linear-v1 names are confined to the legacy archive wrapper and static tests confirm that active wrappers do not expose them.

Issue found and fixed: `active_river_modules.json` still contained stale Bundle A/C notes listing removed aliases as compatibility exports. I updated that manifest so it reflects the current active river workflow state.

## Bundle D implementation

Bundle D focuses on the active WSE path. It does not change the canonical parent/export/final DEM route.

Changes made:

- Added explicit WSE diagnostic construction helpers in `pipeline/river_workflow/river_workflow_stage_wse_proxy.py`:
  - `build_wse_support_artifact(...)`
  - `build_wse_trend_artifact(...)`
  - `build_wse_pre_smooth_artifact(...)`
  - `build_wse_proxy_final_artifact(...)`
- Added WSE one-path contract output:
  - `reports/wse_artifacts/wse_one_path_stage_contract.json`
  - `reports/wse_artifacts/wse_one_path_stage_contract.txt`
- Added the WSE stage contract path to the WSE stage outputs recorded by `river_workflow_pipeline.py`.
- Removed broad `except Exception` handlers from the active WSE proxy stage and replaced them with typed, local exception handling.
- Added static test coverage in `tests/test_bundle_d_wse_one_path_static.py`.

## Important limitation

Bundle D makes the WSE stage easier to diagnose as four explicit steps, but it still lives inside the current `pipeline/river_workflow/river_workflow_stage_wse_proxy.py` module. A later bundle should physically split the WSE code into smaller stage modules after this contract is confirmed in a real run.

## Verification performed

- Python syntax compile for modified active files.
- `bash -n` for all compare scripts.
- Static Bundle A/B/C/D test functions executed directly.
- Confirmed no broad `except Exception` remains in the active wrapper files or active WSE proxy stage.
