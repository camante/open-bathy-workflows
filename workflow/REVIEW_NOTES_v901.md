# Review notes for workflow_v901

Bundle K final hygiene pass on top of workflow_v900.

## Main invariant checked

The active river workflow remains the single canonical parent -> AOI export -> final DEM route. This pass did not change river science or final DEM construction.

## Fixes made

- Removed stale root-level Bundle A/B compatibility shims that duplicated active river contract modules:
  - `river_workflow_context.py`
  - `river_workflow_contract.py`
  - `river_workflow_paths.py`
  - `river_workflow_receipts.py`
  - `river_workflow_validation.py`
- Moved the active stage-artifact contract helper into the real active package:
  - `pipeline/river_workflow/river_workflow_stage_artifact_contract.py`
- Updated `active_pipeline.py` to import stage-contract helpers from the active package instead of root compatibility modules.
- Updated `repo_contract_checks.py` / `repo_runtime_modes.py` so the repo contract check reflects the current package layout rather than stale debug-module expectations.
- Added `tests/test_bundle_k_final_hygiene_static.py`.

## Verification summary

- `repo_contract_checks.py --root .` passes.
- Active construction surface has no `except Exception` or bare `except:` hits.
- Root-level stale compatibility shims are gone.
- `compare.sh`, `final_compare.sh`, and `compare_5aois_union_final_products_parent_ref_v12.sh` pass shell syntax checks.

## Remaining limitation

The package still contains historical review notes, docs, legacy manifests, and tests that mention old names for explanation or quarantine verification. Those are not active construction code. A later documentation-only pass can reduce historical note volume if desired, but it should not touch active workflow behavior.
