# Review notes — workflow_v899

## Bundle I goal

Quarantine remaining method-style river routing and legacy participation from the active workflow.

## Changes

- Removed `river_method` from active `RiverWorkflowContext`.
- Removed the deprecated `river_method` field from `BathyConfig`.
- Removed `river_method` config flattening from `_load_yaml_config()`.
- Replaced the default YAML `river_method:` block with `river_workflow:`.
- Removed historical method aliases from `repo_runtime_modes.RIVER_WORKFLOW_ALIASES`.
- Removed `--river-method=skeleton` from example/debug shell scripts.
- Updated guidance assembly and run summary reporting so they no longer consult `river_method`.
- Added `docs/RIVER_LEGACY_QUARANTINE_BUNDLE_I.md`.
- Added `tests/test_bundle_i_legacy_method_quarantine_static.py`.

## What remains intentionally

- Historical references remain inside `legacy/`, old review notes, and legacy documentation.
- Skeleton-named science knobs still exist because they are parameter names in older science code. They are not active route selectors.

## Next recommended bundle

Bundle J: explicit canonical-build versus AOI-export-only mode receipts, plus synthetic execution tests for parent/export/final DEM identity.
