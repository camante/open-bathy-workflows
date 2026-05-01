# REVIEW_NOTES_v893 — Bundle B reapplied after Bundle C

## Goal/invariant

This pass applies Bundle B on top of the latest available v892 package while preserving the Bundle C active-name/legacy-containment cleanup.

The invariant remains:

```text
one canonical river parent -> one AOI export -> one final DEM writer -> one receipt/check per active stage
```

## What changed

- Added a behavior-preserving stage-artifact contract writer around the current active river stage chain.
- Added `reports/river_stage_artifact_contract.json` and `reports/river_stage_artifact_contract.txt` outputs during active river workflow execution.
- Expanded:
  - `river_workflow_contract.py`
  - `river_workflow_receipts.py`
  - `river_workflow_validation.py`
- Wired the contract writer into:
  - the first active stage-chain write,
  - the post-run-summary refresh,
  - and finalization.
- Removed the remaining broad `except Exception` from the active wrapper layer and replaced it with explicit JSON/file-read exception types.
- Added static regression coverage in `tests/test_bundle_b_stage_contract_static.py`.

## What this does not do yet

This still does not split the deeper implementation under `pipeline/river_workflow/` into one physical module per final stage. This pass adds the enforceable contract layer around the active chain; the physical stage-module cleanup remains a later bundle.

## Expected new outputs

After a successful run, each AOI should include:

```text
reports/river_stage_artifact_contract.json
reports/river_stage_artifact_contract.txt
reports/river_stage_chain_summary.json
```

These are traceability products only; they should not alter DEM construction science.
