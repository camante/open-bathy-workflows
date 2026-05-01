# Active Linear Phase 5: Output Ownership

This phase narrows output registration ownership for the active the built-in river workflow route.

## Ownership

- `river_runner.py` owns river-internal outputs and runner-specific metadata.
- `active_pipeline.py` owns active root deliverables:
  - `river_workflow_dem_enhanced`
  - `combined_warped`
  - `final`
  - `river_workflow_run_contract`
  - `river_workflow_pipeline_manifest`
- finalize owns postrun/report artifacts such as `bathy_report.json`, `io_manifest.json`, and curated reports.

The runner no longer owns `final` or `combined_warped`.
