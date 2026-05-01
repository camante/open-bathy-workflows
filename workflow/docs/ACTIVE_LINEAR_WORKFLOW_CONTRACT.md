# Deprecated active-linear workflow contract name

This filename is retained only as a compatibility breadcrumb for older review notes. The active workflow contract is now documented in:

- `docs/ACTIVE_RIVER_WORKFLOW_CONTRACT.md`

The active entrypoints are:

- `river_workflow_entry.execute_river_workflow_entry(...)`
- `active_pipeline.run_active_river_workflow(ctx)`
- `active_pipeline.finalize_active_river_workflow(ctx, workflow_result)`

Do not add new active behavior to this deprecated document.
