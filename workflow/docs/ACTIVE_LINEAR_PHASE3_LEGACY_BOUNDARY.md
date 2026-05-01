# Active Linear Workflow Phase 3

The active river runtime now routes only through the dedicated direct runner.

- Active path: `run_river(...)` calls `river_runner.run_river_workflow_direct(...)`.
- User-facing river method selection has been removed. Active river runs always route through the built-in shared-solve workflow.
- Legacy code may still exist in the repo for reference or archival cleanup, but it is no longer part of the active runtime boundary.
