# River Workflow Active Contract

This file is retained for compatibility with older references.
The authoritative active workflow description is now:

- `docs/ACTIVE_WORKFLOW.md`

## Active river contract summary

The active simple river workflow is the explicit linear path run with:


Stage order:

`entry bundle -> solve_domain -> grids -> authoritative -> centerline -> wse_proxy -> authoritative_bed -> observed_offset -> modeled_offset -> backbone -> corridor -> surface -> lock -> export -> final_dem`

## Normal run output contract

For a normal run, start with:

- `RUN_OVERVIEW.txt`
- `bathy_report.json`
- `io_manifest.json`
- the screen log if needed

Older receipt-heavy or debug-heavy descriptions in this repo should be treated as historical or optional, not as the primary operator path.
