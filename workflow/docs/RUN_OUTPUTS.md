# Run Outputs

This document defines the normal run output contract for the workflow.

## Normal run

A normal run should be read through these files first:

- `RUN_OVERVIEW.txt`
- `reports/README_FIRST.txt`
- `reports/WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt`
- `reports/WORKFLOW_INPUT_OUTPUT_TRACE.txt`
- `reports/river_active_summary.json`
- `bathy_report.json` when you need machine-readable detail
- `io_manifest.json` when you need exact file paths
- `run_logs/screen_*.log` when the run failed or the result looks wrong

These files are the intended entry point for understanding what happened.

## Meaning of the retained files

### `RUN_OVERVIEW.txt`
Human-readable first file.
Use it to answer:
- did the run succeed
- what methods ran
- what final deliverables were written
- what to inspect next

### `bathy_report.json`
Machine-readable run summary.
Use it for:
- component statuses
- recorded outputs
- run-level metadata
- traceability summary state

### `io_manifest.json`
Explicit input/output path list for the run.
Use it when you need the exact file paths that were consumed or produced.

### `run_logs/screen_*.log`
The main human log.
Use it when the run failed or when you need the chronological runtime story.

## What is not part of the normal run contract

A normal run should not require you to inspect:
- `io_manifest.md`
- `unified_bathy_report.json`
- old workflow trace/debug text files
- large receipt or manifest trees

Those are not part of the intended normal-run operator path.

## Intermediate/debug runs

When `--save-intermediates` or other explicit debug/validation modes are enabled, the run may also write:
- stage artifacts
- receipts
- manifests
- validation summaries
- seam/nested-AOI comparison products

Those outputs are optional and should be read only when you are doing a deeper investigation.
