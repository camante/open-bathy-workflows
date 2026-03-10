# Workflow checklist (A-grade, enforceable)

This workflow checklist is intended to stay short, testable, and hard to misread.

## Required before packaging or committing

- All `*.py` files pass `py_compile`.
- `./verify_repo.sh` passes.
- `./ci_smoke.sh` passes.
- The repo tree contains no committed `__pycache__`, `.pytest_cache`, `*.pyc`, or similar cache artifacts.
- Documentation does not claim canonical output filenames when the run manifest is the true source of record.
- New outputs mentioned in docs are either written by code or clearly described as conditional/optional.
- Deliverable depth rasters and bed rasters are described with the correct sign/datum interpretation.
- River outputs remain clipped to the river/channel domain; coastal outputs remain clipped to their intended water domain.
- Any new mask, datum, CRS, training-source, or seam-related change is reflected in docs and receipts/manifests.
- Changes are minimal, readable, and free of duplicate or dead documentation blocks.
