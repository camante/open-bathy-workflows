# Review Notes v27 (Focused batch refactor)

## What changed
This cycle is the scheduled batch refactor (every 4th “review again”):

- **Deduplicated vertical datum conversion**: moved `convert_sdb_msl_to_navd88()` into `vdatum_utils.py` and imported it from both `bathy_main.py` and `sdb_main.py`. This prevents drift between two previously-diverging implementations.
- **Standardized subprocess execution**: added `process_utils.py` with `run_cmd()` and `CmdResult` (captured stdout/stderr + tail helpers). Updated:
  - `bathy_main.py` (river DEM auto-download)
  - `sdb_main.py` (internal command runner)
  - `predict.py` (gdalwarp call)
- **Removed unused imports** after the refactor (notably `subprocess` from `sdb_main.py`).

## Verification performed
- `py_compile` across all `*.py` files: **PASS**
- Import sanity on core modules (`bathy_main`, `sdb_main`, `train`, `model_bank`, `vdatum_utils`, `process_utils`, `predict`): **PASS**
- CLI parse `--help` remains valid (unchanged argparse wiring).

## Why this improves quality
- Fewer duplicated “nearly identical” functions → less “AI soup” and fewer divergence bugs.
- One subprocess helper → consistent error handling and safer defaults (`shell=False`, captured output).
- One vdatum conversion implementation → reduces risk of future scientific inconsistency.

## Next high-ROI scientific improvement (not implemented here)
- **Segmented/stratified model banks** (clear-ocean vs turbid vs inland water) and/or depth-bin balancing. This improves cross-tile consistency without mixing incompatible optical regimes.

