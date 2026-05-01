# Active Linear Phase 10 — Export Border Identity Checks

This phase adds direct utilities and tests for validating that two AOI exports:

1. point at the same canonical solve-state
2. have matching border cells where the AOIs touch

The intended use is north/south AOIs on the same river system.

## Utilities

- `load_export_run_identity(run_dir)`
  - reads the run's stage summary and bathy report
  - resolves the canonical solve contract path
  - resolves the final deliverable path
  - loads canonical identity hashes

- `compare_north_south_export_identity(north_run_dir, south_run_dir)`
  - compares canonical cache key
  - compares canonical solve grid hash
  - compares canonical raster stack hashes
  - compares north/south touching border cells

## Purpose

This is the first direct run-level utility for checking whether two AOI exports are truly reusing the same canonical solve artifacts and whether their exported border cells agree.
