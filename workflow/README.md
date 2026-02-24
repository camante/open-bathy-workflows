# Smoke tests

These are **offline sanity checks** intended to catch obvious breakages (syntax, imports, basic math/geometry logic).
They do **not** validate scientific correctness or data-dependent behavior.

## Run

```bash
./verify_repo.sh
# or
./run_smoke.sh
```

Both scripts are offline and should not download anything.

## What is tested

1) **Compile-only**: compiles all `*.py` files (`tests/test_compile.py`)

2) **River XS deconflict**: constructs two intersecting synthetic cross-sections + one non-intersecting,
then verifies `_global_deconflict_xs_all` drops the harmful intersection deterministically (`tests/smoke_test_river_xs.py`)

3) **WSE profile fitting (SWOT-like)**: fits a monotone-smoothed WSE profile from a noisy synthetic series
and checks the result is broadly non-increasing downstream (`tests/smoke_test_river_skeleton_swot.py`)

## Notes

- For compatibility with older docs, `./tests/run_smoke.sh` is a thin wrapper that calls `../run_smoke.sh`.
