# REVIEW_NOTES_v898

## Basis

Built from `workflow_v897.zip`.

## Immediate error fixed

`pipeline/final_dem_materialization.py` recorded source/destination SHA-256 hashes but did not define `_sha256_file()`, causing:

```text
NameError: name '_sha256_file' is not defined
```

The helper is now defined in the materialization module and is used by the final-route guard to prove that `combined/DEM_enhanced.tif` is a byte-for-byte copy of the named AOI export.

## Bundle H implemented

Bundle H physically separates the explicit WSE four-step artifact/contract helpers from the large WSE proxy construction module.

New module:

```text
pipeline/river_workflow/river_workflow_wse_steps.py
```

It owns:

```text
build_wse_support_artifact()
build_wse_trend_artifact()
build_wse_pre_smooth_artifact()
build_wse_proxy_final_artifact()
write_wse_stage_contract()
```

`river_workflow_stage_wse_proxy.py` still owns the science construction logic, but the WSE contract/artifact helper layer is now physically isolated, making the support -> trend -> pre-smooth -> proxy-final chain easier to inspect and eventually split further.

## Remaining limitation

The WSE science logic is still mostly inside `river_workflow_stage_wse_proxy.py`. This pass did not rewrite the WSE profile fitting algorithm; it only moved the explicit WSE step artifact/contract builders into a dedicated module and fixed the final materialization hash bug.
