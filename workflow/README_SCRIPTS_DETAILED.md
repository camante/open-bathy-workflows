# Script reference (v2.0.2)

This document describes each Python script in the v2.0.2 zip (typically placed in `workflow/`).
For each script:
- **Pipeline** indicates whether it supports **SDB**, **River**, **Fusion**, or shared infrastructure.
- **Invocation** notes whether it is called directly (CLI), called by `bathy_main.py`, or imported as a module.


## bathy_main.py

- **Pipeline**: SDB, River, Fusion
- **Size**: 3365 lines
- **Invocation**: Can be run as a CLI script
- **What it does**: bathy_main.py – Unified Coastal + River Bathymetry Pipeline
- **Key CLI args (subset)**: --aoi, --start, --end, --out-dir, --methods, --priority, --cloud, --icesat, --sdb-mode, --cache-root, --align-mode, --glint-correct, --glint-nir-band, --glint-vis-bands …
- **Output artifacts (hints)**: .tmp_copy.tif, .tmp_masked.tif, SDB_RF_10m.tif, SDB_RF_10m_ALIGNED.tif, LAND_MASK_aligned.tif, manifest.json, river_bed_elev_patch.tif, river_depth_terrain_patch.tif
- **Depends on (internal modules)**: bathy_fusion, checkpoints, contract_tests, gapfill_intelligent, logging_config, manning_inversion, predict_parallel, region_resolver, river_diagnostics, run_summary, validation

## sdb_main.py

- **Pipeline**: SDB
- **Size**: 2415 lines
- **Invocation**: Called by `bathy_main.py` via subprocess; Can be run as a CLI script
- **What it does**: sdb_main.py – The Master Orchestrator for the Modular SDB Pipeline.
- **Key CLI args (subset)**: --aoi, --start, --end, --out-dir, --cache-root, --cache-strict, --no-cache-strict, --cache-code-strict, --cache-ignore-code, --no-cache-ignore-code, --s2-module, --sdb-mode, --icesat, --cloud …
- **Output artifacts (hints)**: sdb_config.json, run_report.json, eval_metrics_raster_vs_icesat_test.json, S2_DATE_QC.json, RGB_10m.tif, LAND_MASK_aligned.tif, ATL03_tracks.shp, training_data_fused.csv
- **Depends on (internal modules)**: atl, checkpoints, fusion, log_report, logging_config, predict, predict_chunked, predict_parallel, train, validation, vis

## bathy_fusion.py

- **Pipeline**: Fusion
- **Size**: 979 lines
- **Invocation**: Can be run as a CLI script; Imported by: bathy_main
- **What it does**: bathy_fusion.py – Multi-Source Bathymetry Fusion
- **Key CLI args (subset)**: --sdb, --river, --measured, --dem, --sdb-uncert, --river-uncert, --out-dir, --strategy, --priority, --taper, --template
- **Output artifacts (hints)**: BATHY_FUSED.tif, BATHY_FUSED_PROVENANCE.tif, BATHY_FUSED_UNCERTAINTY.tif
- **Depends on (internal modules)**: chunked_processing

## fusion.py

- **Pipeline**: Fusion
- **Size**: 730 lines
- **Invocation**: Can be run as a CLI script; Imported by: sdb_main
- **What it does**: fusion.py – Hierarchical fusion of ATL03, ATL24, and extra XYZ training data
- **Key CLI args (subset)**: --atl03-csv, --atl24-csv, --xyz-csv, --out, --xyz-max-dist-m, --xyz-max-abs-diff-m, --atl-max-dist-m, --atl-max-abs-diff-m, --atl-max-rel-diff, --atl-rel-gate-m, --atl03-weight, --atl24-weight, --xyz-weight
- **Output artifacts (hints)**: training_fused.csv
- **Depends on (internal modules)**: spatial_sampling

## xs_builder.py

- **Pipeline**: River

**New in v2.0.2 (artifact reduction):** by default, xs_builder now (1) densifies centerlines before tangent estimation, (2) skips cross-sections near confluences/junctions, and (3) drops cross-sections that still intersect non-adjacent cross-sections within a reach. You can disable these with `--no-skip-junctions` and/or `--no-global-deconflict`, and tune tolerances with `--junction-buffer-m` and `--deconflict-tol-m`.
- **Size**: 755 lines
- **Invocation**: Called by `bathy_main.py` via subprocess; Can be run as a CLI script
- **What it does**: xs_builder.py – Build river cross-sections (XS) from a river network + DEM/topo rasters
- **Key CLI args (subset)**: --river-gpkg, --dem, --topo-lidar, --out-gpkg, --out-csv, --rivers-layer, --edges-layer, --spacing-m, --half-width-m, --sample-step-m, --bank-search-m, --min-centerline-len-m, --smoothing-window-m, --trim-overlaps …
- **Depends on (internal modules)**: logging_config

## xs_infer_bathy_raster.py

- **Pipeline**: River
- **Size**: 4143 lines
- **Invocation**: Called by `bathy_main.py` via subprocess; Can be run as a CLI script
- **What it does**: xs_infer_bathy_raster.py – Infer river bathymetry from cross-sections and rasterize a DEM-aligned bed patch
- **Key CLI args (subset)**: --xs-gpkg, --out-gpkg, --dem, --xs-lines-layer, --xs-points-layer, --soundings, --extra-xyz, --soundings-depth-col, --soundings-elev-col, --soundings-x-col, --soundings-y-col, --calib-max-dist-m, --calib-stat, --usgs-sites …
- **Depends on (internal modules)**: constants, logging_config, manning_inversion, river_wse, usgs_nwis

## river_network.py

- **Pipeline**: River
- **Size**: 1094 lines
- **Invocation**: Called by `bathy_main.py` via subprocess; Can be run as a CLI script
- **What it does**: river_network.py – Download/prepare a river network for cross-section generation (TNM/NHD with HydroRIVERS fallback)
- **Key CLI args (subset)**: --aoi, --out-gpkg, --cache-dir, --out-crs, --simplify-m, --nhd-flowlines, --layer, --id-field, --keep-fields, --tnm-enable, --no-tnm, --tnm-dataset, --tnm-formats, --tnm-max-items …
- **Depends on (internal modules)**: logging_config

## river_domain_mask.py

- **Pipeline**: River
- **Size**: 246 lines
- **Invocation**: Called by `bathy_main.py` via subprocess; Can be run as a CLI script
- **What it does**: river_domain_mask.py — Build a *river-only* raster mask (river vs. open water) using:   1) NHD flowlines (from river_network.gpkg) as the river "seed", and   2) a water/land mask (ideally waffles coastline mask: water=0, land=1) as the boundary.
- **Key CLI args (subset)**: --river-gpkg, --rivers-layer, --template-raster, --water-mask, --ocean-mask, --channel-buffer-m, --max-channel-width-m, --mainstem-min-order, --max-mainstem-width-m, --write-debug, --out-channel-mask, --out-open-water-mask
- **Output artifacts (hints)**: debug_width_proxy_m.tif, debug_d_center_m.tif, debug_corridor_mask.tif, debug_mainstem_corridor_mask.tif, debug_skeleton_mask.tif

## river_skeleton_bathy.py

- **Pipeline**: River
- **Size**: 798 lines
- **Invocation**: Called by `bathy_main.py` via subprocess; Can be run as a CLI script
- **What it does**: river_skeleton_bathy.py — Channel-skeleton, distance-transform river bathymetry (NO cross-sections).
- **Key CLI args (subset)**: --river-gpkg, --rivers-layer, --template-raster, --dem, --channel-mask, --authoritative-bed-raster, --authoritative-bed-max-dist-m, --residual-blend-sigma-m, --out-bed, --shape-exp, --dmax-min-m, --dmax-max-m, --prior-mode, --mv-a0 …
- **Output artifacts (hints)**: depth_m.tif, wse_m.tif, r.tif, d_bank_m.tif, d_center_m.tif, dmax_m.tif, snd_depth_obs_m.tif, snd_bed_obs_m.tif

## alignment.py

- **Pipeline**: SDB, Fusion
- **Size**: 875 lines
- **Invocation**: Imported by: predict
- **What it does**: alignment.py – Post-prediction residual alignment of an SDB raster to reference tie points.

## atl.py

- **Pipeline**: SDB
- **Size**: 1584 lines
- **Invocation**: Can be run as a CLI script; Imported by: sdb_main, vis
- **What it does**: atl.py – ICESat‑2 (ATL03/ATL24) utilities for the Open Bathy Workflows
- **Key CLI args (subset)**: --aoi, --start, --end, --product, --cache-dir, --out-csv, --land-mask
- **Depends on (internal modules)**: cache_utils, log_report

## bottom_physics.py

- **Pipeline**: SDB
- **Size**: 1057 lines
- **Invocation**: Imported by: physics_integration, predict
- **What it does**: bottom_physics.py - Physics-Based SDB Enhancements
- **Depends on (internal modules)**: constants

## cache_utils.py

- **Pipeline**: SDB, River, Fusion
- **Size**: 321 lines
- **Invocation**: Imported by: atl, s2_optics
- **What it does**: cache_utils.py — Deterministic, exact-match cache fingerprints for the SDB pipeline.

## checkpoints.py

- **Pipeline**: SDB, River, Fusion
- **Size**: 719 lines
- **Invocation**: Can be run as a CLI script; Imported by: bathy_main, sdb_main
- **What it does**: checkpoints.py - Pipeline Checkpoint and Resume Capability
- **Depends on (internal modules)**: constants, logging_config

## chunked_processing.py

- **Pipeline**: SDB, River, Fusion
- **Size**: 567 lines
- **Invocation**: Imported by: bathy_fusion, predict_chunked
- **What it does**: chunked_processing.py - Memory-Efficient Tile-Based Raster Processing
- **Depends on (internal modules)**: constants

## constants.py

- **Pipeline**: SDB, River, Fusion
- **Size**: 399 lines
- **Invocation**: Imported by: bottom_physics, checkpoints, chunked_processing, predict, predict_parallel, train …
- **What it does**: constants.py - Centralized Constants for SDB + River Bathymetry Pipeline

## contract_tests.py

- **Pipeline**: SDB, River, Fusion
- **Size**: 525 lines
- **Invocation**: Can be run as a CLI script; Imported by: bathy_main
- **What it does**: Contract Tests for SDB Pipeline

## errors.py

- **Pipeline**: SDB, River, Fusion
- **Size**: 337 lines
- **Invocation**: Imported by: validation
- **What it does**: errors.py - Custom exception hierarchy for SDB pipeline

## gapfill_intelligent.py

- **Pipeline**: Fusion
- **Size**: 1202 lines
- **Invocation**: Can be run as a CLI script; Imported by: bathy_main
- **What it does**: gapfill_intelligent.py - Physics-Informed Gap-Filling for Bathymetry
- **Key CLI args (subset)**: --prior, --hq, --out, --sigma, --prov, --prior-sigma, --water-mask, --river-mask, --bank-elev, --xs-gpkg, --method, --rbf-function, --cudem-xyz, -v

## kd_estimation.py

- **Pipeline**: SDB
- **Size**: 672 lines
- **Invocation**: Can be run as a CLI script; Imported by: sdb_uncertainty, train
- **What it does**: kd_estimation.py - Physics-based water clarity and depth penetration estimation
- **Key CLI args (subset)**: --b02, --b03, --b04, --b08, --algorithm, --output-json, --output-qc-raster, --depth-raster
- **Depends on (internal modules)**: logging_config

## log_report.py

- **Pipeline**: SDB, River, Fusion
- **Size**: 485 lines
- **Invocation**: Can be run as a CLI script; Imported by: atl, s2_optics, sdb_main, train
- **What it does**: log_report.py — Lightweight "flight recorder" for the SDB pipeline.
- **Key CLI args (subset)**: --out-dir

## logging_config.py

- **Pipeline**: SDB, River, Fusion
- **Size**: 183 lines
- **Invocation**: Imported by: bathy_main, checkpoints, kd_estimation, manning_inversion, physics_integration, predict …
- **What it does**: logging_config.py - Centralized Logging Configuration for SDB Pipeline

## manning_inversion.py

- **Pipeline**: River
- **Size**: 736 lines
- **Invocation**: Can be run as a CLI script; Imported by: bathy_main, xs_infer_bathy_raster
- **What it does**: manning_inversion.py - Manning's Equation Inversion for River Depth Estimation
- **Key CLI args (subset)**: --discharge, --width, --slope, --manning-n, --dist-tide, --elevation, --json
- **Depends on (internal modules)**: logging_config

## nss_services.py

- **Pipeline**: Shared/Infra
- **Size**: 200 lines
- **Invocation**: Imported by: region_resolver
- **What it does**: nss_services.py - Minimal client for USGS StreamStats NSS Services.

## physics_integration.py

- **Pipeline**: SDB, Fusion
- **Size**: 627 lines
- **Invocation**: Can be run as a CLI script; Imported by: predict, train
- **What it does**: physics_integration.py - Integration of Kim et al. (2024) Physics Improvements
- **Key CLI args (subset)**: --s2-dir, --output-dir, --kd, --sza, --vza, --max-depth
- **Depends on (internal modules)**: bottom_physics, logging_config

## predict.py

- **Pipeline**: SDB
- **Size**: 1000 lines
- **Invocation**: Can be run as a CLI script; Imported by: predict_chunked, predict_parallel, sdb_main
- **What it does**: predict.py – SDB inference engine (Scene-wide prediction)
- **Key CLI args (subset)**: --s2-dir, --model-dir, --land-mask-path, --out-tif, --sdb-mode, --tile-size, --smooth-kernel, --nad83, --no-doa, --doa-threshold, --doa-soft-k, --write-doa-score, --land-mask-type, --land-mask-water-val …
- **Depends on (internal modules)**: alignment, bottom_physics, constants, logging_config, physics_integration, sdb_uncertainty

## predict_chunked.py

- **Pipeline**: SDB
- **Size**: 388 lines
- **Invocation**: Can be run as a CLI script; Imported by: sdb_main
- **What it does**: predict_chunked.py - Memory-efficient chunked prediction wrapper
- **Key CLI args (subset)**: --model-dir, --out-dir, --bands, --land-mask, --max-depth, --max-memory, --tile-size, --overlap, --force-chunked
- **Depends on (internal modules)**: chunked_processing, logging_config, predict

## predict_parallel.py

- **Pipeline**: SDB
- **Size**: 774 lines
- **Invocation**: Can be run as a CLI script; Imported by: bathy_main, sdb_main
- **What it does**: predict_parallel.py - Parallel Tile Processing for SDB Prediction
- **Key CLI args (subset)**: --b02, --b03, --b04, --b08, --model, --meta, --output, --uncertainty, --land-mask, --workers, --tile-size, --overlap, --estimate
- **Depends on (internal modules)**: constants, logging_config, predict

## region_resolver.py

- **Pipeline**: SDB, River
- **Size**: 159 lines
- **Invocation**: Imported by: bathy_main
- **What it does**: region_resolver.py - Resolve AOI -> State -> Regression/Curve coefficients.
- **Depends on (internal modules)**: nss_services

## river_diagnostics.py

- **Pipeline**: River
- **Size**: 287 lines
- **Invocation**: Imported by: bathy_main
- **What it does**: river_diagnostics.py - Diagnostic utilities for River Bathymetry Pipeline
- **Depends on (internal modules)**: river_report

## river_report.py

- **Pipeline**: River
- **Size**: 445 lines
- **Invocation**: Imported by: river_diagnostics
- **What it does**: River Pipeline Report Generation

## river_wse.py

- **Pipeline**: River
- **Size**: 178 lines
- **Invocation**: Imported by: xs_infer_bathy_raster
- **What it does**: river_wse.py – Fit a smoothed longitudinal water-surface elevation (WSE) profile along river stationing

## run_summary.py

- **Pipeline**: SDB, River, Fusion
- **Size**: 179 lines
- **Invocation**: Imported by: bathy_main
- **What it does**: Human-readable run summary.

## s2_optics.py

- **Pipeline**: SDB
- **Size**: 2679 lines
- **Invocation**: Can be run as a CLI script
- **What it does**: s2_optics.py — Weighted Shared-Date Composite (STRICT default)
- **Key CLI args (subset)**: --aoi, --start, --end, --out-dir, --max-cloud, --scene-limit, --preferred-months, --shared-date-mode, --stac-url, --collection, --stac-limit, --stac-page-limit, --stac-max-items, --stac-chunk-months …
- **Depends on (internal modules)**: cache_utils, log_report, logging_config

## sdb_uncertainty.py

- **Pipeline**: SDB, Fusion
- **Size**: 1054 lines
- **Invocation**: Can be run as a CLI script; Imported by: predict
- **What it does**: sdb_uncertainty.py - Uncertainty-Aware Satellite Derived Bathymetry
- **Key CLI args (subset)**: --s2-dir, --output-dir, --region, --mode
- **Depends on (internal modules)**: kd_estimation, logging_config

## spatial_cv.py

- **Pipeline**: SDB
- **Size**: 992 lines
- **Invocation**: Can be run as a CLI script; Imported by: train
- **What it does**: spatial_cv.py - Spatial Cross-Validation for SDB Training
- **Key CLI args (subset)**: --input-csv, --feature-cols, --target-col, --strategy, --n-folds, --output-dir, --seed
- **Depends on (internal modules)**: logging_config

## spatial_sampling.py

- **Pipeline**: SDB, Fusion
- **Size**: 1032 lines
- **Invocation**: Imported by: fusion
- **What it does**: Sophisticated Adaptive Spatial Sampling for Bathymetry Training Data

## train.py

- **Pipeline**: SDB
- **Size**: 2566 lines
- **Invocation**: Can be run as a CLI script; Imported by: sdb_main
- **What it does**: train.py – SDB model training with Feature Importance & Robust Metadata
- **Key CLI args (subset)**: --train-csv, --out-model-dir, --s2-dir, --cw-min, --land-max, --max-depth, --s2-choice, --s2-ab-rmse-margin, --rmse-target-sdb, --depth-bin-m, --depth-binning, --max-depth-bins, --min-samples-per-bin, --validate-spatial …
- **Depends on (internal modules)**: constants, kd_estimation, log_report, physics_integration, spatial_cv, training_diversity

## training_diversity.py

- **Pipeline**: SDB
- **Size**: 649 lines
- **Invocation**: Can be run as a CLI script; Imported by: train
- **What it does**: training_diversity.py - Analyze and improve training data spatial/spectral diversity
- **Key CLI args (subset)**: --input-csv, --output-dir, --max-depth
- **Depends on (internal modules)**: logging_config

## usgs_nwis.py

- **Pipeline**: River
- **Size**: 295 lines
- **Invocation**: Imported by: xs_infer_bathy_raster
- **What it does**: usgs_nwis.py – Fetch USGS NWIS site metadata and discharge measurement records (RDB parser)

## validation.py

- **Pipeline**: SDB, River, Fusion
- **Size**: 769 lines
- **Invocation**: Can be run as a CLI script; Imported by: bathy_main, sdb_main
- **What it does**: validation.py - Centralized Input Validation for SDB Pipeline
- **Depends on (internal modules)**: constants, errors, logging_config

## vis.py

- **Pipeline**: SDB
- **Size**: 467 lines
- **Invocation**: Can be run as a CLI script; Imported by: sdb_main
- **What it does**: vis.py – Visual QC utilities for ICESat‑2 (ATL03) and training/prediction artifacts
- **Key CLI args (subset)**: --csv, --atl03-dir, --out-dir, --gpkg-full, --gpkg-slim, --no-gpkg, --water-temp-c, --wavelength-nm, --depth-mode, --max-depth-sdb, --model-meta, --aoi
- **Depends on (internal modules)**: atl, logging_config
