# open-bathy-workflows – Test Suite

## Overview

54 tests across three files. No network or geospatial library dependencies required
(rasterio and pyproj are mocked; all raster I/O uses tifffile backed by numpy arrays).

## Running

```bash
cd tests/
python3 -m unittest test_helpers test_fusion_atl test_pipeline -v
```

Or individually:

```bash
python3 -m unittest test_helpers -v      # 29 tests, ~0.1s  pure-Python helpers
python3 -m unittest test_fusion_atl -v  # 7 tests,  ~5s    training + fusion
python3 -m unittest test_pipeline -v    # 18 tests, ~17s   predict_scene end-to-end
```

## What is covered

### test_helpers.py (29 tests)
| Class | Tests |
|---|---|
| `TestParseAoiBbox` | comma / space / whitespace parsing, garbage, None, short input, negative coords |
| `TestNwaterGuard` | None skips processing, 0 triggers skip, positive doesn't skip |
| `TestAtlColumnMapping` | standard, short (lon/lat/depth), XYZ column names |
| `TestDepthSignDetection` | positive→flip, negative→unchanged, mixed→unchanged, all-NaN→no crash |
| `TestChunkedSiblingCopy` | all siblings copied, missing siblings skipped gracefully |
| `TestConfidenceMath` | DOA×optical_q×unc_q in [0,1]; half-weight at 1.5 m reference; zero DOA→zero |
| `TestDepthSignGuard` | negative depths pass, positive detected, abs() always positive |
| `TestNodataCollisionGuard` | nodata=0 keeps water pixels, nodata=-9999 excluded, None→no exclusion |

### test_fusion_atl.py (7 tests)
| Class | Tests |
|---|---|
| `TestFusionMerge` | both sources in output, None atl24 handled, depths stay negative |
| `TestSourceFractionGuardrail` | single-source domination warns but trains successfully |
| `TestNodataCollisionGuard` | nodata=0 / -9999 / None guard behaviour |

### test_pipeline.py (18 tests)
| Class | Tests |
|---|---|
| `TestTrainSdbModel` | RF returned, feature columns in meta, train/test split covers input, RF predicts positive magnitudes, input depths are negative, max_depth key in meta, too-few-points graceful |
| `TestPredictScene` | depth raster written, confidence on/off, provenance on/off, uncertainty always written, provenance invariant (where prov==1, depth finite and ≠ nodata), depths are negative-down, min_confidence_threshold reduces coverage, threshold-without-confidence emits WARNING |
| `TestPredictChunked` | output raster produced, stumpf_lr.pkl discovered from model_dir |

## Architecture

`conftest.py` installs all mocks into `sys.modules` at import time so no geo
libraries are needed. The rasterio mock wraps tifffile for real array I/O — code
paths that read/write GeoTIFFs exercise real indexing and windowing logic, not stubs.

Fixtures build a 64×64 synthetic scene (S2 bands + land mask) and a 10-tree RF
model with saved artifacts matching exactly what `predict_scene` expects to load.
Session-scoped `setUpClass` means fixtures are built once and shared across tests
in each class.
