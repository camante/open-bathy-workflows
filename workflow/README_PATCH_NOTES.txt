v268 — Exception Logging Sweep + Reference Test Hardening
==========================================================

A. SILENT EXCEPTION LOGGING (274 blocks → 97.3% coverage)
-----------------------------------------------------------
Converted 274 bare `except Exception:` blocks across 45 production files
to emit `log.debug("<context>: suppressed exception", exc_info=True)`.

Before: silent swallowing masked root causes (stacked depth-limiting bugs,
sign convention failures, and training data issues all went undetected).

After: every suppressed exception leaves a breadcrumb in debug logs. The
remaining 19 unlogged blocks are intentional: logging_config.py bootstrap
(6), smoke_test.py test infrastructure (3), and test-file import guards (10).

Worst offenders fixed:
  51  xs_infer_bathy_raster.py
  21  bathy_main.py
  21  river_skeleton_bathy.py
  14  geo/raster_ops.py
  11  s2_optics.py
  11  sdb_main.py
  10  train.py
  10  river_structured_scaffold.py
   7  bathy_fusion.py

Files that needed logger object added: river_structured_scaffold.py,
river_guidance.py, river_masking.py, river_longitudinal_profile.py,
final_dem_contract_validator.py, plot_utils.py, swot_riversp_fetch.py,
cudem_authoritative.py (fixed LOG vs log mismatch), guidance_domains.py
(fixed LOG vs log mismatch), dominant_trunk.py (fixed LOG vs log).

B. MERRIMACK REFERENCE SPEC TIGHTENED
---------------------------------------
- Expanded from 2 to 6 check points covering estuary channel (deep, mid),
  nearshore shallows, upstream river bed, and inland reach.
- Bounds tightened: estuary channel now [-18, -2]m (was [-30, +5]m).
- Added sign_checks: estuary points must have negative bed elevation.
- Added monotonicity_checks: downstream must be deeper than upstream.
- These catch sign flips, depth-cap regressions, and conditioning blend
  weight shifts that the old loose bounds would miss.

C. REFERENCE TEST ENHANCED (test_merrimack_reference.py)
---------------------------------------------------------
- Now validates sign_checks from spec (catches sign convention failures).
- Now validates monotonicity_checks (catches depth-cap and channel
  inversion regressions).
- Removed incorrect @pytest.mark.heavy decorator from helper function.

D. CONTRACT TESTS EXECUTION (new: test_contract_tests_execution.py)
--------------------------------------------------------------------
- Added pytest test that actually executes ContractTestSuite.run_all()
  with synthetic data and validates output structure.
- Tests: runs with synthetic context, empty context (graceful failure),
  xyz_not_provided skip path, and result field validation.
- The smoke test only checked loading; now the test suite verifies
  that each contract test class produces well-formed results.

REMAINING TOP 3 IMPROVEMENTS:
1. SDB library-call interface (SDBResult dataclass) — eliminate subprocess
   boundary, return Python objects instead of filesystem-coupled manifests.
2. Along-channel anisotropic distance in river corridors — the existing
   _nearest_surface_within_domain has the centerline routing but should
   weight along-channel neighbors more heavily at meander bends.
3. Inverse-variance blending propagation — the terrain interpolator has
   the machinery but guidance_uncertainty is still NaN in some paths,
   causing fallback to heuristic weights.
