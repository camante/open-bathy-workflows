Terrain Interpolation Engine
===========================

Phase 6 introduces an explicit support-aware terrain interpolation engine.

Purpose
-------
The engine is responsible for generating the final continuous terrain surface
under authoritative hard-control while letting subordinate SDB and river
guidance shape eligible gaps.

Contract
--------
Inputs:
- authoritative hard-control raster
- candidate terrain raster
- support masks for SDB and river domains
- optional guidance weights / trusted-interior masks
- optional river authoritative support depth anchors

Behavior:
- authoritative finite cells are locked
- only authoritative gaps are eligible for interpolation
- SDB and river guidance act as confidence-weighted subordinate influence
- river channel structure guidance areas can elevate support class to scaffold_inferred
- remaining gaps use a deterministic nearest-valid backstop to maintain a
  continuous raster, reported as low_confidence_continuous_fill when needed

Outputs:
- conditioned terrain raster
- support class raster
- provenance raster
- support distance / density diagnostics
- guidance influence diagnostics
- coastal and river confidence diagnostics

Module ownership
----------------
`terrain_interpolator.py` owns the array-level deterministic interpolation
behavior. `authoritative_conditioning.py` remains a compatibility wrapper so
existing code paths and tests do not break while the workflow is migrated to
the new engine-oriented architecture.
