# River Workflow Active Contract

This document defines the simplified active river path introduced in workflow v314.

## Primary active river path

1. Canonical river support classification
2. Backbone construction and smoothing
3. Width propagation from a canonical active interior target
4. One primary river guidance surface for final conditioning
5. Support-aware river evaluation with one active benchmark mode

## Canonical river support classes

- `authoritative_interior`
- `bank_margin_only`
- `weak_supported_interior`
- `unsupported_interior`

These classes are the only active top-level river support meanings. Finer distinctions such as mainstem vs side component remain diagnostic-only subordinate fields.

## Canonical active target hierarchy

The active interior target must resolve to exactly one source per station:

- `authoritative_interior`
- `backbone_led_interior`

Bank targets are margin bounds and may remain in diagnostics, but they are not peer active interior targets.

## Primary outputs

- one primary river guidance raster or array
- one backbone smoothing summary
- one width-propagation summary
- one active benchmark mode summary
- one reports manifest that labels artifacts as primary, diagnostic-only, or deprecated

## Diagnostic-only artifacts

These may still be written, but they are not allowed to redefine the active river target hierarchy:

- bank-fit fallback target counts
- section-envelope fallback target counts
- role-agreement and section-target comparison details
- alternate surface/debug tables

## Deprecated-but-still-present paths

These paths may still exist while the simplified contract stabilizes, but they are no longer supposed to drive the active result:

- legacy alternate target ladders
- continuity substitute paths
- historical parallel river surfaces


## v315 routing note

The active river guidance product for final conditioning is `river_primary_surface`. Its summary receipt is `river_primary_guidance_summary.json`, and any continuity/backstop safeguard that keeps the conditioned surface continuous must be reported there as degraded mode rather than treated as an equivalent primary science path.


## v316 evaluation note

The primary benchmark receipt is now `benchmark_river_active_evaluation_summary.json`. It carries the one active river evaluation story for the run. The older receipts `benchmark_river_mode_summary.json`, `benchmark_river_receipt_triage_summary.json`, and `benchmark_river_primary_focus_summary.json` remain available as diagnostic drilldowns, but they are no longer supposed to compete as peer primary interpretations.


## v317 river-shape note

The primary science-side river receipt is now `river_active_shape_summary.json`. It carries one active river-shape story for the run: whether backbone smoothing moved, whether width propagation engaged, the dominant remaining lateral issue, and the suggested next action. The older receipts `river_backbone_smoothing_summary.json`, `river_centerline_width_propagation_summary.json`, `river_role_agreement_summary.json`, and `river_section_target_agreement_summary.json` remain available as diagnostic drilldowns, but they are no longer supposed to compete as peer primary interpretations.


## v318 runtime/product note

- `river_active_runtime_summary.json` is the primary runtime/product receipt for the river path.
- `river_primary_guidance_summary.json` and `river_primary_surface_contract.json` remain available, but only as runtime drilldowns.
- Final-route receipts may still carry richer artifact inventories, but the runtime interpretation should begin with `river_active_runtime_summary.json`.

## v319 river-level summary note

The primary river-level receipt is now `river_active_summary.json`. It carries one active river story for the run by rolling up:

- `river_active_runtime_summary.json`
- `river_active_shape_summary.json`
- `benchmark_river_active_evaluation_summary.json`

The older runtime, shape, and evaluation receipts remain available, but they are now river drilldowns instead of peer primary interpretations. Use `benchmark_river_withheld_support_receipt.json` only when `river_active_summary.json` says a withheld-support benchmark plan is available or recommended.


## v320 simplification note

The active interior target contract is now intentionally exclusive: only `authoritative_interior` and `backbone_led_interior` may populate `active_interior_target_source`. The final route also now treats `river_primary_surface` as the one primary river runtime product; guidance summaries and contracts remain drilldowns.


## v321 runtime simplification note

- Canonical runtime mode continues to allow legacy `river_depth_guidance` as a deprecated input alias only for backward compatibility.
- Under canonical contract modes, that legacy alias is reported as deprecated/rejected rather than treated as a clean primary input path.
- Runtime product roles are narrowed so the primary runtime products are the aligned authoritative base, the conditioned depth, and `river_primary_surface`; audits and contracts remain drilldowns.


## v322 canonical-path enforcement

In canonical contract modes, the interpolator now enforces a single active river-surface path:
- `river_primary_surface` built from the structured channel surface is the only active river runtime surface
- direct `river_depth_guidance` alias use is deprecated and rejected
- native river guide-point rasterization inside the interpolator is deprecated and rejected
- dense candidate rasters are no longer allowed to become peer river-surface substitutes in canonical mode
- when the structured channel surface is absent, canonical mode reports degraded runtime state instead of building a legacy backbone/XS/bank fallback river surface inside the interpolator


## v323 benchmark simplification note

In canonical workflow, support-aware river evaluation is now the default active benchmark path unless a withheld-support benchmark is explicitly applied.

- `support_aware_primary` is the default active river benchmark mode
- `withheld_support_primary` becomes the active mode only when withheld-support evaluation is actually applied
- direct hard-problem holdout coverage remains available only as a diagnostic drilldown, not as a competing active benchmark mode


## v325 minimal reset path

A new `--river-method=v1` minimal reset path exists for support + backbone + one explicit `river_primary_surface` generation without XS/scaffold complexity.
It deliberately stops before surface generation and conditioning, writes only `river_support_points.gpkg`, `river_support_summary.json`, `river_centerline_points.gpkg`, and `river_backbone_summary.json`, and returns no raster for fusion.


## v327 note
In `v327`, the reset path advances beyond artifact-only mode: `--river-method=v1` now writes one explicit `river_primary_surface.tif` and returns it as the single active river raster for fusion.
