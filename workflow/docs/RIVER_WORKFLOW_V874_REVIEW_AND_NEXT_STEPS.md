# River workflow v874 review and next steps

## Current assessment

The active river route is now close to the desired simple framework:

1. canonical parent DEM
2. AOI export DEM
3. final user DEM

The important invariant is that an AOI run should not construct a different river answer. It should export from the same canonical parent solution and then materialize the final DEM from that named AOI export artifact.

The v873 comparison log showed the core seam and writer contracts passing for the Merrimack north/south test:

- same canonical river system
- same canonical cache key
- same canonical parent hash
- AOI export versus parent passed
- combined DEM versus AOI export passed
- single final DEM writer passed
- north/south overlap had zero mismatch pixels

## Issues fixed in v874

### 1. Export-only AOIs now read retained canonical-parent science receipts

The run summary previously only looked for stage receipts directly attached to the AOI export run. In an export-only AOI run, that can make read-only science checks appear as skipped even though the canonical parent manifest may reference the immutable parent-stage receipts.

The summary now reads the retained canonical manifest and uses its canonical construction stage receipt paths when direct AOI stage receipts are not present.

### 2. Skipped science/support checks now say why they were skipped

Skipped checks no longer appear as empty `{}` diagnostics. If a check is skipped, the summary includes an explicit reason such as:

- `stage_science_receipts_not_retained_or_not_available_in_this_aoi_export`
- `support_or_composition_counts_not_found_in_retained_receipts`

This keeps the workflow honest without inventing science metrics.

### 3. Canonical manifest composition counts are now surfaced as support/composition context

The active workflow receipt and run summary now look for `composition_summary` in the retained canonical manifest. This gives export-only AOI runs a stable place to report finite/support/guidance/background pixel counts without scanning rasters or rerunning construction.

### 4. Runner logs now distinguish requested role from effective route

The runner previously logged `role=canonical_build` even when the downstream pipeline immediately found an existing canonical parent and performed export-only routing. The log now records the requested role before handoff and the effective route after the parent/export decision.

## Remaining river cleanup before copying the framework to SDB

### Phase 0 — Freeze the river contract as the template

Keep this as the non-negotiable contract:

- one canonical solve domain
- one canonical parent solution
- AOI runs export only from the canonical parent
- one final DEM writer
- final DEM source is the named AOI export
- retained receipts prove parent/export/final identity

Do not add a second river method, fallback route, or final DEM patch path.

### Phase 1 — Make reporting boring and complete

Every run should produce one readable river summary, one canonical manifest, one AOI export identity receipt, one final output receipt, and one workflow trace. Missing optional science checks should explain whether the evidence is missing, not retained, or not applicable.

### Phase 2 — Quarantine inactive legacy river code

Create an explicit active-path map and legacy-path map. The active path should import only the built-in canonical parent/export runner and its required stage modules. Legacy modules can remain for reference, but active execution should not discover or call them implicitly.

### Phase 3 — Keep `bathy_main.py` as orchestration only

Move remaining river-specific construction, receipt, and final-folder logic out of `bathy_main.py` into small pipeline modules. The desired structure is:

- CLI and configuration parsing
- route selection
- river runner call
- final receipt/report handoff

### Phase 4 — Make canonical science receipts first-class parent artifacts

The canonical parent build should retain compact science receipts for WSE, observed offset, modeled offset, backbone, primary surface, authoritative lock, and final parent DEM composition. AOI exports should read these receipts only; they should not recompute any science.

### Phase 5 — Add contract tests that run without full science data

Add lightweight tests for:

- cache hit export-only route
- no-cache canonical build route
- stale parent hash rejection
- missing manifest rejection
- AOI export identity
- single-writer enforcement
- summary/report skip reasons

### Phase 6 — Only then mirror the pattern in SDB

SDB should use the same pattern:

1. canonical/source-domain SDB guidance build
2. AOI export/subset from retained parent guidance
3. final materialization from named export artifact
4. immutable receipts proving source/export/final identity

Do not let SDB write the final DEM directly until it has the same parent/export/final contract.

### Phase 7 — Reintroduce combined/fusion only after river and SDB share the contract

Fusion should combine retained guidance artifacts under one final-writer policy. It should not bring back competing final DEM writers.
