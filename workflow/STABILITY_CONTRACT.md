# River Bathymetry Stability Contract (Phase AB)

**Purpose**: Define the exact overlap-identity invariant for river bathymetry, enumerate every
operator's dependency footprint from actual code defaults, identify all current AOI-sensitive
heuristics that violate invariance, and specify the minimum halo, trust classification, scaffold
artifact, and admissibility requirements needed to enforce the invariant in production.

This document is the prerequisite for all scaffold-related implementation work. Nothing in
Phases C–G should be designed before the support numbers in Section 2 are agreed upon.

---

## 1. The Invariant

```
For any two runs R_A and R_B where AOI_B contains AOI_A:

  river_bed(pixel p, R_A) == river_bed(pixel p, R_B)

  for all pixels p ∈ AOI_A ∩ TRUST_ZONE

  if and only if:
    - scaffold_version(R_A) == scaffold_version(R_B)
    - support_contract_version(R_A) == support_contract_version(R_B)
    - effective_hydro_inputs(R_A, neighborhood(p)) == effective_hydro_inputs(R_B, neighborhood(p))
    - config_affecting_scaffold(R_A) == config_affecting_scaffold(R_B)
```

**TRUST_ZONE** is defined in Section 5. Pixels in the boundary band are explicitly excluded from
the invariant guarantee.

**effective_hydro_inputs** covers: NHD flowlines, DEM, WAFFLES water mask, soundings files, and
SWOT RiverSP. Any change to these inputs within `neighborhood(p)` is a permitted reason for
results to differ.

---

## 2. Operator Support Table

All values are derived from current code defaults (see column Source). When a parameter is
user-configurable, the default is used for the minimum halo specification; a wider config requires
a correspondingly larger halo and must be declared in the contract manifest.

### 2a. XS Method Operators

| Operator | Raster support (m) | Along-network support | Junction radius (m) | AOI-sensitive? | Source |
|---|---|---|---|---|---|
| XS placement tangent smoothing | — | `xs_smoothing_window_m` (auto→`xs_spacing_m`/2 = **100 m** one-sided) | — | No (local) | `xs_builder.py:759` |
| Junction exclusion buffer | — | — | **120 m** | No (local) | `xs_builder.py:68` |
| Curvature half-window | **500 m** raster/along-channel | — | — | No (local) | `xs_infer_bathy_raster.py:194` |
| WSE proxy slope (rolling median) | — | `slope_proxy_window=9` XS × `xs_spacing_m=200 m` = **900 m one-sided, 1800 m total** | — | **Yes — truncated at reach ends** | `xs_infer_bathy_raster.py:221` |
| WSE profile fit (river_wse.py) | — | `wse_profile_window=9` XS × 200 m = **900 m one-sided, 1800 m total** | — | **Yes — truncated at reach ends** | `river_wse.py:36` |
| Dmax longitudinal smoothing | — | `smooth_window=7` XS × 200 m = **700 m one-sided, 1400 m total** | — | **Yes — truncated at component ends** | `xs_infer_bathy_raster.py:249` |
| Manning inversion (per-XS) | — | None (point operator) | — | No | `manning_inversion.py` |
| Manning backwater guard (tidal) | — | — | — | **Yes — uses distance_to_tide, which is a reach-level attribute** | `manning_inversion.py:112–113` |
| walid IDW along-channel | `aniso_along_scale_m=500 m` (exponential; effective 3σ ≈ **1500 m**) | — | — | No (local kernel) | `bathy_main.py:984` |
| walid IDW cross-channel | `aniso_cross_scale_m=30 m` (effective 3σ ≈ **90 m**) | — | — | No | `bathy_main.py:985` |
| walid corridor buffer | `continuous_buffer_m` (auto: max(3×px, 150 m) = **150 m** at 10 m res) | — | — | No | `xs_infer_bathy_raster.py:5368` |
| XS soundings influence (XS method) | `river_soundings_max_dist_m=1500 m` | — | — | No (bounded) | `bathy_main.py:851` |
| Component pruning (keep_top=1) | — | **Entire AOI** — takes the largest connected component in the clipped graph | — | **CRITICAL — AOI-sensitive** | `xs_builder.py:987` |
| Global XS deconfliction | — | **Entire AOI** — drops XS that intersect any non-adjacent XS anywhere in the solve domain | — | **CRITICAL — AOI-sensitive** | `xs_builder.py:57,64` |

### 2b. Skeleton Method Operators

| Operator | Raster support (m) | Along-network support | Junction radius (m) | AOI-sensitive? | Source |
|---|---|---|---|---|---|
| Distance-transform channel skeleton | Pixel-local | — | — | No | `river_skeleton_bathy.py` |
| Dmax prior from soundings (skeleton) | `soundings_max_dist_m=10000 m` | — | — | No (bounded, but very wide) | `river_skeleton_bathy.py:2387` |
| Authoritative bed blending | `authoritative_bed_max_dist_m=2000 m` | — | — | No (bounded) | `river_skeleton_bathy.py:2127` |
| Residual blend (soundings) | `residual_blend_sigma_m=120 m` (3σ = **360 m**) | — | — | No | `bathy_main.py:857` |
| WSE field smooth | `wse_smooth_sigma_m=0` (disabled by default) | — | — | — | `bathy_main.py:882` |
| WSE profile smooth | `wse_profile_smooth_sigma_m=200 m` (3σ = **600 m**) | — | — | **Yes — truncated at component ends** | `bathy_main.py:822` |
| SWOT along-channel correction | `swot_correct_sigma_m=2000 m` (3σ = **6000 m**) | — | — | **Yes — truncated at reach ends** | `bathy_main.py:803` |
| Junction buffer | — | — | **120 m** | No | `bathy_main.py:885` |
| Junction smoothing (Gaussian) | — | — | `junction_smooth_sigma_m=80 m` (3σ = **240 m**) | No (local) | `river_skeleton_bathy.py:2317` |
| Largest-component mainstem fallback | — | **Entire AOI** — uses the largest corridor component when stream order is absent | — | **CRITICAL — AOI-sensitive** | `river_domain_mask.py:393–394` |

### 2c. Hybrid Method Additional Operators

| Operator | Raster support (m) | Along-network support | Junction radius (m) | AOI-sensitive? | Source |
|---|---|---|---|---|---|
| Mainstem seed (stream order threshold) | — | **Entire AOI** — expands from seed reaches to connected corridor components | — | **CRITICAL — AOI-sensitive** | `river_domain_mask.py:349,382` |
| XS/skeleton blend at mainstem boundary | `residual_blend_sigma_m=120 m` (3σ = **360 m**) | — | — | No (local) | `bathy_main.py:857` |

### 2d. Post-processing Operators

| Operator | Raster support (m) | Along-network support | Junction radius (m) | AOI-sensitive? | Source |
|---|---|---|---|---|---|
| Gapfill along-river Gaussian smooth | `river_smooth_sigma_m=500 m` (3σ = **1500 m**) | — | — | **Yes — truncated at AOI edges** | `bathy_main.py:1009` |
| Gapfill uncertainty growth | `distance_sigma_scale_m=1500 m` | — | — | No (local model) | `gapfill_intelligent.py:193` |
| SDB tile edge smooth | `tile_edge_smooth_sigma_km=10 km` (3σ = **30 km**) | — | — | **Yes — AOI-edge dependent** | `bathy_main.py:689` |

---

## 3. AOI-Sensitive Heuristics That Currently Violate Invariance

The following behaviors are explicitly conditioned on the clipped AOI and will produce different
results whenever the AOI changes. These are the root causes of non-invariance and must each be
addressed in Phase C–D before the invariant can be claimed.

### 3a. CRITICAL (will always produce different results)

**C1 — AOI-clipped river graph topology**
`river_network.py` builds the graph from NHD flowlines clipped to the requested AOI. Every
downstream component ID, mainstem classification, stationing origin, and XS depends on this
graph. Even a 1-pixel AOI change can change which flowline ends are visible and alter the topology.
*Fix required*: build graph on a canonical stable domain; clip only for export.

**C2 — Component pruning: keep_top_components=1**
`xs_builder.py` retains only the largest connected component(s) by total flowline length within
the clipped graph. A larger AOI may include additional network that changes which component is
"largest," silently dropping or including entire river branches.
*Fix required*: component retention must be decided on the canonical graph, not the clipped one.

**C3 — Global XS deconfliction (global_deconflict_all=True)**
`xs_builder.py` drops XS that geometrically intersect any non-adjacent XS anywhere within the
current solve domain. Adding more flowlines (from a larger AOI) can cause new conflicts that
propagate deconfliction decisions into what was a clean interior in the smaller AOI.
*Fix required*: deconflict on the canonical scaffold at build time; runtime selection is read-only.

**C4 — Mainstem classification from AOI corridor connectivity**
`river_domain_mask.py` seeds the mainstem from stream-order-qualifying reaches and then expands
to connected corridor components. When stream order is absent, it falls back to the largest
corridor component in the current AOI. Both behaviors are AOI-sensitive.
*Fix required*: mainstem classification is a scaffold attribute, not a per-run computation.

**C5 — XS stationing origin**
Cross-section along-channel station (`s_center_m`) is assigned starting from 0.0 at the upstream
end of each clipped reach. A larger AOI that includes the upstream portion of the same reach will
produce different `s_center_m` values for all downstream XS, breaking the entire longitudinal
smoothing chain (Dmax smoothing, WSE profile fit, slope proxy).
*Fix required*: stationing must be assigned on the canonical full-reach graph and preserved.

### 3b. SIGNIFICANT (produce different results near AOI boundaries)

**S1 — Slope proxy rolling window truncation**
`slope_proxy_window=9` XS (~1800 m total) is computed per reach with a centered rolling median.
At reach ends clipped by the AOI boundary, the window is truncated, biasing the slope estimate.
This bias propagates into Manning inversion and Dmax estimation for the nearest ~900 m of XS.
*Minimum halo required*: 900 m along-network past each trust-band boundary.

**S2 — WSE profile fit truncation**
Same as S1 but for the longitudinal WSE profile. `wse_profile_window=9` XS, ~900 m one-sided.

**S3 — Dmax longitudinal smoothing truncation**
`smooth_window=7` XS (~700 m total). Truncated at component ends, biasing the smoothed Dmax
for the nearest ~700 m.
*Minimum halo required*: 700 m along-network (covered by S1/S2 halo).

**S4 — SWOT along-channel correction truncation**
`swot_correct_sigma_m=2000 m` Gaussian (3σ = 6000 m). This is the largest along-network operator.
For any XS within 6000 m of the solve boundary where SWOT is used, the correction will differ
depending on how far the network extends.
*Minimum halo required*: **6000 m along-network** past each trust-band boundary when SWOT is active.

**S5 — Gapfill river smoothing truncation**
`river_smooth_sigma_m=500 m` Gaussian (3σ = 1500 m). Truncated at the AOI raster edge.
*Minimum raster halo required*: 1500 m when gapfill is active.

**S6 — Skeleton soundings Dmax influence (very wide)**
`soundings_max_dist_m=10000 m` in the skeleton method. Any sounding that falls between 0 and
10 km of a channel cell will influence its Dmax prior. This is the dominant raster-space support
when soundings are present.
*Minimum raster halo required*: **10000 m when skeleton + soundings are active**.

**S7 — Manning tidal guard (distance_to_tide)**
`DIST_TO_TIDE_TIDAL=5000 m`, `DIST_TO_TIDE_TRANSITION=15000 m`. These are reach-level
attributes derived from the network. If the network is truncated, some reaches may be missing
their tidal context, causing Manning to be applied where it should be suppressed (or vice versa).
*Fix required*: `distance_to_tide_m` must be a scaffold attribute computed on the full canonical network.

### 3c. MINOR (produce floating-point differences only, not systematic bias)

**M1 — Soundings random subsampling**
`soundings_sample_seed=0` — if subsampling is active with an AOI-conditioned count, the retained
set can differ. With `seed=0` (fixed) and spatial-only downsampling, this is benign if
downsampling is grid-cell-based (not AOI-count-based). **Verify that downsampling is spatial, not
fractional.** If fractional, upgrade to SIGNIFICANT.

**M2 — Floating-point rasterization edge effects**
Rasterio rasterization of vector features near the raster boundary can produce 1-pixel differences
when the raster grid origin shifts. This is addressed by requiring the canonical scaffold to use a
fixed grid origin and resolution aligned to a standard tiling grid (e.g., CUDEM 1/9 arc-second).

---

## 4. Minimum Halo and Solve Domain Requirements

The minimum halo dimensions below are derived from the maximum operator support in each space.
These numbers define how much larger the solve domain must be relative to the export/trust zone.

### 4a. Raster-space halo (metric)

| Condition | Required raster halo |
|---|---|
| No soundings, no gapfill | max(walid along 1500 m, authoritative bed 2000 m) = **2000 m** |
| Gapfill active | max(2000 m, gapfill smooth 1500 m) = **2000 m** |
| Skeleton + soundings active | max(2000 m, skeleton sounding dist 10000 m) = **10000 m** |
| SDB tile edge smooth active | max(above, tile edge smooth 30000 m) = **30000 m** (SDB only) |

**Recommended minimum for river**: **2000 m raster halo** under default config. If skeleton with
soundings, **10000 m**.

### 4b. Along-network halo (from solve boundary in both directions)

| Condition | Required along-network halo |
|---|---|
| No SWOT correction | max(slope_proxy 900 m, WSE profile 900 m, Dmax smooth 700 m) = **900 m** |
| SWOT correction active | max(900 m, SWOT 3σ 6000 m) = **6000 m** |
| Tidal transition zone present | Extend to capture full tidal guard context: min **15000 m** from tidal boundary |

**Recommended minimum for river**: **900 m along-network** under default config without SWOT.
**6000 m** when SWOT is active.

### 4c. Confluence carry-in rule

If a confluence falls within the trust-band boundary (defined in Section 5), the solve domain
**must be extended** along all contributing tributary branches by at least the along-network halo
distance. This applies recursively: if a sub-confluence falls within the extended domain, it also
triggers carry-in.

Rationale: WSE anchoring, slope fitting, and Manning inversion at the mainstem downstream of a
confluence are sensitive to the tributary's hydraulic context. Truncating the tributary inside the
halo degrades the mainstem solution even when the mainstem itself appears fully supported.

---

## 5. Trust Classification

Every exported river cell should carry one of four trust tiers. The trust raster is a required
scaffold output.

| Tier | Code | Condition |
|---|---|---|
| **Fully trusted interior** | 1 | Raster distance from solve boundary ≥ raster halo AND along-network distance from solve boundary ≥ along-network halo AND no confluence carry-in deficit |
| **Raster-supported, hydraulically marginal** | 2 | Raster halo satisfied but along-network halo not fully satisfied (e.g., within 900–6000 m of a solve boundary with SWOT active) |
| **Hydraulically supported, raster-marginal** | 3 | Along-network halo satisfied but within raster halo distance from solve boundary |
| **Untrusted boundary zone** | 0 | Either halo not satisfied OR within confluence carry-in deficit zone |

**Export policy**: only Tier 1 cells should be treated as authoritative for CUDEM gridding. Tier
2–3 cells may be included with explicit uncertainty inflation. Tier 0 cells must be flagged
nodata or explicitly marked as boundary-contaminated.

The trust raster must be recomputed whenever the solve domain boundary changes, and must be stored
as a named output in the scaffold contract manifest.

---

## 6. Scaffold Artifact Definition

The scaffold is a versioned, persisted artifact. Each instance consists of three separable
components.

### 6a. Geometry/data artifact

Persisted outputs that must be stable across runs sharing the same scaffold version:

- `river_graph.gpkg` — canonical directed river graph with reach IDs, component IDs, Strahler order
- `reach_stationing.parquet` — persistent `(reach_id, s_from_m, s_to_m)` table; stationing is
  assigned from the canonical network root, not from clipped AOI ends
- `junction_nodes.gpkg` — junction/confluence locations with type classification
- `xs_scaffold.gpkg` — canonical cross-sections with stable `xs_id`, `s_center_m`, `component_id`;
  generated from the full canonical domain, not the user AOI
- `mainstem_mask.tif` — binary mainstem corridor raster on the canonical grid
- `trust_region.tif` — per-cell trust tier raster (see Section 5)
- `distance_to_tide.parquet` — per-reach tidal distance attribute for Manning guard

### 6b. Generation manifest (`scaffold_generation_manifest.json`)

Answers: **is this scaffold stale?**

Required fields:
```json
{
  "scaffold_version": "<semver string>",
  "schema_version": "<schema semver>",
  "canonical_domain": "<HUC8 ID or bounding box string>",
  "nhd_source": "<NHD version or file fingerprint>",
  "dem_source": "<DEM version or file fingerprint>",
  "waffles_mask_source": "<mask fingerprint>",
  "xs_spacing_m": 200.0,
  "xs_half_width_m": 150.0,
  "junction_buffer_m": 120.0,
  "mainstem_min_order": 5,
  "channel_buffer_m": 400.0,
  "tidal_distance_source": "<NHD reach attribute or external tide model>",
  "created_at": "<ISO 8601 timestamp>",
  "software_version": "<git commit or release tag>",
  "locked": true
}
```

A scaffold is **stale** if any of the following change:
- NHD source or version
- DEM source used for mainstem corridor rasterization
- Any parameter in the generation manifest that affects geometry (spacing, buffer widths, order thresholds)
- Software version changes that alter XS placement or graph construction logic
- `locked` is `false` (scaffold was built experimentally and must not be reused across runs)

### 6c. Contract manifest (`scaffold_contract_manifest.json`)

Answers: **is this scaffold sufficient for the requested inference?**

Required fields:
```json
{
  "scaffold_version": "<must match generation manifest>",
  "raster_halo_m": 2000.0,
  "along_network_halo_m": 900.0,
  "swot_halo_m": 6000.0,
  "skeleton_soundings_halo_m": 10000.0,
  "junction_buffer_m": 120.0,
  "confluence_carry_in_m": 900.0,
  "trust_tiers_present": [0, 1, 2, 3],
  "xs_spacing_m": 200.0,
  "xs_coverage_pct": 97.5,
  "operators_covered": [
    "xs_method", "skeleton_method", "hybrid_method",
    "manning_inversion", "walid_idw", "dmax_smoothing",
    "wse_profile_fit", "slope_proxy", "gapfill_river_smooth"
  ],
  "operators_NOT_covered": [],
  "max_supported_swot_sigma_m": 2000.0,
  "max_supported_soundings_dist_m": 10000.0,
  "tidal_guard_supported": true
}
```

A scaffold is **insufficient** (admissibility gate fails) if:
- The requested `swot_correct_sigma_m` exceeds `max_supported_swot_sigma_m`
- The requested `soundings_max_dist_m` exceeds `max_supported_soundings_dist_m`
- The requested operator is not in `operators_covered`
- The requested export domain invades the Tier 0 zone

---

## 7. Admissibility Gate

The following checks must run at scaffold-load time, before any inference begins. Failure modes
are specified as HARD (abort run) or WARN+SHRINK (reduce trust zone, continue).

| Check | Failure mode | Condition |
|---|---|---|
| Scaffold freshness | **HARD** | Any generation manifest field that affects geometry has changed vs. current config |
| Scaffold lock | **HARD** | `locked == false` in generation manifest |
| Schema version compatibility | **HARD** | Schema version not compatible with current software |
| SWOT sigma vs. contract | **HARD** if violation is in Tier 1 interior; **WARN+SHRINK** if only at boundary | `swot_correct_sigma_m > max_supported_swot_sigma_m` |
| Soundings distance vs. contract | **WARN+SHRINK** | `soundings_max_dist_m > max_supported_soundings_dist_m` |
| Operator coverage | **HARD** | Requested operator not in `operators_covered` |
| Export domain in Tier 0 | **HARD** if strict; **WARN** otherwise | Any export pixel has trust tier 0 |
| Confluence carry-in | **WARN+SHRINK** | Confluence within halo but tributary not present in scaffold |
| XS coverage | **WARN** | `xs_coverage_pct < 90%` in requested export domain |

**WARN+SHRINK** behavior: the trust zone is automatically contracted until all checks pass. The
contracted boundary is recorded in the run report. The operator is notified that the effective
export region is smaller than the requested AOI.

---

## 8. Scaffold Invalidation Rules

The following config changes require a new scaffold version. Changes not listed here require only
a new inference run (scaffold can be reused).

### Requires new scaffold
- NHD source / version change
- `xs_spacing_m` change
- `xs_half_width_m` change
- `junction_buffer_m` change
- `mainstem_min_order` change
- `channel_buffer_m` or `max_channel_width_m` change
- DEM source change (affects mainstem corridor rasterization and trust region geometry)
- Canonical domain boundary change (HUC or bounding box)
- Any change to graph construction logic in `river_network.py` that affects topology
- Any change to `xs_builder.py` that affects XS placement or station assignment
- Addition of new operators with support > current contract manifest values

### Does NOT require new scaffold (inference-only changes)
- Manning prior parameters (`river_mv_a0`, `river_mv_bw`, `river_mv_ba`)
- `river_thalweg_weight` change
- `river_idw_power` or anisotropy scales (within contract limits)
- `swot_correct_sigma_m` change (within contract `max_supported_swot_sigma_m`)
- `soundings_max_dist_m` change (within contract limits)
- `residual_blend_sigma_m` change (within contract limits)
- Addition/removal of soundings files (changes results by design; does not change scaffold validity)
- Gapfill parameters (within contract limits)
- Date range changes
- Output format / CRS changes

### Special case: soundings and WSE anchors
Soundings and SWOT RiverSP are **inference-level inputs**, not scaffold inputs. Their presence
or absence will change results by design and is expected. They do NOT invalidate the scaffold.
However, if a sounding or SWOT file influences Manning prior or WSE anchoring at a range that
exceeds the scaffold's declared `max_supported_soundings_dist_m` or `max_supported_swot_sigma_m`,
the admissibility gate will catch this (Section 7).

---

## 9. Phase Sequencing

The implementation phases that follow from this contract are:

| Phase | Work | Blocking dependency |
|---|---|---|
| **C** | Canonical domain policy (HUC8 unit, neighboring context rules, hydraulic break exceptions) | This document |
| **D** | Scaffold materialization (graph, XS, stationing, mainstem, trust region, manifests) | Phase C + Sections 4–6 of this document |
| **E** | Admissibility gate implementation | Phase D + Section 7 |
| **F** | Deterministic inference refactor (remove C1–C5, make inference scaffold-relative) | Phase E |
| **G** | Nested-AOI regression test (small AOI vs containing AOI, same scaffold, assert Tier 1 equality) | Phase F |

Phase G is not a one-time test. It is a continuous invariant check that must pass on every
CI run that touches river inference logic.

---

## 10. Open Questions (must be resolved before Phase C)

1. **Canonical domain unit**: HUC8 is the recommended starting point. Confirm that HUC8 is
   hydraulically sufficient for the dominant use case (CUDEM 1-degree tiles). Some large river
   systems (Mississippi, Columbia, Colorado) may require HUC6 or explicit multi-HUC context.

2. **Sounding subsampling mode**: Verify that `soundings_sample_seed=0` with current downsampling
   logic produces spatially deterministic subsets (grid-cell-representative, not AOI-fraction-based).
   If any AOI-conditioned count-based subsampling exists, it must be replaced with spatial hashing
   before Phase F.

3. **Schema migration policy**: Decide between (a) all old scaffolds invalidated on schema change,
   (b) migration functions, or (c) multi-version coexistence. Recommendation: option (a) for v0→v1
   transition, with explicit migration support added only if operational cost proves too high.

4. **Tidal distance attribute source**: Confirm whether `distance_to_tide_m` is derived from NHD
   reach attributes, from a separate tide model, or computed from the canonical graph topology.
   This must be a scaffold attribute (Section 6a), not a per-run computation.

5. **Grid alignment**: Confirm the standard raster grid origin and resolution for the scaffold
   trust region and mainstem mask. Recommend aligning to the CUDEM 1/9 arc-second grid origin
   to avoid rasterization edge effects (Issue M2) when reusing scaffold across production tiles.
