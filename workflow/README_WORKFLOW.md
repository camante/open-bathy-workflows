# Unified SDB + River Bathymetry Workflow (v0.7.8 optionA + science patch5)

This repository contains a unified workflow that can produce:

- **SDB bed elevation** from Sentinel‑2 imagery trained with ICESat‑2 (ATL03/ATL24) and optional extra XYZ.
- **River bed elevation** from cross‑section inference plus priors/anchors (including Option A: USGS discharge‑measurement anchors).
- **A fused surface** where SDB is prioritized near the coast and the river model fills inland channel gaps.

The river side is designed to be *useful even when you have zero in‑river bathymetry* (no sonar, no bathy‑lidar): it builds a river network, generates cross‑sections, then estimates bed elevations using a hierarchy of **soft priors** and **calibration anchors**, each with provenance + uncertainty.

---

## Quick start

### SDB + river (fused)

```bash
python bathy_main.py   --aoi "-88/-87.75/30.5/30.75"   --start 2025-01-01 --end 2026-01-01   --out-dir output/mobile_bay   --methods sdb,river   --priority sdb   --extra-xyz-cudem=hydronos,ehydro   --river-prior-mode multivariate   --river-manning-enabled   --river-manning-region auto   --river-regional-curve-enabled   --river-regional-curve-da-units km2   --river-regional-curve-depth-units m
```

### River only (requires a river/topo DEM)

```bash
python bathy_main.py   --aoi "-74.5/-74.25/40.25/40.5"   --start 2025-01-01 --end 2026-01-01   --out-dir output/raritan   --methods river   --river-dem USACE_2012_NCMP_Lidar_DEM.tif
```

**Auto Manning region:** If you set `--river-manning-region auto` (default), the workflow will reverse-geocode the AOI center
to a US state (via the US Census Geocoder) and map that state to a coarse physiographic region key used by the simplified
Q2 regression in `manning_inversion.py`. You can override the mapping with `--river-manning-region-auto-map path.json`.

---

## What the workflow does

### 1) Orchestration (`bathy_main.py`)

`bathy_main.py` orchestrates both sub‑pipelines and then fuses outputs:

1. **SDB**: calls `sdb_main.py` (download/composite Sentinel‑2, sample training points from ICESat‑2 and any extra XYZ, train RF model, predict bed elevation raster).
2. **River**: builds/loads a river network, generates cross‑sections, infers bed elevations, and rasterizes them.
3. **Fusion**: blends or prioritizes surfaces in overlap using `--fusion-strategy`.

All subprocess commands are logged, and a run‑report JSON is written under the output folder.

### 2) River pipeline (core scientific steps)

The river model tries to infer a plausible channel bed when there is no sonar or bathy‑lidar.

**Inputs (minimum):**
- A **topo DEM** covering the river corridor (`--river-dem`).
- River flowlines (downloaded via TNM by default).

**Steps:**

1. **River network extraction (`river_network.py`)**
   - Builds a river centerline/network inside the AOI.
   - Preserves reach attributes when available (e.g., NHDPlus drainage area).

2. **Cross‑section generation (`xs_builder.py`)**
   - Places cross‑sections along each river_id at `--xs-spacing-m`.
   - Samples bank elevations from the DEM.
   - Computes a wet top‑width estimate and stores cross‑section geometry.

3. **Depth inference + rasterization (`xs_infer_bathy_raster.py`)**
   - Computes an initial **Dmax prior** (maximum depth at the thalweg) from width (and optional reach attributes).
   - Optionally blends in:
     - **Regional curves** (Drainage Area → bankfull depth)
     - **Manning inversion** (Q + width + slope → depth), with a backwater/tidal guard
     - **USGS discharge‑measurement anchors** (Option A)
     - **Width–stage inversion anchors** (if you provide width/stage time‑series)
     - **Direct soundings** (if you have them)
   - Converts Dmax into a cross‑section shape and assigns bed elevation `z_bed = z_wse − depth`.
   - Writes a **bed elevation raster**, plus diagnostic/provenance/uncertainty layers.

4. **Optional monotonic enforcement (`xs_adjust_monotonic.py`)**
   - Enforces downstream‑consistent bed profiles and reduces unphysical oscillations.

---

## River depth priors and anchors

### A. Base Dmax prior

`xs_infer_bathy_raster.py` supports:

- `--prior-mode powerlaw` (default):
  - `Dmax = a * W^b`

- `--prior-mode multivariate`:
  - `Dmax = a0 * W^bw * (DA + eps_a)^ba * (S + eps_s)^bs`
  - where DA is drainage area, S is slope.

### B. Regional curves (Drainage Area → bankfull depth)

If you have coefficients from a published USGS/state “regional curve” of the form:

`Dbkf = c * DA^f`

…you can blend that information in as a **soft prior**:

- Enable: `--river-regional-curve-enabled`
- Provide coefficients (recommended): `--river-regional-curve-c`, `--river-regional-curve-f`
- Specify units: `--river-regional-curve-da-units {km2|mi2}`, `--river-regional-curve-depth-units {m|ft}`

**Important:** “bankfull depth” is usually a *mean* depth, not necessarily Dmax. The pipeline converts to Dmax either:

- `--river-regional-curve-to-dmax auto` (default): uses the trapezoid mean→Dmax conversion (depends on bottom_width_frac)
- `--river-regional-curve-to-dmax factor`: use a fixed factor (e.g., 1.25)

Built‑in region keys exist only so the workflow runs end‑to‑end; for defensible science, supply your published coefficients.

### C. Manning inversion (Q + width + slope → depth), with backwater/tidal guard

Manning inversion is a **physics prior**. It can improve depths where slope and discharge are meaningful, but it must be down‑weighted in tidal/backwater zones.

Enable (alias):

- `--river-manning-enabled` (equivalent to `--river-manning-mode q2_regional`)

Or explicitly:

- `--river-manning-mode constant` with `--river-manning-q-cms`
- `--river-manning-mode from_field` with `--river-manning-q-field`
- `--river-manning-mode q2_regional` with `--river-manning-region`

The workflow computes a Manning confidence and blends it into Dmax using `--river-manning-max-weight`, automatically reduced when:

- Slope is very low (tidal/backwater)
- Distance‑to‑mouth/tide is small (if the attribute exists)

### D. Option A: USGS discharge measurement anchors (high ROI)

Many USGS gaging stations have **discrete discharge measurements** that include width and cross‑sectional area (velocity–area method). From these, the pipeline derives an effective mean depth:

`Dmean = Area / Width`

It then fits a site‑specific scaling of the prior and blends that adjustment into nearby cross‑sections.

Enable in `bathy_main.py`:

```bash
--river-usgs-sites 01646500,01651000 --river-usgs-start 2020-01-01 --river-usgs-end 2025-12-31 --river-usgs-max-dist-m 5000 --river-gage-snap-max-dist-m 1000
```

---

## Outputs

Typical outputs under `--out-dir`:

- `sdb/…/*.tif` – SDB bed elevation raster(s)
- `river/river_bed.tif` – river bed elevation raster
- `river/river_bathy.gpkg` – cross‑section parameters + bed points + diagnostics
- `run_report.json` – commands + status + tails of stdout/stderr

Within `river_bathy.gpkg`, the `xs_params` layer includes columns such as:

- `dmax_prior_m` – final blended Dmax prior used for the cross‑section
- `dmax_regional_curve_m`, `regional_curve_wt` – regional curve contribution
- `manning_dmax_m`, `manning_wt`, `manning_conf`, `manning_q_cms_used` – Manning prior diagnostics
- `a_site_*` – USGS anchor fit diagnostics (when enabled)

---

## Major scientific limitations (and how the code mitigates them)

1. **Backwater/tidal zones violate Manning assumptions**
   - Mitigation: slope‑based confidence, optional distance‑to‑mouth guard, and max blend weight.

2. **Regional curves are region‑specific and often defined at bankfull**
   - Mitigation: treated as soft priors with explicit uncertainty + weight cap; requires user‑supplied coefficients for serious use.

3. **Drainage area and slope attributes may be missing or inconsistent**
   - Mitigation: the code searches common NHDPlus field names and can compute a slope proxy from WSE trend along the river.

4. **Cross‑section shape is simplified** (trapezoid/compound approximation)
   - Mitigation: anchors (USGS, width–stage) push the parameters toward locally plausible shapes; monotonic smoothing stabilizes longitudinal artifacts.

5. **Widths from masks/geometry can be biased** (vegetation, turbidity, resolution)
   - Mitigation: width‑ratio checks around USGS sites; conservative blending when width mismatch is large.

6. **Outputs are conditional on the reference WSE** (water surface elevation)
   - Mitigation: the workflow carries provenance and uncertainty rasters; future improvements should explicitly normalize to a reference discharge.

## Automating USGS published coefficients (Q2 regressions & bankfull curves)

This pipeline can now **auto-select** published (or user-supplied) coefficients based on the AOI centroid:

- **Q2 (2‑year peak flow) regression** used by the optional Manning inversion prior
- **Bankfull depth vs drainage area** (“regional curve”) used by the regional-curve depth prior

### Where coefficients live

All coefficient registries live in `sdb_config.json` under:

```json
{
  "river": {
    "nss_services": {...},
    "registry": {
      "auto_region_rules": { "AL": {"q2":"AL_default", "bankfull":"AL_default"}, ... },
      "q2_regressions": { "AL_default": {"c":..., "f":..., "da_units":"km2", "q_units":"cms", "source":"..."}, ... },
      "bankfull_depth_curves": { "AL_default": {"c":..., "f":..., "da_units":"km2", "depth_units":"m", "source":"..."}, ... }
    }
  }
}
```

The `*_default` entries shipped with this repo are **placeholders**; replace them with coefficients from USGS regional curves / state regressions for your area.

### How AOI → state is detected

When you run `bathy_main.py` (or `xs_infer_bathy_raster.py` directly), the code:

1) computes the AOI centroid from `--aoi=w/e/s/n`
2) reverse‑geocodes to a **US state** using the Census coordinate geocoder (no API key required)
3) selects the coefficient set from `auto_region_rules`

### NSS Services integration (best-effort)

If enabled, the pipeline will also make a best‑effort call to the USGS **NSS Services** API to locate the *Q2* regression metadata for the AOI state (responses are cached under `cache/nss`). Because NSS Services response schemas can change and some statistics require additional basin characteristics, the pipeline treats NSS results as **metadata** unless a power‑law `(c,f)` is available.

You can disable NSS lookups via:

```json
"river": { "nss_services": { "enabled": false } }
```

### Recommended workflow for “production” coefficients

1) Start with your AOI state(s) and locate the USGS report/regression used by StreamStats/NSS for Q2 and bankfull depth.
2) Add entries to `q2_regressions` and `bankfull_depth_curves` with:
   - `c`, `f`
   - unit tags (`da_units`, `q_units` / `depth_units`)
   - `source` citation (report title / DOI)
3) Map your state to the correct keys in `auto_region_rules`.

This keeps the pipeline deterministic and reproducible even if external services change.

