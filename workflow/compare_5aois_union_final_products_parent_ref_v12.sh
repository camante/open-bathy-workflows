#!/usr/bin/env bash
set -euo pipefail

trap 'echo "ERROR: $0 failed at line $LINENO" >&2' ERR

usage() {
  cat >&2 <<'EOF'
Usage: $0 <name> <version_tag> [aoi_count] [extra bathy_main args ...]
       $0 <name> <version_tag> --aoi-count <2|3|4|5> [extra bathy_main args ...]

AOI count controls how many fixed Merrimack review AOIs are run and compared:
  2 = north, south
  3 = north, south, contained
  4 = north, south, contained, downstream_overlap
  5 = north, south, contained, downstream_overlap, upstream_overlap

The version tag may be passed with or without a leading v. Extra arguments are
forwarded unchanged to bathy_main.py.
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

NAME="$1"
VERSION_TAG="${2#v}"
shift 2

AOI_COUNT="5"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --aoi-count|--aois|--num-aois)
      [[ $# -ge 2 ]] || { echo "Missing value after $1" >&2; usage; exit 2; }
      AOI_COUNT="$2"
      shift 2
      ;;
    --aoi-count=*|--aois=*|--num-aois=*)
      AOI_COUNT="${1#*=}"
      shift
      ;;
    2|3|4|5)
      AOI_COUNT="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$AOI_COUNT" in
  2|3|4|5) ;;
  *) echo "AOI count must be 2, 3, 4, or 5; got: $AOI_COUNT" >&2; usage; exit 2 ;;
esac

START_DATE="2025-01-01"
END_DATE="2026-01-01"
ROOT_OUT="output/${NAME}_v${VERSION_TAG}"
FINAL_COMPARE_DIR="${ROOT_OUT}/final_comparison"
SOURCE_COMPARE_DIR="${ROOT_OUT}/final_comparison_sources"
mkdir -p "$ROOT_OUT"
rm -rf "$FINAL_COMPARE_DIR" "$SOURCE_COMPARE_DIR"
mkdir -p "$FINAL_COMPARE_DIR" "$SOURCE_COMPARE_DIR"

AOI_LABELS_ALL=(north south contained downstream_overlap upstream_overlap)
AOI_LABELS=("${AOI_LABELS_ALL[@]:0:${AOI_COUNT}}")
declare -A AOIS
AOIS[north]='-71.15/-71.04/42.75/42.78'
AOIS[south]='-71.15/-71.04/42.70/42.75'
AOIS[contained]='-71.13/-71.07/42.72/42.76'
AOIS[downstream_overlap]='-71.15/-71.04/42.735/42.78'
AOIS[upstream_overlap]='-71.15/-71.04/42.70/42.765'

declare -A OUT_DIRS
for label in "${AOI_LABELS[@]}"; do
  OUT_DIRS[$label]="${ROOT_OUT}/${label}"
done

SELECTED_AOIS=()
for label in "${AOI_LABELS[@]}"; do
  SELECTED_AOIS+=("${AOIS[$label]}")
done

SOLVE_DOMAIN=$(python3 - "${SELECTED_AOIS[@]}" <<'PY_SOLVE'
import sys
if len(sys.argv) <= 1:
    raise SystemExit("No AOIs supplied")
aois = []
for item in sys.argv[1:]:
    try:
        w, e, s, n = (float(part) for part in item.split('/'))
    except Exception as exc:
        raise SystemExit(f"Invalid AOI string {item!r}: {exc}") from exc
    aois.append((w, e, s, n))
print(f"{min(a[0] for a in aois)}/{max(a[1] for a in aois)}/{min(a[2] for a in aois)}/{max(a[3] for a in aois)}")
PY_SOLVE
)

echo "=== AOI comparison count: ${AOI_COUNT} (${AOI_LABELS[*]}) ==="
echo "=== Shared solve domain from union(selected AOIs): ${SOLVE_DOMAIN} ==="

for label in "${AOI_LABELS[@]}"; do
  out_dir="${OUT_DIRS[$label]}"
  echo "=== Running ${label} AOI: ${AOIS[$label]} -> ${out_dir} ==="
  mkdir -p "$out_dir"
  ./bathy_main.py \
    --aoi="${AOIS[$label]}" \
    --solve-domain="${SOLVE_DOMAIN}" \
    --start="${START_DATE}" \
    --end="${END_DATE}" \
    --out-dir="$out_dir" \
    "${EXTRA_ARGS[@]}"
  echo "=== Finished ${label} ==="
done

DEM_PATHS=()
AUTH_PATHS=()
RAW_PARENT_REF_PATH=""

for label in "${AOI_LABELS[@]}"; do
  out_dir="${OUT_DIRS[$label]}"
  final_dir="${out_dir}/final"
  dem_src="${final_dir}/DEM_enhanced.tif"
  auth_src="${final_dir}/authoritative_base_aligned.tif"
  parent_src="${final_dir}/canonical_parent_dem.tif"

  [[ -f "$dem_src" ]] || { echo "Missing $dem_src" >&2; exit 1; }
  [[ -f "$auth_src" ]] || { echo "Missing $auth_src" >&2; exit 1; }
  [[ -f "$parent_src" ]] || { echo "Missing $parent_src" >&2; exit 1; }

  dem_dst="${SOURCE_COMPARE_DIR}/DEM_enhanced_${label}.tif"
  auth_dst="${SOURCE_COMPARE_DIR}/authoritative_base_aligned_${label}.tif"
  cp -f "$dem_src" "$dem_dst"
  cp -f "$auth_src" "$auth_dst"
  DEM_PATHS+=("$dem_dst")
  AUTH_PATHS+=("$auth_dst")
  echo "Copied source DEM and authoritative DEM for ${label} -> ${SOURCE_COMPARE_DIR}"

  if [[ -z "$RAW_PARENT_REF_PATH" ]]; then
    RAW_PARENT_REF_PATH="${SOURCE_COMPARE_DIR}/canonical_parent_dem_reference_raw.tif"
    cp -f "$parent_src" "$RAW_PARENT_REF_PATH"
  fi
done

if ! command -v gdalbuildvrt >/dev/null 2>&1; then
  echo "Missing required command: gdalbuildvrt" >&2
  exit 1
fi
if ! command -v gdalwarp >/dev/null 2>&1; then
  echo "Missing required command: gdalwarp" >&2
  exit 1
fi
if ! command -v gdaldem >/dev/null 2>&1; then
  echo "Missing required command: gdaldem" >&2
  exit 1
fi
if ! command -v perspecto >/dev/null 2>&1; then
  echo "Missing required command: perspecto" >&2
  exit 1
fi

gdalbuildvrt -overwrite "${FINAL_COMPARE_DIR}/DEM_enhanced_all_aois.vrt" "${DEM_PATHS[@]}" >/dev/null
echo "Built DEM_enhanced_all_aois.vrt"
gdalbuildvrt -overwrite "${FINAL_COMPARE_DIR}/authoritative_DEM_all_aois.vrt" "${AUTH_PATHS[@]}" >/dev/null
echo "Built authoritative_DEM_all_aois.vrt"

IFS='/' read -r UNION_W UNION_E UNION_S UNION_N <<< "$SOLVE_DOMAIN"

# Requested lean, union-AOI, geographic reference rasters. The AOI exports are
# commonly in the projected working CRS, so do not require source rasters to be
# geographic. Explicitly warp the enhanced parent to EPSG:4269 over the union
# AOI, then force the original DEM to the same geographic grid dimensions and
# bounds for direct visual comparison.
gdalwarp -overwrite -of GTiff -t_srs EPSG:4269 -te_srs EPSG:4269   -te "$UNION_W" "$UNION_S" "$UNION_E" "$UNION_N"   -r near   "$RAW_PARENT_REF_PATH" "${FINAL_COMPARE_DIR}/DEM_enhanced.tif" >/dev/null
echo "Built final_comparison/DEM_enhanced.tif as union AOI in geographic coordinates"

read TARGET_WIDTH TARGET_HEIGHT < <(python3 - "${FINAL_COMPARE_DIR}/DEM_enhanced.tif" <<'PY_GRID'
import sys
import rasterio
with rasterio.open(sys.argv[1]) as ds:
    if not ds.crs or not ds.crs.is_geographic:
        raise SystemExit(f"Expected warped DEM_enhanced.tif to be geographic, got {ds.crs}")
    print(ds.width, ds.height)
PY_GRID
)

gdalwarp -overwrite -of GTiff -t_srs EPSG:4269 -te_srs EPSG:4269   -te "$UNION_W" "$UNION_S" "$UNION_E" "$UNION_N"   -ts "$TARGET_WIDTH" "$TARGET_HEIGHT" -r near   "${FINAL_COMPARE_DIR}/authoritative_DEM_all_aois.vrt" "${FINAL_COMPARE_DIR}/DEM_original.tif" >/dev/null
echo "Built final_comparison/DEM_original.tif as union AOI in geographic coordinates on the DEM_enhanced.tif grid"

python3 - "${FINAL_COMPARE_DIR}" "${FINAL_COMPARE_DIR}/DEM_enhanced.tif" "${FINAL_COMPARE_DIR}/DEM_original.tif" <<'PY_CPT'
import json, os, sys
import numpy as np
import rasterio

out_dir = sys.argv[1]
paths = sys.argv[2:]
mins, maxs = [], []
for path in paths:
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True)
        vals = arr.compressed()
        if vals.size:
            mins.append(float(np.nanmin(vals)))
            maxs.append(float(np.nanmax(vals)))
if not mins:
    raise SystemExit("No finite values found while computing shared CPT range")
vmin = min(mins)
vmax = max(maxs)
if vmax <= vmin:
    vmax = vmin + 1.0

stops = [
    (0.00, (10, 24, 110)),
    (0.10, (20, 60, 170)),
    (0.20, (40, 120, 220)),
    (0.30, (80, 190, 255)),
    (0.40, (160, 230, 255)),
    (0.50, (235, 225, 190)),
    (0.60, (170, 210, 140)),
    (0.70, (100, 165, 95)),
    (0.80, (165, 140, 100)),
    (0.90, (205, 190, 170)),
    (1.00, (255, 255, 255)),
]
cpt_path = os.path.join(out_dir, "shared_dem_color_table.cpt")
with open(cpt_path, "w", encoding="utf-8") as f:
    for frac, rgb in stops:
        value = vmin + frac * (vmax - vmin)
        f.write(f"{value:.6f} {rgb[0]} {rgb[1]} {rgb[2]}\n")
    f.write("nv 0 0 0 0\n")
with open(os.path.join(out_dir, "shared_dem_color_table_range.json"), "w", encoding="utf-8") as f:
    json.dump({"min": vmin, "max": vmax, "inputs": paths}, f, indent=2)
print(cpt_path)
PY_CPT

echo "Generated shared_dem_color_table.cpt in ${FINAL_COMPARE_DIR}"

gdaldem color-relief -alpha \
  "${FINAL_COMPARE_DIR}/DEM_enhanced.tif" \
  "${FINAL_COMPARE_DIR}/shared_dem_color_table.cpt" \
  "${FINAL_COMPARE_DIR}/DEM_enhanced_shared_cpt.tif" >/dev/null

gdaldem color-relief -alpha \
  "${FINAL_COMPARE_DIR}/DEM_original.tif" \
  "${FINAL_COMPARE_DIR}/shared_dem_color_table.cpt" \
  "${FINAL_COMPARE_DIR}/DEM_original_shared_cpt.tif" >/dev/null

echo "Generated shared-CPT review rasters for DEM_enhanced.tif and DEM_original.tif"

(cd "${FINAL_COMPARE_DIR}" && perspecto "DEM_enhanced.tif" >/dev/null)
(cd "${FINAL_COMPARE_DIR}" && perspecto "DEM_original.tif" >/dev/null)
[[ -f "${FINAL_COMPARE_DIR}/DEM_enhanced_hillshade.tif" ]] || { echo "Missing perspecto hillshade for DEM_enhanced.tif" >&2; exit 1; }
[[ -f "${FINAL_COMPARE_DIR}/DEM_original_hillshade.tif" ]] || { echo "Missing perspecto hillshade for DEM_original.tif" >&2; exit 1; }
echo "Generated perspecto hillshades for DEM_enhanced.tif and DEM_original.tif"

echo "=== Pairwise metadata/export identity comparisons for all AOI pairs ==="
PAIR_JSONS=()
for ((i=0; i<${#AOI_LABELS[@]}; i++)); do
  for ((j=i+1; j<${#AOI_LABELS[@]}; j++)); do
    a="${AOI_LABELS[$i]}"
    b="${AOI_LABELS[$j]}"
    json_out="${ROOT_OUT}/${a}_vs_${b}.json"
    txt_out="${ROOT_OUT}/${a}_vs_${b}.txt"
    echo "=== Comparing metadata/export identity ${a} vs ${b} ==="
    python3 tools/compare_aoi_exports.py \
      --json-out "$json_out" \
      --text-out "$txt_out" \
      "${OUT_DIRS[$a]}" "${OUT_DIRS[$b]}"
    PAIR_JSONS+=("$json_out")
  done
done

python3 - "${ROOT_OUT}/pairwise_compare_summary.json" "${PAIR_JSONS[@]}" <<'PY_PAIR'
import json, sys
out_json = sys.argv[1]
inputs = sys.argv[2:]
failures = []
results = {}
for path in inputs:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    passed = data.get("passed", data.get("status") == "PASS" or data.get("overall_status") == "PASS")
    results[path] = {"passed": bool(passed)}
    if not passed:
        failures.append({"path": path, "data": data})
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"all_passed": len(failures) == 0, "results": results, "failures": failures}, f, indent=2)
if failures:
    print("Metadata/export identity comparison failures:", file=sys.stderr)
    for item in failures:
        print(f"- {item['path']}", file=sys.stderr)
    raise SystemExit(1)
PY_PAIR

echo "=== O(n) raster math: compare each AOI final DEM to canonical parent reference ==="
python3 - "${ROOT_OUT}/final_dem_parent_reference_raster_math_summary.json" "${RAW_PARENT_REF_PATH}" "${SOURCE_COMPARE_DIR}" "${AOI_LABELS[@]}" <<'PY_RASTER'
import json, os, sys
import numpy as np
import rasterio
from rasterio.windows import Window

out_json = sys.argv[1]
parent_path = sys.argv[2]
source_dir = sys.argv[3]
labels = sys.argv[4:]
TOL = 0.0
results = {}
failures = []

def same_grid_spacing(a, b, eps=1.0e-12):
    return abs(a.a - b.a) <= eps and abs(a.b - b.b) <= eps and abs(a.d - b.d) <= eps and abs(a.e - b.e) <= eps

def is_integerish(value, eps=1.0e-6):
    return abs(value - round(value)) <= eps

def mask_array(arr):
    mask = np.ma.getmaskarray(arr)
    if mask.shape == ():
        return np.zeros(arr.shape, dtype=bool)
    return mask

with rasterio.open(parent_path) as parent:
    parent_crs = parent.crs
    parent_transform = parent.transform
    for label in labels:
        dem_path = os.path.join(source_dir, f"DEM_enhanced_{label}.tif")
        with rasterio.open(dem_path) as ds:
            item = {"path": dem_path, "comparison_mode": "aoi_subset_window_against_canonical_parent"}
            if ds.crs != parent_crs:
                item.update({"passed": False, "reason": "crs_mismatch", "dem_crs": str(ds.crs), "parent_crs": str(parent_crs)})
                failures.append((label, item)); results[label] = item; continue
            if not same_grid_spacing(ds.transform, parent_transform):
                item.update({"passed": False, "reason": "grid_spacing_or_rotation_mismatch", "dem_transform": tuple(ds.transform), "parent_transform": tuple(parent_transform)})
                failures.append((label, item)); results[label] = item; continue

            col_offset_f = (ds.transform.c - parent_transform.c) / parent_transform.a
            row_offset_f = (ds.transform.f - parent_transform.f) / parent_transform.e
            if not (is_integerish(col_offset_f) and is_integerish(row_offset_f)):
                item.update({"passed": False, "reason": "aoi_origin_not_aligned_to_parent_grid", "col_offset_float": col_offset_f, "row_offset_float": row_offset_f})
                failures.append((label, item)); results[label] = item; continue

            col_offset = int(round(col_offset_f))
            row_offset = int(round(row_offset_f))
            if col_offset < 0 or row_offset < 0 or col_offset + ds.width > parent.width or row_offset + ds.height > parent.height:
                item.update({"passed": False, "reason": "aoi_window_outside_parent_grid", "window": [col_offset, row_offset, ds.width, ds.height], "parent_shape": [parent.height, parent.width]})
                failures.append((label, item)); results[label] = item; continue

            win = Window(col_offset, row_offset, ds.width, ds.height)
            d = ds.read(1, masked=True)
            p = parent.read(1, window=win, masked=True)
            if d.shape != p.shape:
                item.update({"passed": False, "reason": "shape_mismatch", "dem_shape": d.shape, "parent_shape": p.shape})
                failures.append((label, item)); results[label] = item; continue

            dmask = mask_array(d)
            pmask = mask_array(p)
            mask_mismatch = int(np.count_nonzero(dmask != pmask))
            valid = (~dmask) & (~pmask)
            compared = int(valid.sum())
            if compared == 0:
                item.update({"passed": False, "reason": "no_valid_overlap_pixels", "mask_mismatch_pixels": mask_mismatch})
                failures.append((label, item)); results[label] = item; continue

            diff = np.abs(d.data[valid] - p.data[valid])
            mismatch = int(np.count_nonzero(diff > TOL))
            item.update({
                "passed": mismatch == 0 and mask_mismatch == 0,
                "parent_window": {"col_off": col_offset, "row_off": row_offset, "width": ds.width, "height": ds.height},
                "compared_pixels": compared,
                "mask_mismatch_pixels": mask_mismatch,
                "mismatch_pixels_gt_tolerance": mismatch,
                "tolerance": TOL,
                "max_abs_diff": float(diff.max()) if diff.size else 0.0,
            })
            if not item["passed"]:
                item["reason"] = "nonzero_pixel_or_mask_difference"
                failures.append((label, item))
            results[label] = item

with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"all_passed": len(failures) == 0, "results": results, "failures": [{k:v} for k,v in failures]}, f, indent=2)

text_path = os.path.splitext(out_json)[0] + ".txt"
with open(text_path, "w", encoding="utf-8") as f:
    f.write(f"all_passed: {len(failures) == 0}\n")
    for label, item in results.items():
        f.write(f"{label}: {item}\n")

if failures:
    print("Raster math failures against canonical parent reference:", file=sys.stderr)
    for label, item in failures:
        print(f"- {label}: {item}", file=sys.stderr)
    raise SystemExit(1)
PY_RASTER

python3 - "${ROOT_OUT}/run_status.json" "${SOLVE_DOMAIN}" "${ROOT_OUT}" "${FINAL_COMPARE_DIR}" "${SOURCE_COMPARE_DIR}" "${AOI_LABELS[@]}" <<'PY_STATUS'
import json, os, sys
out_json = sys.argv[1]
solve_domain = sys.argv[2]
root_out = sys.argv[3]
final_dir = sys.argv[4]
source_dir = sys.argv[5]
labels = sys.argv[6:]
obj = {
    "solve_domain_source": "union_of_requested_aois",
    "solve_domain": solve_domain,
    "final_comparison_dir": final_dir,
    "source_comparison_dir": source_dir,
    "runs": {label: {"out_dir": os.path.join(root_out, label)} for label in labels},
    "final_products": {
        "enhanced_dem": os.path.join(final_dir, "DEM_enhanced.tif"),
        "original_dem": os.path.join(final_dir, "DEM_original.tif"),
        "enhanced_hillshade": os.path.join(final_dir, "DEM_enhanced_hillshade.tif"),
        "original_hillshade": os.path.join(final_dir, "DEM_original_hillshade.tif"),
        "enhanced_shared_cpt": os.path.join(final_dir, "DEM_enhanced_shared_cpt.tif"),
        "original_shared_cpt": os.path.join(final_dir, "DEM_original_shared_cpt.tif"),
        "shared_cpt": os.path.join(final_dir, "shared_dem_color_table.cpt"),
        "shared_cpt_range": os.path.join(final_dir, "shared_dem_color_table_range.json"),
        "dem_vrt": os.path.join(final_dir, "DEM_enhanced_all_aois.vrt"),
        "authoritative_vrt": os.path.join(final_dir, "authoritative_DEM_all_aois.vrt"),
        "pairwise_compare_summary": os.path.join(root_out, "pairwise_compare_summary.json"),
        "parent_reference_raster_math_summary": os.path.join(root_out, "final_dem_parent_reference_raster_math_summary.json"),
    },
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2)
PY_STATUS

python3 - "${FINAL_COMPARE_DIR}/final_products_manifest.json" <<'PY_MANIFEST'
import json, sys
manifest = {
    "description": "Lean final comparison folder. DEM_enhanced.tif and DEM_original.tif are both union-AOI rasters in geographic coordinates.",
    "files": {
        "DEM_enhanced.tif": "canonical/enhanced parent surface cropped to union AOI in EPSG:4269",
        "DEM_original.tif": "authoritative/original DEM mosaic cropped to union AOI in EPSG:4269",
        "DEM_enhanced_hillshade.tif": "perspecto hillshade from DEM_enhanced.tif",
        "DEM_original_hillshade.tif": "perspecto hillshade from DEM_original.tif",
        "DEM_enhanced_shared_cpt.tif": "color-relief from DEM_enhanced.tif using shared_dem_color_table.cpt",
        "DEM_original_shared_cpt.tif": "color-relief from DEM_original.tif using shared_dem_color_table.cpt",
        "shared_dem_color_table.cpt": "single shared color table for enhanced/original color-relief comparison",
        "shared_dem_color_table_range.json": "range and inputs used to build shared_dem_color_table.cpt",
        "DEM_enhanced_all_aois.vrt": "VRT of AOI final enhanced DEM source rasters",
        "authoritative_DEM_all_aois.vrt": "VRT of AOI authoritative/original DEM source rasters"
    }
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
PY_MANIFEST

echo "Status written to: ${ROOT_OUT}/run_status.json"
echo "Lean final comparison products written to: ${FINAL_COMPARE_DIR}"
echo "Source rasters used for VRT/raster math retained in: ${SOURCE_COMPARE_DIR}"
echo "Final products manifest written to: ${FINAL_COMPARE_DIR}/final_products_manifest.json"
echo "Pairwise comparison summary: ${ROOT_OUT}/pairwise_compare_summary.json"
echo "Raster math summary: ${ROOT_OUT}/final_dem_parent_reference_raster_math_summary.json"
