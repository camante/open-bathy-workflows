#!/usr/bin/env bash
set -euo pipefail

# Merrimack Coast river-skeleton regression suite
#
# Cases:
#   1) baseline
#   2) bed_only
#   3) manning_only
#   4) bed_plus_manning
#
# NOTE: This script intentionally DOES NOT use any --river-hydraulic-* flags.

AOI="-71.25/-71/42.75/43"
START="2025-01-01"
END="2026-01-01"

BASE_OUT="output/merrimack_coast_regress"
BASE_CACHE="cache/merrimack_coast_regress"

# Set to 1 to delete cached outputs for each case so steps truly re-run.
CLEAN_BEFORE_RUNS="${CLEAN_BEFORE_RUNS:-1}"
# What to delete when CLEAN_BEFORE_RUNS=1
CLEAN_RIVER_CACHE="${CLEAN_RIVER_CACHE:-1}"
CLEAN_DOMAIN_MASKS="${CLEAN_DOMAIN_MASKS:-1}"
CLEAN_RIVER_OUTPUTS="${CLEAN_RIVER_OUTPUTS:-1}"
CLEAN_FUSION_OUTPUTS="${CLEAN_FUSION_OUTPUTS:-1}"

BATHY_MAIN="./bathy_main.py"
METRICS_PY="./validation/regression_metrics.py"

# Prefer ArcGIS REST only (if CLI supports it)
RIVER_HYDRO_SOURCE="arcgis"

# Channel domain
RIVER_CHANNEL_SOURCE="auto"
RIVER_CONNECTIVITY_FILTER="on"  # on|off

# Conservative Manning prior
MANNING_Q_CMS="300"
MANNING_N="0.035"

mkdir -p "${BASE_OUT}" "${BASE_CACHE}"

HELP_TEXT="$(${BATHY_MAIN} -h 2>&1 || true)"
has_flag () { echo "${HELP_TEXT}" | grep -q -- "$1"; }

add_if_supported () {
  local __arrname="$1"; shift
  local __arg="$1"
  local __flag="${__arg%%=*}"
  if has_flag "${__flag}"; then
    eval "${__arrname}+=(\"${__arg}\")"
  fi
}

# ---- Common args ----
COMMON_CANDIDATES=(
  "--aoi=${AOI}"
  "--start=${START}" "--end=${END}"
  "--methods=river"
  "--extra-xyz-cudem=hydronos,ehydro"
  "--river-soundings-mode=bed_elev"

  "--river-hydrography-source=${RIVER_HYDRO_SOURCE}"
  "--river-channel-source=${RIVER_CHANNEL_SOURCE}"
  "--river-connectivity-filter"

  "--river-channel-buffer-m=400"
  "--river-max-channel-width-m=600"
  "--river-mainstem-min-order=5"
  "--river-max-mainstem-width-m=2500"
  "--river-ocean-keep-dist-m=300"

  "--river-skeleton-wse-mode=bank_profile"
  "--river-skeleton-wse-profile-step-m=20"
  "--river-skeleton-wse-profile-resample-m=20"
  "--river-skeleton-wse-profile-smooth-sigma-m=250"
  "--river-skeleton-wse-profile-max-slope=0.004"
  "--river-skeleton-wse-profile-min-samples=10"
  "--river-skeleton-wse-profile-max-query-dist-m=200"

  "--river-skeleton-asymmetry-mode=curvature"
  "--river-skeleton-asymmetry-strength=0.25"
  "--river-skeleton-junction-mode=smooth"
  "--river-skeleton-junction-buffer-m=120"
  "--river-skeleton-junction-degree-min=3"
  "--river-skeleton-junction-smooth-sigma-m=80"
  "--river-skeleton-junction-max-width-m=300"

  "--river-swot-auto"
  "--river-swot-wse-field=wse"
  "--river-swot-max-dist-m=300"
  "--river-swot-correct-sigma-m=1000"
  "--river-swot-weight=1.0"
  "--river-swot-max-correction-m=2.0"
  "--river-swot-offset-mode=median_mad"

  "--fusion-strategy=spatial_taper"
  "--fusion-primary-weight=0.85"
  "--fusion-secondary-weight=0.15"
  "--fusion-taper-m=100"

  "--river-save-skeleton-debug"
)

COMMON_ARGS=()
for a in "${COMMON_CANDIDATES[@]}"; do
  if [[ "${a}" == "--river-connectivity-filter" ]]; then
    if [[ "${RIVER_CONNECTIVITY_FILTER}" == "on" ]]; then
      add_if_supported COMMON_ARGS "--river-connectivity-filter"
    else
      add_if_supported COMMON_ARGS "--no-river-connectivity-filter"
    fi
  else
    add_if_supported COMMON_ARGS "${a}"
  fi
done

# ---- Bed constraints ----
BED_CANDIDATES=(
  "--river-bed-profile-max-slope=0.02"
  "--river-bed-profile-max-curv=2e-4"
  "--river-bed-profile-step-m=25"
  "--river-bed-profile-strength=0.6"
  "--river-bed-profile-power=2.0"
)
BED_ARGS=()
for a in "${BED_CANDIDATES[@]}"; do add_if_supported BED_ARGS "${a}"; done

# ---- Manning priors (no --river-hydraulic-* flags) ----
MANNING_ARGS=()
if has_flag "--river-manning-enabled"; then
  MANNING_ARGS+=("--river-manning-enabled")
fi
add_if_supported MANNING_ARGS "--river-manning-mode=constant"
add_if_supported MANNING_ARGS "--river-manning-q-cms=${MANNING_Q_CMS}"
add_if_supported MANNING_ARGS "--river-manning-n=${MANNING_N}"
add_if_supported MANNING_ARGS "--river-manning-max-weight=0.2"
add_if_supported MANNING_ARGS "--river-manning-min-confidence=0.2"
add_if_supported MANNING_ARGS "--river-manning-backwater-slope-thresh=0.0002"
add_if_supported MANNING_ARGS "--river-usgs-mean-to-dmax=1.3"


clean_case () {
  local name="$1"
  local out_dir="${BASE_OUT}/${name}"
  local cache_root="${BASE_CACHE}/${name}"

  echo "[clean] out_dir=${out_dir}"
  echo "[clean] cache_root=${cache_root}"

  # Safety guards: only delete within expected roots
  if [[ "${out_dir}" != "${BASE_OUT}/"* ]]; then
    echo "[clean] Refusing to delete unexpected out_dir: ${out_dir}" >&2
    return 1
  fi
  if [[ "${cache_root}" != "${BASE_CACHE}/"* ]]; then
    echo "[clean] Refusing to delete unexpected cache_root: ${cache_root}" >&2
    return 1
  fi

  # Delete outputs (prevents reusing old rasters, makes diffs obvious)
  if [[ "${CLEAN_RIVER_OUTPUTS}" == "1" || "${CLEAN_FUSION_OUTPUTS}" == "1" ]]; then
    if [[ -d "${out_dir}" ]]; then
      /bin/rm -rf -- "${out_dir}" || {
        echo "[clean] rm -rf failed; falling back to removing contents of: ${out_dir}" >&2
        /bin/rm -rf -- "${out_dir}/"* "${out_dir}/".* 2>/dev/null || true
        rmdir --ignore-fail-on-non-empty -- "${out_dir}" 2>/dev/null || true
      }
    fi
  fi

  # Delete *river cache* so Step 2/3 cannot be satisfied from cached products
  if [[ "${CLEAN_RIVER_CACHE}" == "1" ]]; then
    if [[ -d "${cache_root}/river" ]]; then
      echo "[clean] Removing river cache: ${cache_root}/river/*"
      /bin/rm -rf -- "${cache_root}/river"/*
    fi
  fi

  # Delete cached domain masks if you want Step 2 (channel mask) to re-run from scratch
  if [[ "${CLEAN_DOMAIN_MASKS}" == "1" ]]; then
    if [[ -d "${cache_root}/masks" ]]; then
      echo "[clean] Removing cached masks: ${cache_root}/masks/*"
      /bin/rm -rf -- "${cache_root}/masks"/*
    fi
  fi
}

run_case () {
  local name="$1"; shift
  local out_dir="${BASE_OUT}/${name}"
  local cache_root="${BASE_CACHE}/${name}"

  if [[ "${CLEAN_BEFORE_RUNS}" == "1" ]]; then
    echo "[clean] Cleaning cached data for case: ${name}"
    clean_case "${name}" || exit $?
  fi

  echo
  echo "============================================================"
  echo "Running case: ${name}"
  echo "  out_dir    : ${out_dir}"
  echo "  cache_root : ${cache_root}"
  echo "============================================================"

  mkdir -p "${out_dir}" "${cache_root}"

  "${BATHY_MAIN}" \
    "--out-dir=${out_dir}" \
    "--cache-root=${cache_root}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

run_case "baseline"

if ((${#BED_ARGS[@]})); then
  run_case "bed_only" "${BED_ARGS[@]}"
else
  echo "NOTE: bed_only skipped (bed-profile flags not supported by this bathy_main.py)"
fi

if ((${#MANNING_ARGS[@]})); then
  run_case "manning_only" "${MANNING_ARGS[@]}"
else
  echo "NOTE: manning_only skipped (Manning/USGS flags not supported by this bathy_main.py)"
fi

if ((${#BED_ARGS[@]})) && ((${#MANNING_ARGS[@]})); then
  run_case "bed_plus_manning" "${BED_ARGS[@]}" "${MANNING_ARGS[@]}"
else
  echo "NOTE: bed_plus_manning skipped (requires both bed-profile + Manning flags)"
fi

echo
echo "============================================================"
echo "Computing regression metrics + summary CSV..."
echo "============================================================"

# validation/regression_metrics.py discovers runs by scanning subdirectories under --root.
# Use a *directory-only* glob so we don't accidentally pass files and confuse discovery.
# (Your earlier **/* pattern can match lots of non-run files and end up with "no run dirs".)
python "${METRICS_PY}" \
  --root "${BASE_OUT}" \
  --glob "*" \
  --write-csv

echo
echo "Done."
echo "Metrics summary: ${BASE_OUT}/metrics_summary.csv"
