"""Helpers for building river subprocess commands."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional


def build_river_skeleton_command(
    cfg: Any,
    *,
    network_gpkg: str | Path,
    template_raster: str | Path,
    dem: str | Path,
    channel_mask: str | Path,
    out_bed: str | Path,
    authoritative_passthrough_args: Optional[Iterable[str]] = None,
    debug_dir: Optional[str | Path] = None,
    channel_template_json: Optional[str | Path] = None,
) -> List[str]:
    """Build the common river_skeleton_bathy.py command used by bathy_main.

    This captures the shared numerical and WSE/junction/asymmetry options so the
    orchestration file only appends stage-specific extras.
    """
    cmd: List[str] = [
        sys.executable, "river_skeleton_bathy.py",
        f"--river-gpkg={network_gpkg}",
        f"--template-raster={template_raster}",
        f"--dem={dem}",
        f"--channel-mask={channel_mask}",
        f"--out-bed={out_bed}",
        f"--shape-exp={cfg.river_shape_exp}",
        f"--dmax-min-m={cfg.river_dmax_min_m}",
        f"--dmax-max-m={cfg.river_dmax_max_m}",
        f"--bed-profile-max-slope={float(cfg.river_bed_profile_max_slope or 0.0)}",
        f"--bed-profile-max-curv={float(cfg.river_bed_profile_max_curv or 0.0)}",
        f"--bed-profile-step-m={float(cfg.river_bed_profile_step_m or 25.0)}",
        f"--bed-profile-strength={float(cfg.river_bed_profile_strength or 0.6)}",
        f"--bed-profile-power={float(cfg.river_bed_profile_power or 2.0)}",
        f"--prior-mode={cfg.river_prior_mode}",
        f"--mv-a0={cfg.river_mv_a0}",
        f"--mv-bw={cfg.river_mv_bw}",
        f"--mv-ba={cfg.river_mv_ba}",
        f"--mv-bs={cfg.river_mv_bs}",
        f"--mv-eps-a={cfg.river_mv_eps_a}",
        f"--mv-eps-s={cfg.river_mv_eps_s}",
        f"--residual-blend-sigma-m={cfg.river_residual_blend_sigma_m}",
        f"--authoritative-bed-max-dist-m={cfg.river_authoritative_bed_max_dist_m}",
        f"--wse-mode={cfg.river_skeleton_wse_mode}",
    ]
    if authoritative_passthrough_args:
        cmd.extend(str(x) for x in authoritative_passthrough_args)

    wse_sig = float(cfg.river_skeleton_wse_smooth_sigma_m or 0.0)
    if wse_sig > 0.0:
        cmd.append(f"--wse-smooth-sigma-m={wse_sig}")
    if str(cfg.river_skeleton_wse_mode).strip().lower() == "bank_profile":
        cmd.extend([
            f"--wse-profile-step-m={float(cfg.river_skeleton_wse_profile_step_m or 20.0)}",
            f"--wse-profile-resample-m={float(cfg.river_skeleton_wse_profile_resample_m or 20.0)}",
            f"--wse-profile-smooth-sigma-m={float(cfg.river_skeleton_wse_profile_smooth_sigma_m or 0.0)}",
            f"--wse-profile-max-slope={float(cfg.river_skeleton_wse_profile_max_slope or 0.0)}",
            f"--wse-profile-min-samples={int(cfg.river_skeleton_wse_profile_min_samples or 10)}",
            f"--wse-profile-max-query-dist-m={float(cfg.river_skeleton_wse_profile_max_query_dist_m or 0.0)}",
        ])
        if getattr(cfg, "river_swot_riversp", None):
            for fp in cfg.river_swot_riversp:
                cmd.append(f"--swot-riversp={fp}")
            if getattr(cfg, "river_swot_wse_field", None):
                cmd.append(f"--swot-wse-field={cfg.river_swot_wse_field}")
            if getattr(cfg, "river_swot_qual_field", None):
                cmd.append(f"--swot-qual-field={cfg.river_swot_qual_field}")
            cmd.extend([
                f"--swot-max-dist-m={float(cfg.river_swot_max_dist_m or 0.0)}",
                f"--swot-min-samples={int(cfg.river_swot_min_samples or 0)}",
                f"--swot-correct-sigma-m={float(cfg.river_swot_correct_sigma_m or 0.0)}",
                f"--swot-weight={float(cfg.river_swot_weight or 0.0)}",
                f"--swot-max-correction-m={float(cfg.river_swot_max_correction_m or 0.0)}",
                f"--swot-wse-offset-m={float(cfg.river_swot_wse_offset_m or 0.0)}",
                f"--swot-offset-mode={str(cfg.river_swot_offset_mode or 'median_mad')}",
                f"--swot-offset-min-samples={int(cfg.river_swot_offset_min_samples or 0)}",
                f"--swot-offset-mad-z={float(cfg.river_swot_offset_mad_z or 3.5)}",
                f"--swot-offset-max-abs-m={float(cfg.river_swot_offset_max_abs_m or 0.0)}",
            ])

    jmode = str(cfg.river_skeleton_junction_mode).strip().lower()
    if jmode and jmode != "none":
        cmd.extend([
            f"--junction-mode={jmode}",
            f"--junction-buffer-m={cfg.river_skeleton_junction_buffer_m}",
            f"--junction-degree-min={cfg.river_skeleton_junction_degree_min}",
            f"--junction-max-width-m={cfg.river_skeleton_junction_max_width_m}",
        ])
        jsig = float(cfg.river_skeleton_junction_smooth_sigma_m or 0.0)
        if jsig > 0.0:
            cmd.append(f"--junction-smooth-sigma-m={jsig}")

    amode = str(cfg.river_skeleton_asymmetry_mode).strip().lower()
    if amode and amode != "none":
        cmd.extend([
            f"--asymmetry-mode={amode}",
            f"--asymmetry-strength={cfg.river_skeleton_asymmetry_strength}",
            f"--asymmetry-curv-ref={cfg.river_skeleton_asymmetry_curv_ref}",
            f"--asymmetry-max-shift={cfg.river_skeleton_asymmetry_max_shift}",
            f"--asymmetry-min-width-m={cfg.river_skeleton_asymmetry_min_width_m}",
            f"--asymmetry-min-curv={cfg.river_skeleton_asymmetry_min_curv}",
            f"--asymmetry-densify-step-m={cfg.river_skeleton_asymmetry_densify_step_m}",
        ])
    if debug_dir:
        cmd.append(f"--debug-dir={debug_dir}")
    if channel_template_json:
        _tp = Path(channel_template_json)
        if _tp.exists():
            cmd.append(f"--channel-template-json={_tp}")
    return cmd
