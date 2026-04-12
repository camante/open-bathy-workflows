from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from hybrid_merge import merge_hybrid_river_bed, reconstruct_xs_mainstem_relative
from river_mask_stage import RiverMaskStageResult, run_river_mask_stage


@dataclass
class HybridRiverStageResult:
    channel_mask_tif: Path
    open_water_mask_tif: Path
    mainstem_mask_tif: Path
    xs_gpkg: Path
    xs_contract_receipt_json: Path
    soundings_subset_path: Path
    soundings_subset_receipt_json: Path
    bed_xs_tif: Path
    bathy_gpkg: Path
    xs_mainstem_meta_json: Path
    xs_mainstem_acct_json: Path
    xs_bank_contract_receipt_json: Optional[Path]
    bed_skel_tif: Path
    skeleton_receipt_json: Path
    merged_bed_tif: Path
    hybrid_merge_receipt_json: Path
    river_run_receipt_json: Path
    merge_receipt: Dict[str, Any]
    river_run_receipt: Dict[str, Any]
    xs_reconstruction_receipt_json: Optional[Path] = None
    estuary_clip_mask: Optional[Path] = None
    river_mask_stage_receipt_json: Optional[Path] = None
    river_mask_receipt: Optional[Dict[str, Any]] = None

    def as_report_outputs(self) -> Dict[str, str]:
        out = {
            "mainstem_mask": str(self.mainstem_mask_tif),
            "bed_elev_xs_mainstem": str(self.bed_xs_tif),
            "bed_elev_skeleton_full": str(self.bed_skel_tif),
            "hybrid_merge_receipt": str(self.hybrid_merge_receipt_json),
            "river_run_receipt": str(self.river_run_receipt_json),
            "xs_contract_receipt": str(self.xs_contract_receipt_json),
            "soundings_subset_receipt": str(self.soundings_subset_receipt_json),
            "skeleton_receipt": str(self.skeleton_receipt_json),
            "xs_mainstem_constraint_meta": str(self.xs_mainstem_meta_json),
            "xs_mainstem_constraint_accounting": str(self.xs_mainstem_acct_json),
        }
        if self.xs_reconstruction_receipt_json is not None:
            out["xs_mainstem_reconstruction_receipt"] = str(self.xs_reconstruction_receipt_json)
        if self.estuary_clip_mask is not None:
            out["estuary_clip_mask"] = str(self.estuary_clip_mask)
        if self.river_mask_stage_receipt_json is not None:
            out["river_mask_stage_receipt"] = str(self.river_mask_stage_receipt_json)
        if self.xs_bank_contract_receipt_json is not None:
            out["xs_bank_contract_receipt"] = str(self.xs_bank_contract_receipt_json)
        return out


def _count_valid_pixels(rp: Path, nodata_val: float) -> int:
    with rasterio.open(rp) as ds:
        nod = ds.nodata if ds.nodata is not None else nodata_val
        total = 0
        for _, w in ds.block_windows(1):
            a = ds.read(1, window=w)
            total += int(np.count_nonzero(np.isfinite(a) & (a != nod)))
        return int(total)


def _write_stage_failure_receipt(work_dir: Path, stage_name: str, exc: Exception, extra: Optional[Dict[str, Any]] = None) -> Path:
    payload: Dict[str, Any] = {
        "stage": stage_name,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if extra:
        payload.update(extra)
    return _write_json(work_dir / f"{stage_name}_failure_receipt.json", payload)


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_json_if_exists(path: Path | None) -> Optional[Dict[str, Any]]:
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_hybrid_river_stage(
    *,
    cfg: Any,
    report: Dict[str, Any],
    script_dir: Path,
    work_dir: Path,
    cache_dir: Path,
    network_gpkg: Path,
    build_domain_masks_fn: Callable[..., Tuple[Path, Path, Path]],
    estuary_clip_fn: Callable[..., Tuple[int, Optional[Path]]],
    run_command_fn: Callable[..., Tuple[int, str, str]],
    validate_xs_artifacts_fn: Callable[[Path], Any],
    validate_soundings_subset_fn: Callable[[Path], Any],
    append_river_soundings_args_fn: Callable[..., None],
    build_river_skeleton_command_fn: Callable[..., list],
    authoritative_passthrough_args_fn: Callable[..., list],
    record_authoritative_child_passthrough_fn: Callable[..., None],
    logger: Any,
) -> HybridRiverStageResult:
    logger.info("[RIVER] Using HYBRID method: XS(mainstem) + skeleton(elsewhere)")
    logger.info("[RIVER] Step 2: Building river channel/mainstem masks (hybrid)...")
    mask_stage = run_river_mask_stage(
        work_dir=work_dir,
        report=report,
        logger=logger,
        build_domain_masks_fn=build_domain_masks_fn,
        estuary_clip_fn=estuary_clip_fn,
        cfg=cfg,
    )
    channel_mask_tif = mask_stage.channel_mask_tif
    open_water_mask_tif = mask_stage.open_water_mask_tif
    mainstem_mask_tif = mask_stage.mainstem_mask_tif
    estuary_clip_mask: Optional[Path] = mask_stage.estuary_clip_mask_tif
    river_mask_stage_receipt_json: Optional[Path] = mask_stage.receipt_json
    river_mask_receipt: Optional[Dict[str, Any]] = mask_stage.receipt

    logger.info("[RIVER] Step 3a: Generating cross-sections (mainstem only, conservative defaults)...")
    xs_gpkg = work_dir / "cross_sections_mainstem.gpkg"
    cmd = [
        cfg.python_exe if hasattr(cfg, "python_exe") else __import__("sys").executable,
        "xs_builder.py",
        f"--river-gpkg={network_gpkg}",
        "--rivers-layer=major_system_network",
        f"--dem={cfg.river_dem}",
        f"--out-gpkg={xs_gpkg}",
        f"--spacing-m={cfg.xs_spacing_m}",
        f"--half-width-m={cfg.xs_length_m / 2.0}",
        f"--smoothing-window-m={cfg.xs_smoothing_window_m}",
        f"--deconflict-tol-m={cfg.xs_deconflict_tol_m}",
        f"--junction-snap-m={cfg.xs_junction_snap_m}",
        f"--junction-buffer-m={cfg.xs_junction_buffer_m}",
        f"--densify-step-m={cfg.xs_densify_step_m}",
        f"--min-stream-order={cfg.river_mainstem_min_order}",
        "--keep-top-components=1",
    ]
    if not cfg.xs_trim_overlaps:
        cmd.append("--no-trim-overlaps")
    if not cfg.xs_global_deconflict:
        cmd.append("--no-global-deconflict")
    if not cfg.xs_skip_junctions:
        cmd.append("--no-skip-junctions")

    rc, out, err = run_command_fn(cmd, cwd=script_dir, prefix="[RIVER] ")
    xs_diag_text = "\n".join(str(v) for v in (out, err) if v)
    no_xs_generated = (
        "No cross-sections were generated" in xs_diag_text
        or "centerlines kept: 0" in xs_diag_text
    )
    step_status = "success" if rc == 0 else ("skipped" if no_xs_generated else "failed")
    step_reason = "no_mainstem_cross_sections_generated" if no_xs_generated else None
    report["river"]["steps"]["xs_builder_mainstem"] = {
        "status": step_status,
        "reason": step_reason,
        "returncode": rc,
        "command": " ".join(str(c) for c in cmd),
        "stdout_tail": out,
        "stderr_tail": err,
    }
    if rc != 0 and not no_xs_generated:
        raise RuntimeError("Failed to build mainstem cross-sections for hybrid method.")

    xs_contract_receipt_json = work_dir / "xs_contract_receipt.json"
    xs_available = bool(xs_gpkg.exists()) and not no_xs_generated
    if xs_available:
        xs_contract_info = validate_xs_artifacts_fn(xs_gpkg)
        if isinstance(xs_contract_info, dict):
            lines_n = xs_contract_info.get("xs_lines") or xs_contract_info.get("xs_lines_count") or xs_contract_info.get("n_lines")
            if lines_n is not None:
                xs_available = int(lines_n) > 0
        xs_contract_receipt_json = _write_json(xs_contract_receipt_json, xs_contract_info if isinstance(xs_contract_info, dict) else {"status": "ok", "path": str(xs_gpkg)})
    if not xs_available:
        xs_contract_receipt_json = _write_json(
            xs_contract_receipt_json,
            {
                "status": "skipped",
                "reason": "no_mainstem_cross_sections_generated",
                "path": str(xs_gpkg),
                "hybrid_degraded_to": "skeleton_only",
            },
        )
        logger.warning("[RIVER] No mainstem cross-sections generated for this AOI; degrading hybrid stage to skeleton-only.")
    else:
        logger.info("[RIVER] Mainstem cross-sections generated: %s", xs_gpkg)

    logger.info("[RIVER] Step 3b: Inferring mainstem bathymetry from XS...")
    bed_xs_tif = work_dir / "river_bed_elev_xs_mainstem.tif"
    bathy_gpkg = work_dir / "river_bathy_xs_mainstem.gpkg"
    xs_mainstem_meta_json = work_dir / "xs_mainstem_constraints_meta.json"
    xs_mainstem_acct_json = work_dir / "xs_mainstem_constraints_accounting.json"

    cmd = [
        cfg.python_exe if hasattr(cfg, "python_exe") else __import__("sys").executable,
        "xs_infer_bathy_raster.py",
        f"--xs-gpkg={xs_gpkg}",
        f"--template-raster={cfg.river_dem}",
        f"--out-gpkg={bathy_gpkg}",
        f"--out-bathy-raster={bed_xs_tif}",
        f"--out-meta-json={xs_mainstem_meta_json}",
        f"--out-accounting-json={xs_mainstem_acct_json}",
        f"--continuous={cfg.river_continuous}",
        f"--continuous-k={cfg.river_continuous_k}",
        f"--idw-power={cfg.river_idw_power}",
        f"--aniso-along-scale-m={cfg.river_aniso_along_scale_m}",
        f"--aniso-cross-scale-m={cfg.river_aniso_cross_scale_m}",
        f"--thalweg-weight={cfg.river_thalweg_weight}",
        f"--nodata={cfg.river_nodata}",
        f"--overlap-reducer={cfg.river_overlap_reducer}",
        f"--xs-profile-shape={cfg.river_xs_profile_shape}",
    ]
    if bool(getattr(cfg, "river_channel_template_enabled", False)):
        cmd.append("--channel-template-enabled")
    else:
        cmd.append("--no-channel-template")
        logger.info("[RIVER] Forwarding explicit channel-template disable to XS child (hybrid stage).")
    if getattr(cfg, "river_enable_1d_energy_solver", False):
        energy_inputs = work_dir / "xs_mainstem_1d_solver_inputs.json"
        energy_outputs = work_dir / "xs_mainstem_1d_solver_outputs.json"
        energy_accounting = work_dir / "xs_mainstem_1d_solver_accounting.json"
        cmd.append("--enable-1d-energy-solver")
        if cfg.river_energy_allow_dem_proxy_wse:
            cmd.append("--energy-allow-dem-proxy-wse")
        cmd.extend([
            f"--out-1d-solver-inputs-json={energy_inputs}",
            f"--out-1d-solver-outputs-json={energy_outputs}",
            f"--out-1d-solver-accounting-json={energy_accounting}",
        ])
    if channel_mask_tif is not None and Path(channel_mask_tif).exists():
        cmd.extend([f"--channel-mask-raster={channel_mask_tif}", "--channel-mask-inside-value=1"])
    cmd.extend([
        f"--river-gpkg={network_gpkg}",
        f"--prior-mode={cfg.river_prior_mode}",
        f"--mv-a0={cfg.river_mv_a0}",
        f"--mv-bw={cfg.river_mv_bw}",
        f"--mv-ba={cfg.river_mv_ba}",
        f"--mv-bs={cfg.river_mv_bs}",
        f"--mv-eps-a={cfg.river_mv_eps_a}",
        f"--mv-eps-s={cfg.river_mv_eps_s}",
        f"--slope-proxy-window={cfg.river_slope_proxy_window}",
        f"--slope-min={cfg.river_slope_min}",
        f"--slope-max={cfg.river_slope_max}",
        f"--slope-proxy-min-n={cfg.river_slope_proxy_min_n}",
    ])
    cmd.append("--wse-profile-enabled" if cfg.river_wse_profile_enabled else "--no-wse-profile")
    cmd.extend([
        f"--wse-profile-window={cfg.river_wse_profile_window}",
        f"--wse-profile-min-n={cfg.river_wse_profile_min_n}",
    ])
    cmd.append("--wse-profile-monotonic" if cfg.river_wse_profile_monotonic else "--no-wse-profile-monotonic")
    if cfg.river_usgs_sites:
        cmd.extend([f"--usgs-sites={cfg.river_usgs_sites}"])
        if cfg.river_usgs_start:
            cmd.append(f"--usgs-start={cfg.river_usgs_start}")
        if cfg.river_usgs_end:
            cmd.append(f"--usgs-end={cfg.river_usgs_end}")
        if cfg.river_usgs_cache_dir:
            cmd.append(f"--usgs-cache-dir={cfg.river_usgs_cache_dir}")
        cmd.extend([
            f"--usgs-max-dist-m={cfg.river_usgs_max_dist_m}",
            f"--usgs-mean-to-dmax={cfg.river_usgs_mean_to_dmax}",
            f"--usgs-a-stat={cfg.river_usgs_a_stat}",
            f"--usgs-q-quantile-lo={cfg.river_usgs_q_quantile_lo}",
            f"--usgs-q-quantile-hi={cfg.river_usgs_q_quantile_hi}",
            f"--usgs-a-cv-warn={cfg.river_usgs_a_cv_warn}",
            f"--usgs-width-ratio-max={cfg.river_usgs_width_ratio_max}",
            f"--gage-snap-max-dist-m={cfg.river_gage_snap_max_dist_m}",
        ])
        cmd.append("--usgs-width-ratio-blend" if cfg.river_usgs_width_ratio_blend else "--no-usgs-width-ratio-blend")
    if cfg.river_width_stage_csv:
        cmd.extend([
            f"--width-stage-csv={cfg.river_width_stage_csv}",
            f"--width-stage-max-dist-m={cfg.river_width_stage_max_dist_m}",
            f"--width-stage-min-n={cfg.river_width_stage_min_n}",
            f"--width-stage-min-r2={cfg.river_width_stage_min_r2}",
            f"--width-stage-max-weight={cfg.river_width_stage_max_weight}",
        ])

    soundings_subset_path = work_dir / "river_soundings_subset.parquet"
    soundings_subset_meta = work_dir / "river_soundings_subset.meta.json"
    river_soundings_cfg = getattr(cfg, "river_soundings", None)
    if isinstance(river_soundings_cfg, (list, tuple, set)):
        raw_soundings_present = any(str(v).strip() for v in river_soundings_cfg)
    else:
        raw_soundings_present = bool(str(river_soundings_cfg or "").strip())
    sig_now = {
        "river_soundings": str(river_soundings_cfg or ""),
        "soundings_sample_seed": int(getattr(cfg, "soundings_sample_seed", 0) or 0),
        "soundings_max_points": int(getattr(cfg, "soundings_max_points", 0) or 0),
    }
    soundings_subset_receipt_json: Path = work_dir / "soundings_subset_receipt.json"
    subset_valid = False
    if raw_soundings_present and soundings_subset_path.exists() and soundings_subset_meta.exists():
        try:
            meta = json.loads(soundings_subset_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            failure = _write_stage_failure_receipt(work_dir, "soundings_subset", e, {"meta_path": str(soundings_subset_meta)})
            report.setdefault("river", {}).setdefault("outputs", {})["soundings_subset_failure_receipt"] = str(failure)
            raise RuntimeError(f"Cached soundings subset metadata is not valid JSON: {soundings_subset_meta}") from e
        if meta.get("signature") == sig_now:
            subset_info = validate_soundings_subset_fn(soundings_subset_path)
            if isinstance(subset_info, (str, Path)):
                subset_info = {"path": str(soundings_subset_path)}
            soundings_subset_receipt_json = _write_json(work_dir / "soundings_subset_receipt.json", subset_info)
            subset_valid = True
            logger.info("[RIVER][SOUNDINGS] Reusing validated cached subset: %s", soundings_subset_path)
    if raw_soundings_present and not subset_valid:
        cmd_subset = [
            cfg.python_exe if hasattr(cfg, "python_exe") else __import__("sys").executable,
            "xs_infer_bathy_raster.py",
            f"--xs-gpkg={xs_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-gpkg={bathy_gpkg}",
            f"--out-bathy-raster={bed_xs_tif}",
            f"--nodata={cfg.river_nodata}",
            f"--write-soundings-subset={soundings_subset_path}",
            "--only-write-soundings-subset",
        ]
        append_river_soundings_args_fn(cmd_subset, cfg, include_calib_args=True, include_mode_args=False)
        rc_s, out_s, err_s = run_command_fn(cmd_subset, cwd=script_dir, prefix="[RIVER] ")
        report["river"]["steps"]["soundings_subset"] = {
            "status": "success" if rc_s == 0 else "failed",
            "returncode": rc_s,
            "command": " ".join(str(c) for c in cmd_subset),
            "stdout_tail": out_s,
            "stderr_tail": err_s,
        }
        if rc_s != 0 or not soundings_subset_path.exists():
            exc = RuntimeError(f"Failed to create cached soundings subset (rc={rc_s}).")
            failure = _write_stage_failure_receipt(work_dir, "soundings_subset", exc, {"stderr_tail": (err_s or "")[-500:]})
            report.setdefault("river", {}).setdefault("outputs", {})["soundings_subset_failure_receipt"] = str(failure)
            raise exc
        subset_info = validate_soundings_subset_fn(soundings_subset_path)
        if isinstance(subset_info, (str, Path)):
            subset_info = {"path": str(soundings_subset_path)}
        soundings_subset_receipt_json = _write_json(work_dir / "soundings_subset_receipt.json", subset_info)
        meta = {
            "signature": sig_now,
            "subset_path": str(soundings_subset_path),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "soundings_max_points": int(getattr(cfg, "soundings_max_points", 0) or 0),
            "soundings_sample_seed": int(getattr(cfg, "soundings_sample_seed", 0) or 0),
        }
        soundings_subset_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    if raw_soundings_present and soundings_subset_path.exists():
        cfg.river_soundings = str(soundings_subset_path)

    if raw_soundings_present and soundings_subset_path.exists():
        cmd.append(f"--soundings-subset={soundings_subset_path}")
        logger.info(
            "[RIVER][SOUNDINGS] Pass 2: wiring via --soundings-subset=%s (hard-fail on missing/empty; raw --soundings not appended).",
            soundings_subset_path,
        )
        d = float(cfg.river_soundings_calib_max_dist_m or 0.0)
        if d > 0:
            cmd.append(f"--calib-max-dist-m={d}")
        st = str(cfg.river_soundings_calib_stat or "").strip()
        if st:
            cmd.append(f"--calib-stat={st}")
    else:
        skip_receipt = {
            "status": "skipped",
            "reason": "no river soundings configured",
            "river_soundings": str(river_soundings_cfg or ""),
            "subset_path": None,
        }
        soundings_subset_receipt_json = _write_json(work_dir / "soundings_subset_receipt.json", skip_receipt)
        report["river"]["steps"]["soundings_subset"] = {
            "status": "skipped",
            "reason": "no river soundings configured",
            "command": None,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }
        logger.info("[RIVER][SOUNDINGS] No raw river soundings configured; skipping subset generation and XS calibration subset wiring.")

    if xs_available:
        rc, out, err = run_command_fn(cmd, cwd=script_dir, prefix="[RIVER] ")
        report["river"]["steps"]["infer_xs_mainstem"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": " ".join(str(c) for c in cmd),
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not bed_xs_tif.exists():
            stderr_tail = (err or "").strip()[-1000:]
            if rc != 0 and bathy_gpkg.exists() and not bed_xs_tif.exists():
                raise RuntimeError(f"XS inference wrote GPKG but no raster (rc={rc}). stderr_tail={stderr_tail}")
            if rc != 0:
                raise RuntimeError(f"XS mainstem inference failed (rc={rc}). stderr_tail={stderr_tail}")
            raise RuntimeError("XS mainstem inference completed but raster output is missing.")
    else:
        report["river"]["steps"]["infer_xs_mainstem"] = {
            "status": "skipped",
            "reason": "no_mainstem_cross_sections_generated",
            "command": " ".join(str(c) for c in cmd),
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }
        xs_mainstem_meta_json.write_text(json.dumps({"status": "skipped", "reason": "no_mainstem_cross_sections_generated", "hybrid_degraded_to": "skeleton_only"}, indent=2, sort_keys=True), encoding="utf-8")
        xs_mainstem_acct_json.write_text(json.dumps({"status": "skipped", "reason": "no_mainstem_cross_sections_generated"}, indent=2, sort_keys=True), encoding="utf-8")

    if xs_available and xs_mainstem_meta_json.exists():
        meta = json.loads(xs_mainstem_meta_json.read_text(encoding="utf-8"))
        report.setdefault("river", {}).setdefault("constraints", {})["xs_mainstem"] = meta.get("constraints", {})
    if xs_mainstem_acct_json.exists():
        acct = json.loads(xs_mainstem_acct_json.read_text(encoding="utf-8"))
        report.setdefault("river", {}).setdefault("constraint_accounting", {})["xs_mainstem"] = acct
    xs_bank_contract_receipt_json = work_dir / "xs_bank_contract_receipt.json"

    logger.info("[RIVER] Step 3c: Inferring bathymetry from skeleton (full network)...")
    bed_skel_tif = work_dir / "river_bed_elev_skeleton_full.tif"
    # Channel template from XS run (Step 3b) — if it was built, pass to skeleton
    _hybrid_template_json = work_dir / "channel_template" / "channel_template.json"
    _hybrid_template_json = (
        _hybrid_template_json
        if bool(getattr(cfg, "river_channel_template_enabled", False)) and _hybrid_template_json.exists()
        else None
    )
    cmd = build_river_skeleton_command_fn(
        cfg,
        network_gpkg=network_gpkg,
        template_raster=cfg.river_dem,
        dem=cfg.river_dem,
        channel_mask=channel_mask_tif,
        out_bed=bed_skel_tif,
        authoritative_passthrough_args=authoritative_passthrough_args_fn(cfg, for_river=True),
        channel_template_json=_hybrid_template_json,
    )
    record_authoritative_child_passthrough_fn(report, "river_full_network", cmd)
    amode = cfg.river_skeleton_asymmetry_mode.strip().lower()
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
    append_river_soundings_args_fn(cmd, cfg, include_calib_args=False, include_mode_args=True)
    rc, out, err = run_command_fn(cmd, cwd=script_dir, prefix="[RIVER] ")
    report["river"]["steps"]["skeleton_full"] = {
        "status": "success" if rc == 0 else "failed",
        "returncode": rc,
        "command": " ".join(str(c) for c in cmd),
        "stdout_tail": out,
        "stderr_tail": err,
    }
    if rc != 0 or not Path(bed_skel_tif).exists():
        raise RuntimeError("Skeleton bathymetry failed (hybrid).")
    skeleton_stage_support_tif = bed_skel_tif.with_name(bed_skel_tif.stem + "_stage_support_class.tif")
    skeleton_stage_support_receipt_json = bed_skel_tif.with_name(bed_skel_tif.stem + "_stage_support_receipt.json")
    skeleton_receipt = {
        "path": str(bed_skel_tif),
        "valid_pixels": int(_count_valid_pixels(Path(bed_skel_tif), cfg.river_nodata)),
        "command": " ".join(str(c) for c in cmd),
        "stage_support_class": str(skeleton_stage_support_tif) if skeleton_stage_support_tif.exists() else None,
        "stage_support_receipt": str(skeleton_stage_support_receipt_json) if skeleton_stage_support_receipt_json.exists() else None,
    }
    skeleton_receipt_json = _write_json(work_dir / "skeleton_inference_receipt.json", skeleton_receipt)

    logger.info("[RIVER] Step 3d: Combining XS(mainstem) + skeleton(full) into cached bed raster...")
    bed_tif = cache_dir / "river_bed_elev_cached.tif"
    nod = cfg.river_nodata
    hybrid_receipt_json = work_dir / "hybrid_merge_receipt.json"
    xs_reconstruction_receipt_json = None
    if xs_available and Path(bed_xs_tif).exists():
        xs_raw_tif = work_dir / "river_bed_elev_xs_mainstem_raw.tif"
        if not xs_raw_tif.exists():
            shutil.copyfile(Path(bed_xs_tif), xs_raw_tif)
        xs_reconstruction_receipt_json = work_dir / "xs_mainstem_reconstruction_receipt.json"
        reconstruct_xs_mainstem_relative(
            Path(xs_raw_tif),
            Path(bed_skel_tif),
            Path(mainstem_mask_tif),
            Path(bed_xs_tif),
            Path(cfg.river_dem),
            nod,
            receipt_json=xs_reconstruction_receipt_json,
        )
        merge_receipt = merge_hybrid_river_bed(
            Path(bed_xs_tif),
            Path(bed_skel_tif),
            Path(mainstem_mask_tif),
            Path(channel_mask_tif) if channel_mask_tif is not None else None,
            Path(bed_tif),
            Path(cfg.river_dem),
            nod,
            receipt_json=hybrid_receipt_json,
        )
        if int(merge_receipt.get("unresolved_mainstem_pixels", 0)) > 0:
            raise RuntimeError(f"Hybrid merge left unresolved mainstem pixels: {merge_receipt.get('unresolved_mainstem_pixels')}")
    else:
        Path(bed_tif).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(bed_skel_tif), Path(bed_tif))
        merge_receipt = {
            "merge_rule": "skeleton_only_no_xs",
            "xs_raster": str(bed_xs_tif),
            "skeleton_raster": str(bed_skel_tif),
            "mainstem_mask": str(mainstem_mask_tif),
            "river_channel_mask": str(channel_mask_tif) if channel_mask_tif is not None else None,
            "out_bed": str(bed_tif),
            "template": str(cfg.river_dem),
            "nodata": float(nod),
            "reason": "no_mainstem_cross_sections_generated",
            "hybrid_degraded_to": "skeleton_only",
            "xs_valid_pixels": 0,
            "skeleton_valid_pixels": int(_count_valid_pixels(Path(bed_skel_tif), nod)),
            "xs_wins_mainstem_pixels": 0,
            "skeleton_only_mainstem_pixels": None,
            "overlap_mainstem_pixels": 0,
            "merged_valid_pixels": int(_count_valid_pixels(Path(bed_skel_tif), nod)),
            "unresolved_mainstem_pixels": 0,
            "unresolved_channel_pixels": 0,
        }
        hybrid_receipt_json.write_text(json.dumps(merge_receipt, indent=2, sort_keys=True), encoding="utf-8")
    hybrid_valid = _count_valid_pixels(Path(bed_tif), nod)
    skel_valid = _count_valid_pixels(Path(bed_skel_tif), nod)
    if skel_valid > 0 and hybrid_valid >= 0:
        min_keep = max(1, int(0.01 * skel_valid))
        if hybrid_valid < min_keep:
            raise RuntimeError(
                f"Hybrid combined bed became too sparse after XS injection ({hybrid_valid} valid vs skeleton {skel_valid}; min_keep={min_keep})."
            )
    river_run_receipt = {
        "river_mask_stage_receipt": str(river_mask_stage_receipt_json) if river_mask_stage_receipt_json is not None else None,
        "xs_contract_receipt": str(xs_contract_receipt_json),
        "soundings_subset_receipt": str(soundings_subset_receipt_json),
        "xs_bank_contract_receipt": str(xs_bank_contract_receipt_json) if xs_bank_contract_receipt_json.exists() else None,
        "xs_mainstem_constraint_meta": str(xs_mainstem_meta_json) if xs_mainstem_meta_json.exists() else None,
        "xs_mainstem_constraint_accounting": str(xs_mainstem_acct_json) if xs_mainstem_acct_json.exists() else None,
        "skeleton_receipt": str(skeleton_receipt_json),
        "hybrid_merge_receipt": str(hybrid_receipt_json),
        "xs_mainstem_reconstruction_receipt": str(xs_reconstruction_receipt_json) if xs_reconstruction_receipt_json is not None else None,
        "summary": {
            "xs_valid_pixels": int(_count_valid_pixels(Path(bed_xs_tif), nod)) if Path(bed_xs_tif).exists() else 0,
            "skeleton_valid_pixels": int(skel_valid),
            "merged_valid_pixels": int(hybrid_valid),
            "unresolved_mainstem_pixels": int(merge_receipt.get("unresolved_mainstem_pixels", 0)),
            "unresolved_channel_pixels": int(merge_receipt.get("unresolved_channel_pixels", 0) or 0),
        },
    }
    river_run_receipt_json = _write_json(work_dir / "river_run_receipt.json", river_run_receipt)
    report.setdefault("river", {}).setdefault("outputs", {}).update({
        "river_run_receipt": str(river_run_receipt_json),
        "xs_contract_receipt": str(xs_contract_receipt_json),
        "soundings_subset_receipt": str(soundings_subset_receipt_json),
        "skeleton_receipt": str(skeleton_receipt_json),
    })
    if xs_bank_contract_receipt_json.exists():
        report.setdefault("river", {}).setdefault("outputs", {})["xs_bank_contract_receipt"] = str(xs_bank_contract_receipt_json)
    return HybridRiverStageResult(
        channel_mask_tif=Path(channel_mask_tif),
        open_water_mask_tif=Path(open_water_mask_tif),
        mainstem_mask_tif=Path(mainstem_mask_tif),
        xs_gpkg=Path(xs_gpkg),
        xs_contract_receipt_json=Path(xs_contract_receipt_json),
        soundings_subset_path=Path(soundings_subset_path),
        soundings_subset_receipt_json=Path(soundings_subset_receipt_json),
        bed_xs_tif=Path(bed_xs_tif),
        bathy_gpkg=Path(bathy_gpkg),
        xs_mainstem_meta_json=Path(xs_mainstem_meta_json),
        xs_mainstem_acct_json=Path(xs_mainstem_acct_json),
        xs_bank_contract_receipt_json=Path(xs_bank_contract_receipt_json) if xs_bank_contract_receipt_json.exists() else None,
        bed_skel_tif=Path(bed_skel_tif),
        skeleton_receipt_json=Path(skeleton_receipt_json),
        merged_bed_tif=Path(bed_tif),
        hybrid_merge_receipt_json=Path(hybrid_receipt_json),
        xs_reconstruction_receipt_json=Path(xs_reconstruction_receipt_json) if xs_reconstruction_receipt_json is not None else None,
        river_run_receipt_json=Path(river_run_receipt_json),
        merge_receipt=merge_receipt,
        river_run_receipt=river_run_receipt,
        estuary_clip_mask=Path(estuary_clip_mask) if estuary_clip_mask is not None else None,
        river_mask_stage_receipt_json=Path(river_mask_stage_receipt_json) if river_mask_stage_receipt_json is not None else None,
        river_mask_receipt=river_mask_receipt,
    )
