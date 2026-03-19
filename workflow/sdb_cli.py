"""Helpers for building SDB subprocess commands."""
from __future__ import annotations

import sys
from typing import Any, Iterable, List, Optional


def _normalize_multi_path_value(value: Any) -> List[str]:
    parts: List[str] = []
    if value is None:
        return parts

    def _append_one(item: Any) -> None:
        if item is None:
            return
        s = str(item).strip()
        if not s:
            return
        for piece in s.split(","):
            piece = piece.strip()
            if not piece:
                continue
            piece = piece.strip().strip("[]").strip("\"'").strip()
            if piece:
                parts.append(piece)

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_one(item)
    else:
        _append_one(value)
    return parts


def build_sdb_command(cfg: Any, *, out_dir: str, authoritative_passthrough_args: Optional[Iterable[str]] = None) -> List[str]:
    """Build the common sdb_main.py command used by bathy_main."""
    cmd: List[str] = [
        sys.executable, "sdb_main.py",
        f"--aoi={cfg.aoi}",
        f"--start={cfg.start_date}",
        f"--end={cfg.end_date}",
        f"--out-dir={out_dir}",
        f"--cloud={cfg.cloud}",
        f"--icesat={cfg.icesat}",
        f"--sdb-mode={cfg.sdb_mode}",
        f"--cache-root={cfg.cache_root}",
        f"--align-mode={cfg.align_mode}",
        f"--working-srs={cfg.working_srs}",
        f"--working-vcrs-epsg={cfg.working_vcrs_epsg}",
    ]
    if authoritative_passthrough_args:
        cmd.extend(str(x) for x in authoritative_passthrough_args)
    return cmd



def augment_sdb_command(cfg: Any, cmd: List[str], *, sdb_main_path: Optional[str] = None, logger: Any = None) -> List[str]:
    log = logger
    try:
        if bool(getattr(cfg, 'sdb_model_bank_enabled', False)):
            cmd.append(f"--model-bank={cfg.sdb_model_bank}")
            cmd.append(f"--bank-max-samples={int(cfg.sdb_bank_max_samples)}")
            cmd.append(f"--bank-seed={int(cfg.sdb_bank_seed)}")
            cmd.append(f"--bank-retrain-min-new={int(cfg.sdb_bank_retrain_min_new)}")
        else:
            cmd.append('--no-model-bank')
    except (AttributeError, TypeError, ValueError) as exc:
        if log:
            log.debug('model bank policy args build failed: %s', exc, exc_info=True)
    try:
        if bool(getattr(cfg, 'sdb_model_cache_enabled', False)):
            cmd.append(f"--model-cache-key={cfg.sdb_model_cache_key}")
        else:
            cmd.append('--no-model-cache')
    except (AttributeError, TypeError, ValueError) as exc:
        if log:
            log.debug('model cache policy args build failed: %s', exc, exc_info=True)
    if bool(getattr(cfg, 'glint_correct', False)):
        supports_glint = False
        try:
            from pathlib import Path
            p = Path(sdb_main_path) if sdb_main_path else None
            if p and p.exists():
                txt = p.read_text(encoding='utf-8', errors='ignore')
                supports_glint = ('--glint-correct' in txt) or ('glint_correct' in txt)
        except OSError:
            supports_glint = False
        if not supports_glint:
            if log:
                log.warning('[SDB][GLINT] --glint-correct requested, but sdb_main.py does not appear to support glint flags; skipping glint passthrough.')
        else:
            cmd.extend([
                '--glint-correct',
                f"--glint-nir-band={cfg.glint_nir_band}",
                f"--glint-vis-bands={cfg.glint_vis_bands}",
                f"--glint-nir-min-percentile={cfg.glint_nir_min_percentile}",
                f"--glint-deepwater-b02-max={cfg.glint_deepwater_b02_max}",
                f"--glint-min-samples={cfg.glint_min_samples}",
                f"--glint-max-samples={cfg.glint_max_samples}",
                f"--glint-clip-min={cfg.glint_clip_min}",
            ])
    xyz_list: List[str] = []
    river_soundings = getattr(cfg, 'river_soundings', None)
    if river_soundings:
        xyz_list.extend(_normalize_multi_path_value(river_soundings))
    auth_xyz = getattr(cfg, 'sdb_authoritative_extra_xyz', None)
    if auth_xyz:
        xyz_list.extend(_normalize_multi_path_value(auth_xyz))
    if xyz_list:
        # de-dup while preserving order
        seen = set()
        xyz_list = [pp for pp in xyz_list if not (pp in seen or seen.add(pp))]
        cmd += ['--extra-xyz'] + xyz_list
        cmd.append(f"--extra-xyz-crs={cfg.extra_xyz_crs}")
    enable_sampling = bool(getattr(cfg, 'enable_adaptive_sampling', False))
    if not enable_sampling:
        cmd.append('--disable-adaptive-sampling')
    cmd.append(f"--sampling-target-points={cfg.sampling_target_points}")
    cmd.append(f"--sampling-min-threshold={cfg.sampling_min_threshold}")
    cmd.append(f"--sampling-max-gap-m={cfg.sampling_max_gap_m}")
    return cmd
