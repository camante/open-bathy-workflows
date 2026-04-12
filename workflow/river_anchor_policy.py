from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _bool_series(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool, name=name)
    return frame[name].fillna(default).astype(bool)


def _str_series(frame: pd.DataFrame, name: str, default: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype='object', name=name)
    vals = frame[name].fillna(default).astype(str).str.strip()
    vals = vals.where(vals.ne(''), default)
    return pd.Series(vals, index=frame.index, dtype='object', name=name)


def _float_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype='float32', name=name)
    return pd.to_numeric(frame[name], errors='coerce').astype('float32')


def build_anchor_policy_table(target_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = target_df.copy()
    out = pd.DataFrame(index=work.index)
    out['component_id'] = _str_series(work, 'component_id', 'main')
    out['station_m'] = _float_series(work, 'station_m')

    exact_hard_anchor = _bool_series(work, 'authoritative_anchor_present', False) | _bool_series(work, 'true_measured_xs_qualified', False)
    curve_hard_anchor = (~exact_hard_anchor) & _bool_series(work, 'authoritative_anchor_curve_present', False)
    bank_only_support = (~exact_hard_anchor) & (~curve_hard_anchor) & _str_series(work, 'station_authoritative_bed_support_class', 'none').eq('bank_margin_only')

    anchor_class = np.where(
        exact_hard_anchor.to_numpy(),
        'exact_hard_anchor',
        np.where(curve_hard_anchor.to_numpy(), 'curve_hard_anchor', np.where(bank_only_support.to_numpy(), 'bank_only_support', 'non_anchor')),
    )
    anchor_source = _str_series(work, 'authoritative_anchor_source', 'none')
    anchor_source = pd.Series(
        np.where(
            exact_hard_anchor.to_numpy() & anchor_source.eq('none').to_numpy(),
            np.where(_bool_series(work, 'authoritative_anchor_present', False).to_numpy(), 'authoritative_anchor', 'true_measured_xs'),
            np.where(curve_hard_anchor.to_numpy() & anchor_source.eq('none').to_numpy(), 'authoritative_anchor_curve', np.where(bank_only_support.to_numpy() & anchor_source.eq('none').to_numpy(), 'bank_only_support', anchor_source.to_numpy())),
        ),
        index=work.index,
        dtype='object',
    )
    anchor_z = _float_series(work, 'target_thalweg_z_m')
    anchor_z = anchor_z.where(np.isfinite(anchor_z), _float_series(work, 'active_core_support_z_m'))
    anchor_exact = exact_hard_anchor.astype(bool)
    anchor_locks_core = (exact_hard_anchor | curve_hard_anchor).astype(bool)
    anchor_blocks_rebuild = (exact_hard_anchor | curve_hard_anchor).astype(bool)
    anchor_confidence = pd.Series(
        np.where(exact_hard_anchor.to_numpy(), 1.0, np.where(curve_hard_anchor.to_numpy(), 0.85, np.where(bank_only_support.to_numpy(), 0.35, 0.0))),
        index=work.index,
        dtype='float32',
    )

    out['anchor_class'] = pd.Series(anchor_class, index=work.index, dtype='object')
    out['exact_hard_anchor'] = exact_hard_anchor.astype(bool)
    out['curve_hard_anchor'] = curve_hard_anchor.astype(bool)
    out['bank_only_support'] = bank_only_support.astype(bool)
    out['non_anchor'] = (~(exact_hard_anchor | curve_hard_anchor | bank_only_support)).astype(bool)
    out['anchor_source'] = anchor_source
    out['anchor_z_m'] = anchor_z.astype('float32')
    out['anchor_confidence'] = anchor_confidence.astype('float32')
    out['anchor_exact'] = anchor_exact.astype(bool)
    out['anchor_locks_core'] = anchor_locks_core.astype(bool)
    out['anchor_blocks_rebuild'] = anchor_blocks_rebuild.astype(bool)

    out = out.sort_values(['component_id', 'station_m']).reset_index(drop=True)
    summary = {
        'schema_version': 1,
        'station_count': int(len(out)),
        'component_count': int(out['component_id'].nunique(dropna=True)),
        'anchor_class_counts': {str(k): int(v) for k, v in out['anchor_class'].astype(str).value_counts(dropna=False).items()},
        'exact_hard_anchor_count': int(out['exact_hard_anchor'].sum()),
        'curve_hard_anchor_count': int(out['curve_hard_anchor'].sum()),
        'bank_only_support_count': int(out['bank_only_support'].sum()),
        'anchor_locks_core_count': int(out['anchor_locks_core'].sum()),
        'anchor_blocks_rebuild_count': int(out['anchor_blocks_rebuild'].sum()),
    }
    return out, summary



def write_anchor_policy_artifacts(*, river_dir: str | Path, anchor_df: pd.DataFrame, summary: Dict[str, Any]) -> Dict[str, str]:
    river_dir = Path(river_dir)
    table_path = river_dir / 'river_anchor_table.csv'
    summary_path = river_dir / 'river_anchor_summary.json'
    anchor_df.to_csv(table_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return {
        'anchor_table': str(table_path),
        'anchor_summary': str(summary_path),
    }
