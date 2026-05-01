from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from repo_runtime_modes import (
    ACTIVE_RIVER_WORKFLOW,
    NORMAL_RUN_OPTIONAL_LOG_GLOB,
    NORMAL_RUN_PRIMARY_FILES,
    PRIMARY_DOCS,
    LEGACY_CODE_DIRS,
    VALIDATION_CODE_DIRS,
    REPORTING_CODE_DIRS,
    DEBUG_CODE_DIRS,
    CORE_CODE_DIRS,
    PIPELINE_CODE_DIRS,
)

def validate_runtime_contract(root: Path) -> list[str]:
    errors: list[str] = []
    text = (root / 'bathy_main.py').read_text(encoding='utf-8', errors='ignore')
    if 'add_argument("--river-method"' in text or "add_argument('--river-method'" in text:
        errors.append('bathy_main.py should not expose a --river-method CLI arg in the active workflow')
    if 'run_river_workflow_direct' not in text:
        errors.append('bathy_main.py should route river execution through run_river_workflow_direct')
    return errors


def validate_primary_docs(root: Path) -> list[str]:
    errors: list[str] = []
    for rel in PRIMARY_DOCS:
        path = root / rel
        if not path.exists():
            errors.append(f'missing primary doc: {rel}')

    readme = (root / 'README.md').read_text(encoding='utf-8', errors='ignore')
    if '--solve-domain' not in readme:
        errors.append('README.md should document --solve-domain for the built-in river workflow')
    for token in NORMAL_RUN_PRIMARY_FILES:
        if token not in readme:
            errors.append(f'README.md missing normal-run file token {token!r}')

    active_doc = (root / 'docs' / 'ACTIVE_WORKFLOW.md').read_text(encoding='utf-8', errors='ignore')
    if ACTIVE_RIVER_WORKFLOW not in active_doc:
        errors.append('docs/ACTIVE_WORKFLOW.md must name the active river workflow')
    stage_order = (
        'entry bundle -> solve_domain -> grids -> authoritative -> centerline -> '
        'wse_proxy -> authoritative_bed -> observed_offset -> modeled_offset -> '
        'backbone -> corridor -> surface -> lock -> export -> final_dem'
    )
    if stage_order not in active_doc:
        errors.append('docs/ACTIVE_WORKFLOW.md missing active stage order string')
    if 'pipeline/river_shared_solve/river_pipeline.py' not in active_doc and 'pipeline/river_shared_solve/' not in active_doc:
        errors.append('docs/ACTIVE_WORKFLOW.md should point to the pipeline/river_shared_solve package')

    run_outputs = (root / 'docs' / 'RUN_OUTPUTS.md').read_text(encoding='utf-8', errors='ignore')
    for token in (*NORMAL_RUN_PRIMARY_FILES, NORMAL_RUN_OPTIONAL_LOG_GLOB):
        if token not in run_outputs:
            errors.append(f'docs/RUN_OUTPUTS.md missing normal-run output token {token!r}')
    for token in ('io_manifest.md', 'unified_bathy_report.json'):
        if token not in run_outputs:
            errors.append(f'docs/RUN_OUTPUTS.md should explicitly call out deprecated default output {token!r}')

    repo_map = (root / 'docs' / 'REPO_MAP.md').read_text(encoding='utf-8', errors='ignore')
    if 'pipeline/river_shared_solve/river_pipeline.py' not in repo_map:
        errors.append('docs/REPO_MAP.md should point to the pipeline/river_shared_solve package')
    if 'pipeline/final_route/final_route_outputs_stage.py' not in repo_map:
        errors.append('docs/REPO_MAP.md should point to the pipeline/final_route package')
    if 'pipeline/final_dem/final_dem_contract.py' not in repo_map:
        errors.append('docs/REPO_MAP.md should point to the pipeline/final_dem package')

    strict_docs = [root / 'README.md', root / 'docs' / 'ACTIVE_WORKFLOW.md', root / 'docs' / 'REPO_MAP.md']
    forbidden = {'unified_bathy_report.json', 'io_manifest.md', 'RIVER_WORKFLOW_DEBUG.txt'}
    for path in strict_docs:
        text = path.read_text(encoding='utf-8', errors='ignore')
        for token in forbidden:
            if token in text:
                errors.append(f'{path.relative_to(root)} should not present legacy/debug output {token!r} in the active docs spine')
    return errors


def validate_repo_hygiene(root: Path) -> list[str]:
    errors: list[str] = []
    old_root_docs = [
        'README_GENERAL_WORKFLOW.md',
        'README_WORKFLOW_PLAIN.md',
        'README_WORKFLOW_TECHNICAL.md',
        'README_WORKFLOW_DETAILED.md',
        'README_SCRIPTS_DETAILED.md',
    ]
    for name in old_root_docs:
        if (root / name).exists():
            errors.append(f'legacy doc should not live at repo root anymore: {name}')
    if list(root.glob('PATCH_NOTES_*.txt')):
        errors.append('patch notes should live under archive/patch_notes/, not repo root')

    for rel in (*LEGACY_CODE_DIRS, *VALIDATION_CODE_DIRS, *REPORTING_CODE_DIRS, *DEBUG_CODE_DIRS, *CORE_CODE_DIRS, *PIPELINE_CODE_DIRS):
        if not (root / rel).exists():
            errors.append(f'missing expected code namespace: {rel}')

    expected_moved = {
        'river_v1_pipeline.py': 'legacy/river/river_v1_pipeline.py',
        'river_v2_pipeline.py': 'legacy/river/river_v2_pipeline.py',
        'simple_river_bundle_b.py': 'legacy/river/simple_river_bundle_b.py',
        'benchmark_workflow_stage.py': 'validation/benchmark_workflow_stage.py',
        'postrun_regression_stage.py': 'validation/postrun_regression_stage.py',
        'scientific_validation_stage.py': 'validation/scientific_validation_stage.py',
        'final_reporting.py': 'reporting/final_reporting.py',
        'final_run_reporting.py': 'reporting/final_run_reporting.py',
        'provenance_reporting.py': 'reporting/provenance_reporting.py',
        'run_summary.py': 'reporting/run_summary.py',
        'debug_bundle.py': 'legacy/debug/debug_bundle.py',
        'river_workflow_debug_report.py': 'legacy/debug/river_workflow_debug_report.py',
        'cache_utils.py': 'core/cache_utils.py',
        'checkpoints.py': 'core/checkpoints.py',
        'chunked_processing.py': 'core/chunked_processing.py',
        'compat_pandas.py': 'core/compat_pandas.py',
        'constants.py': 'core/constants.py',
        'deps.py': 'core/deps.py',
        'errors.py': 'core/errors.py',
        'errors_scientific.py': 'core/errors_scientific.py',
        'flight_recorder.py': 'core/flight_recorder.py',
        'io_artifacts.py': 'core/io_artifacts.py',
        'logging_config.py': 'core/logging_config.py',
        'memory_diag.py': 'core/memory_diag.py',
        'nodata_utils.py': 'core/nodata_utils.py',
        'process_utils.py': 'core/process_utils.py',
        'workflow_execution_state.py': 'core/workflow_execution_state.py',
        'river_workflow_context.py': 'pipeline/river_workflow/river_workflow_context.py',
        'river_workflow_contract.py': 'pipeline/river_workflow/river_workflow_contract.py',
        'river_workflow_paths.py': 'pipeline/river_workflow/river_workflow_paths.py',
        'river_workflow_pipeline.py': 'pipeline/river_workflow/river_workflow_pipeline.py',
        'river_context.py': 'pipeline/river_shared_solve/river_context.py',
        'river_contract.py': 'pipeline/river_shared_solve/river_contract.py',
        'river_paths.py': 'pipeline/river_shared_solve/river_paths.py',
        'river_pipeline.py': 'pipeline/river_shared_solve/river_pipeline.py',
        'river_canonical_domain.py': 'pipeline/river_shared_solve/river_canonical_domain.py',
        'river_workflow_stage_authoritative.py': 'pipeline/river_workflow/river_workflow_stage_authoritative.py',
        'river_workflow_stage_export.py': 'pipeline/river_workflow/river_workflow_stage_export.py',
        'river_workflow_stage_final_dem.py': 'pipeline/river_workflow/river_workflow_stage_final_dem.py',
        'river_workflow_validation.py': 'pipeline/river_workflow/river_workflow_validation.py',
        'final_route_inputs_stage.py': 'pipeline/final_route/final_route_inputs_stage.py',
        'final_route_outputs_stage.py': 'pipeline/final_route/final_route_outputs_stage.py',
        'final_route_receipts.py': 'pipeline/final_route/final_route_receipts.py',
        'final_route_contract.py': 'pipeline/final_route/final_route_contract.py',
        'final_dem_contract.py': 'pipeline/final_dem/final_dem_contract.py',
        'final_dem_policy.py': 'pipeline/final_dem/final_dem_policy.py',
        'final_dem_contract_validator.py': 'pipeline/final_dem/final_dem_contract_validator.py',
    }
    for old_name, new_rel in expected_moved.items():
        if (root / old_name).exists():
            errors.append(f'legacy/debug module should not live at repo root anymore: {old_name}')
        if not (root / new_rel).exists():
            errors.append(f'moved module missing from new namespace: {new_rel}')
    return errors


def run_all_checks(root: Path | None = None) -> list[str]:
    root = Path('.') if root is None else Path(root)
    return [
        *validate_runtime_contract(root),
        *validate_primary_docs(root),
        *validate_repo_hygiene(root),
    ]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Check repo runtime language and output contracts.')
    parser.add_argument('--root', default='.', help='Repo root to inspect')
    args = parser.parse_args(list(argv) if argv is not None else None)
    errors = run_all_checks(Path(args.root))
    if errors:
        print('[repo_contract_checks][ERROR] contract check failed:')
        for err in errors:
            print(f'  - {err}')
        return 1
    print('[repo_contract_checks] OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
