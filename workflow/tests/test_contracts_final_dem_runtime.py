import sys
import tempfile
from pathlib import Path

sys.path = [p for p in sys.path if '/workflow/tests' not in p and not p.endswith('/workflow')]

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.append(str(Path(__file__).resolve().parents[1]))

from contracts_final_dem_runtime import run_final_dem_runtime_contracts
from provenance_schema import ProvenanceClass
from support_classes import SupportClass


def _write_raster(path: Path, arr: np.ndarray, dtype: str, nodata=None):
    transform = from_origin(0.0, float(arr.shape[0]), 1.0, 1.0)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype=dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def _report(final_path: Path, auth_path: Path, support_path: Path, prov_path: Path, guidance_path: Path):
    return {
        'authoritative_base': {
            'outputs': {
                'conditioned_depth': str(final_path),
                'aligned_authoritative_base': str(auth_path),
                'support_class': str(support_path),
                'conditioned_provenance': str(prov_path),
                'guidance_influence': str(guidance_path),
            }
        }
    }


def test_final_dem_support_provenance_alignment_passes_for_exact_mapping():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        final = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        auth = final.copy()
        support = np.array([
            [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB)],
            [int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.SCAFFOLD_INFERRED)],
        ], dtype=np.uint8)
        prov = np.array([
            [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.SDB_CONDITIONED_FILL)],
            [int(ProvenanceClass.RIVER_CONDITIONED_FILL), int(ProvenanceClass.RIVER_SCAFFOLD_DOMINANT_FILL)],
        ], dtype=np.uint8)
        guidance = np.array([[0.0, 0.5], [0.25, 0.75]], dtype=np.float32)

        final_p = td / 'final.tif'; _write_raster(final_p, final, 'float32', nodata=-9999.0)
        auth_p = td / 'auth.tif'; _write_raster(auth_p, auth, 'float32', nodata=-9999.0)
        support_p = td / 'support.tif'; _write_raster(support_p, support, 'uint8', nodata=0)
        prov_p = td / 'prov.tif'; _write_raster(prov_p, prov, 'uint8', nodata=0)
        guidance_p = td / 'guidance.tif'; _write_raster(guidance_p, guidance, 'float32', nodata=-9999.0)

        suite = run_final_dem_runtime_contracts(
            report=_report(final_p, auth_p, support_p, prov_p, guidance_p),
            debug_dir=td / 'debug',
        )
        results = {r.name: r for r in suite.results}
        assert results['support_provenance_alignment'].passed is True
        assert results['guidance_influence_support_eligibility'].passed is True
        assert results['guidance_influence_provenance_eligibility'].passed is True


def test_final_dem_guidance_influence_outside_allowed_support_fails_with_debug_artifact():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        final = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        auth = final.copy()
        support = np.array([
            [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.ANCHORED_INTERPOLATION)],
            [int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.SCAFFOLD_INFERRED)],
        ], dtype=np.uint8)
        prov = np.array([
            [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.ANCHORED_INTERPOLATION)],
            [int(ProvenanceClass.RIVER_CONDITIONED_FILL), int(ProvenanceClass.RIVER_SCAFFOLD_DOMINANT_FILL)],
        ], dtype=np.uint8)
        guidance = np.array([[0.1, 0.2], [0.0, 0.0]], dtype=np.float32)

        final_p = td / 'final.tif'; _write_raster(final_p, final, 'float32', nodata=-9999.0)
        auth_p = td / 'auth.tif'; _write_raster(auth_p, auth, 'float32', nodata=-9999.0)
        support_p = td / 'support.tif'; _write_raster(support_p, support, 'uint8', nodata=0)
        prov_p = td / 'prov.tif'; _write_raster(prov_p, prov, 'uint8', nodata=0)
        guidance_p = td / 'guidance.tif'; _write_raster(guidance_p, guidance, 'float32', nodata=-9999.0)

        suite = run_final_dem_runtime_contracts(
            report=_report(final_p, auth_p, support_p, prov_p, guidance_p),
            debug_dir=td / 'debug',
        )
        results = {r.name: r for r in suite.results}
        assert results['guidance_influence_support_eligibility'].passed is False
        dbg = results['guidance_influence_support_eligibility'].artifact_paths.get('violation_mask')
        assert dbg is not None and Path(dbg).exists()
