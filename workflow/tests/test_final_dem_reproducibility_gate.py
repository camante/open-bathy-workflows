import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import authoritative_conditioning as ac
from final_reporting import (
    write_explicit_final_outputs_manifest,
    write_final_dem_selection_receipt,
)

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import from_bounds
except ImportError:  # pragma: no cover
    rasterio = None
    from_origin = None
    from_bounds = None


@unittest.skipIf(rasterio is None, "rasterio required for real-raster reproducibility gate")
class TestFinalDemReproducibilityGate(unittest.TestCase):

    def _write_geotiff(self, path: Path, arr: np.ndarray, *, west: float, north: float, xres: float = 1.0, yres: float = 1.0, nodata: float = -9999.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": int(arr.shape[0]),
            "width": int(arr.shape[1]),
            "count": 1,
            "dtype": "float32" if np.issubdtype(arr.dtype, np.floating) else str(arr.dtype),
            "crs": "EPSG:4326",
            "transform": from_origin(west, north, xres, yres),
            "nodata": nodata if np.issubdtype(arr.dtype, np.floating) else 0,
            "compress": "deflate",
        }
        write_arr = arr.copy()
        if np.issubdtype(write_arr.dtype, np.floating):
            write_arr = write_arr.astype(np.float32)
            write_arr[~np.isfinite(write_arr)] = np.float32(nodata)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(write_arr, 1)
        return path

    def _read_float(self, path: Path) -> tuple[np.ndarray, dict]:
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
            if ds.nodata is not None:
                arr[np.isclose(arr, np.float32(ds.nodata))] = np.nan
            meta = {
                "crs": ds.crs.to_string() if ds.crs else None,
                "transform": tuple(ds.transform),
                "shape": arr.shape,
                "bounds": tuple(ds.bounds),
                "nodata": ds.nodata,
            }
        return arr, meta

    def _condition_case(self, west: float, north: float, shape=(25, 25), sl=None):
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        candidate = (0.1 * xx + 0.2 * yy).astype(np.float32)
        auth = np.full(shape, np.nan, dtype=np.float32)
        auth[8:17, 8] = 1.0
        auth[8:17, 16] = 2.0
        auth[8, 8:17] = 1.5
        auth[16, 8:17] = 2.5
        sdb_ok = np.zeros(shape, dtype=bool)
        sdb_ok[7:18, 7:18] = True
        river_ok = np.zeros(shape, dtype=bool)
        river_ok[10:15, 10:15] = True
        river_support = np.zeros(shape, dtype=np.uint8)
        river_support[11:14, 11] = 1
        river_support_depth = np.full(shape, np.nan, dtype=np.float32)
        river_support_depth[11:14, 11] = 1.75
        if sl is not None:
            candidate = candidate[sl]
            auth = auth[sl]
            sdb_ok = sdb_ok[sl]
            river_ok = river_ok[sl]
            river_support = river_support[sl]
            river_support_depth = river_support_depth[sl]
        result = ac.support_weighted_condition_arrays(
            candidate=candidate,
            auth=auth,
            sdb_ok=sdb_ok,
            river_ok=river_ok,
            sdb_gw=np.where(sdb_ok, 0.75, 0.0).astype(np.float32),
            sdb_ti=sdb_ok.astype(np.uint8),
            river_gw=np.where(river_ok, 0.65, 0.0).astype(np.float32),
            river_ti=river_ok.astype(np.uint8),
            river_support=river_support,
            river_support_depth=river_support_depth,
            pixel_size_m=10.0,
            support_decay_m=300.0,
            support_density_radius_m=60.0,
            coastal_sdb_support_transition_m=200.0,
            river_anchor_density_radius_m=100.0,
            river_scaffold_transition_m=500.0,
        )
        if sl is None:
            x0 = west
            y0 = north
        else:
            x0 = west + sl[1].start
            y0 = north - sl[0].start
        return result, x0, y0

    def test_same_aoi_rerun_identity_and_nested_overlap_on_real_rasters(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            west = -71.0
            north = 43.0
            full_a, x0_a, y0_a = self._condition_case(west, north)
            full_b, x0_b, y0_b = self._condition_case(west, north)

            run1 = td / "run1"
            run2 = td / "run2"
            p1 = self._write_geotiff(run1 / "final.tif", full_a["conditioned"], west=x0_a, north=y0_a)
            p2 = self._write_geotiff(run2 / "final.tif", full_b["conditioned"], west=x0_b, north=y0_b)
            prov1 = self._write_geotiff(run1 / "prov.tif", full_a["provenance"].astype(np.float32), west=x0_a, north=y0_a)
            prov2 = self._write_geotiff(run2 / "prov.tif", full_b["provenance"].astype(np.float32), west=x0_b, north=y0_b)
            support1 = self._write_geotiff(run1 / "support.tif", full_a["support"].astype(np.float32), west=x0_a, north=y0_a)
            support2 = self._write_geotiff(run2 / "support.tif", full_b["support"].astype(np.float32), west=x0_b, north=y0_b)

            arr1, meta1 = self._read_float(p1)
            arr2, meta2 = self._read_float(p2)
            np.testing.assert_allclose(arr1, arr2, atol=1e-6)
            self.assertEqual(meta1, meta2)
            self.assertEqual(p1.read_bytes(), p2.read_bytes())
            self.assertEqual(prov1.read_bytes(), prov2.read_bytes())
            self.assertEqual(support1.read_bytes(), support2.read_bytes())

            sl = np.s_[5:20, 5:20]
            nested, x0_n, y0_n = self._condition_case(west, north, sl=sl)
            nested_path = self._write_geotiff(td / "nested" / "final.tif", nested["conditioned"], west=x0_n, north=y0_n)
            nested_arr, nested_meta = self._read_float(nested_path)

            with rasterio.open(p1) as ds_full:
                win = from_bounds(*nested_meta["bounds"], transform=ds_full.transform)
                win = win.round_offsets().round_lengths()
                overlap = ds_full.read(1, window=win).astype(np.float32)
                if ds_full.nodata is not None:
                    overlap[np.isclose(overlap, np.float32(ds_full.nodata))] = np.nan
            np.testing.assert_allclose(overlap, nested_arr, atol=1e-6)

    def test_final_dem_contract_on_written_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            west = -71.0
            north = 43.0
            result, x0, y0 = self._condition_case(west, north)
            auth_path = self._write_geotiff(td / "authoritative_base.tif", np.where(result["locked"], result["conditioned"], np.nan), west=x0, north=y0)
            final_native = self._write_geotiff(td / "combined" / "conditioned_final.tif", result["conditioned"], west=x0, north=y0)
            final_user = self._write_geotiff(td / "deliver" / "conditioned_final_user.tif", result["conditioned"], west=x0, north=y0)
            final_prov = self._write_geotiff(td / "combined" / "conditioned_provenance.tif", result["provenance"].astype(np.float32), west=x0, north=y0)
            support_path = self._write_geotiff(td / "combined" / "support_class.tif", result["support"].astype(np.float32), west=x0, north=y0)
            diff_path = self._write_geotiff(td / "comparison" / "conditioned_minus_baseline.tif", np.zeros_like(result["conditioned"], dtype=np.float32), west=x0, north=y0)
            baseline_path = self._write_geotiff(td / "comparison" / "baseline_cudem_aligned.tif", result["conditioned"], west=x0, north=y0)

            cfg = SimpleNamespace(out_dir=td, authoritative_base=auth_path, aoi=f"{west}/{west+25}/{north-25}/{north}", tile_bbox=None)
            report = {
                "authoritative_base": {
                    "outputs": {
                        "aligned_authoritative_base": str(auth_path),
                        "conditioned_depth": str(final_native),
                        "support_class": str(support_path),
                    },
                    "policy": {
                        "support_class_codes": {"1": "authoritative_locked", "2": "support_weighted_interpolation", "3": "sdb", "4": "river", "5": "scaffold"},
                        "provenance_class_codes": {"10": "authoritative_locked", "20": "support_weighted_interpolation", "30": "sdb", "40": "river", "50": "scaffold"},
                    },
                },
                "fusion": {"outputs": {"depth": str(final_native)}},
                "gapfill": {"outputs": {}},
                "outputs": {},
            }
            write_explicit_final_outputs_manifest(cfg, report, final_native=final_native, final_for_user=final_user, final_provenance=final_prov)
            write_final_dem_selection_receipt(cfg, report, final_native=final_native, final_for_user=final_user, final_provenance=final_prov)

            final_outputs = json.loads((td / "final_outputs.json").read_text())
            selection = json.loads((td / "final_dem_selection_receipt.json").read_text())
            self.assertEqual(selection["selected_final_depth"], final_outputs["selected_final_depth"])
            self.assertEqual(selection["selected_final_provenance"], final_outputs["selected_final_provenance"])
            self.assertTrue(Path(selection["selected_final_depth"]).exists())
            self.assertTrue(Path(selection["selected_final_provenance"]).exists())

            auth_arr, _ = self._read_float(auth_path)
            final_arr, _ = self._read_float(final_native)
            self.assertTrue(np.all(np.isfinite(final_arr)))
            locked = np.isfinite(auth_arr)
            np.testing.assert_allclose(final_arr[locked], auth_arr[locked], atol=1e-6)

            packaged = {
                "final_depth_native": str(final_native),
                "baseline_cudem_interpolation_aligned_to_final": str(baseline_path),
                "conditioned_minus_baseline_cudem": str(diff_path),
                "support_class": str(support_path),
                "final_provenance_native": str(final_prov),
            }
            from final_reporting import write_comparison_summary
            summary_path = write_comparison_summary(cfg, report, packaged, td / "comparison")
            self.assertTrue(Path(summary_path).exists())


if __name__ == "__main__":
    unittest.main()
