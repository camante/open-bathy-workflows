from __future__ import annotations

import os
from pathlib import Path

if os.environ.get("WORKFLOW_DISABLE_RASTER_CONTRACTS", "0") != "1":
    try:
        import rasterio
        from raster_contract import validate_raster_against_profile

        _orig_rasterio_open = rasterio.open

        class _CheckedWriter:
            def __init__(self, ds, path, profile):
                self._ds = ds
                self._path = Path(path)
                self._profile = dict(profile)
                self._validated = False

            def __getattr__(self, name):
                return getattr(self._ds, name)

            def __enter__(self):
                self._ds.__enter__()
                return self

            def _validate(self):
                if self._validated or not self._path.exists():
                    return
                self._validated = True
                validate_raster_against_profile(
                    self._path,
                    self._profile,
                    operation=f"rasterio.write:{self._path.name}",
                )

            def __exit__(self, exc_type, exc, tb):
                result = self._ds.__exit__(exc_type, exc, tb)
                if exc_type is None:
                    self._validate()
                return result

            def close(self):
                self._ds.close()
                self._validate()

        def _patched_open(fp, mode="r", *args, **kwargs):
            ds = _orig_rasterio_open(fp, mode, *args, **kwargs)
            if not isinstance(mode, str):
                return ds
            write_like = "w" in mode
            update_with_profile = "+" in mode and any(
                key in kwargs for key in ("driver", "width", "height", "count", "dtype", "crs", "transform", "nodata")
            )
            if write_like or update_with_profile:
                prof = dict(kwargs)
                return _CheckedWriter(ds, fp, prof)
            return ds

        rasterio.open = _patched_open
    except Exception:
        pass

    try:
        from osgeo import gdal
        from raster_contract import validate_gdal_output

        _orig_get_driver = gdal.GetDriverByName
        _orig_data_type_name = gdal.GetDataTypeName

        class _CheckedBandProxy:
            def __init__(self, band, dataset_proxy):
                self._band = band
                self._dataset_proxy = dataset_proxy

            def __getattr__(self, name):
                return getattr(self._band, name)

            def SetNoDataValue(self, value):
                self._dataset_proxy._nodata = float(value) if value is not None else None
                return self._band.SetNoDataValue(value)

        class _CheckedDatasetProxy:
            def __init__(self, ds, path, expected_crs=None, expected_dtype=None):
                self._ds = ds
                self._path = Path(path)
                self._expected_crs = expected_crs
                self._expected_dtype = expected_dtype
                self._nodata = None
                self._validated = False

            def __getattr__(self, name):
                return getattr(self._ds, name)

            def GetRasterBand(self, idx):
                return _CheckedBandProxy(self._ds.GetRasterBand(idx), self)

            def SetProjection(self, proj):
                self._expected_crs = proj
                return self._ds.SetProjection(proj)

            def FlushCache(self):
                result = self._ds.FlushCache()
                self._validate()
                return result

            def _validate(self):
                if self._validated or not self._path.exists():
                    return
                self._validated = True
                validate_gdal_output(
                    self._path,
                    operation=f"gdal.create:{self._path.name}",
                    expected_crs=self._expected_crs,
                    expected_nodata=self._nodata,
                    expected_dtype=self._expected_dtype,
                )

            def __del__(self):
                try:
                    self._validate()
                except Exception:
                    pass

        class _CheckedDriverProxy:
            def __init__(self, drv):
                self._drv = drv

            def __getattr__(self, name):
                return getattr(self._drv, name)

            def Create(self, path, xsize, ysize, bands=1, eType=0, options=None):
                ds = self._drv.Create(path, xsize, ysize, bands, eType, options=options)
                dtype = _orig_data_type_name(eType).lower() if eType is not None else None
                return _CheckedDatasetProxy(ds, path, expected_dtype=dtype)

        def _patched_get_driver(name):
            drv = _orig_get_driver(name)
            if drv is None:
                return None
            return _CheckedDriverProxy(drv)

        gdal.GetDriverByName = _patched_get_driver
    except Exception:
        pass
