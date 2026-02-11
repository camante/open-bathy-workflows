#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup.py - Package setup for SDB & River Interpolation Pipeline

Usage:
    pip install -e .              # Editable install
    pip install .                 # Regular install
    python setup.py sdist         # Create source distribution
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Satellite-Derived Bathymetry & River Interpolation Pipeline"

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    req_file = Path(__file__).parent / filename
    if not req_file.exists():
        return []
    with open(req_file) as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith('#')]

setup(
    name="sdb-river-pipeline",
    version="0.6.0",
    author="SDB Team",
    author_email="",
    description="Satellite-Derived Bathymetry & River Interpolation Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(exclude=['tests', 'docs']),
    py_modules=[
        'sdb_main',
        'bathy_main',
        'train',
        'predict',
        'predict_chunked',
        'atl',
        's2_optics',
        'fusion',
        'vis',
        'alignment',
        'bottom_physics',
        'physics_integration',
        'river_bathy',
        'river_network',
        'xs_builder',
        'xs_infer_bathy_raster',
        'xs_adjust_monotonic',
        'bathy_fusion',
        'constants',
        'errors',
        'cache_utils',
        'log_report',
        'sdb_uncertainty',
        'spatial_cv',
        'kd_estimation',
        'tidal',
        'optical_utils',
        'training_diversity',
        'diagnose_bands',
        'measured_mask_from_points',
        'make_channel_mask',
        'bank_mask_from_xs',
        'cudem_river_burn_taper',
        'cudem_river_fill_monotonic',
        'swot_adjust',
        'chunked_processing',
        'idw_gpu',
        'hyperparameter_tuning',
        'adaptive_spatial_cv',
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        'dev': read_requirements("requirements-dev.txt"),
        'gpu': ['cupy-cuda11x>=12.0.0'],
    },
    entry_points={
        'console_scripts': [
            'sdb-pipeline=sdb_main:main',
            'river-bathy=river_bathy:main',
            'predict-chunked=predict_chunked:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.json', '*.md', '*.txt'],
    },
    zip_safe=False,
)
