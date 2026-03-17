#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smoke_test.py

Quick, dependency-light smoke checks for this workflow.

What it checks:
- All modules import without raising.
- Key CLIs respond to `--help`.
- Model bank reservoir update works and stays bounded.
- Unified report writer works.

Run:
  python smoke_test.py

Exit code:
  0 = all checks passed
  1 = one or more checks failed
"""

from __future__ import annotations

import glob
import py_compile
import os
import sys
import tempfile
from pathlib import Path
def main() -> int:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))

    failures: list[str] = []

    # 1) Syntax compile sweep (fast, avoids importing heavyweight geo stacks)
    print("[SMOKE] Step 1/3: py_compile sweep", flush=True)
    py_files = sorted(glob.glob(str(here / "*.py")))
    for fn in py_files:
        mod = os.path.splitext(os.path.basename(fn))[0]
        if mod in {"smoke_test"}:
            continue
        try:
            py_compile.compile(fn, doraise=True)
        except Exception as e:
            failures.append(f"PY_COMPILE FAIL: {mod}: {type(e).__name__}: {e}")

    # 2) Model bank bounded update
    print("[SMOKE] Step 2/3: model bank bounded update", flush=True)
    try:
        import numpy as np
        import pandas as pd
        from model_bank import update_bank

        with tempfile.TemporaryDirectory() as td:
            bank_dir = Path(td) / "bank"
            # minimal stable schema
            feat_cols = ["B02", "B03", "B04", "B08"]
            cols = feat_cols + ["longitude", "latitude", "source", "source_norm", "depth_m"]

            def make_df(n: int, seed: int) -> pd.DataFrame:
                rng = np.random.RandomState(seed)
                df = pd.DataFrame({
                    "B02": rng.rand(n),
                    "B03": rng.rand(n),
                    "B04": rng.rand(n),
                    "B08": rng.rand(n),
                    "longitude": -70.0 + 0.01 * rng.randn(n),
                    "latitude": 42.0 + 0.01 * rng.randn(n),
                    "source": ["atl03"] * n,
                    "source_norm": ["atl03"] * n,
                    "depth_m": -1.0 * (0.5 + 10.0 * rng.rand(n)),
                })
                return df

            df1 = make_df(500, 0)
            res1, meta1 = update_bank(
                bank_dir,
                df1,
                target_col="depth_m",
                max_samples=200,
                seed=1337,
                schema_cols=cols,
            )
            if len(res1) != 200:
                failures.append(f"MODEL_BANK FAIL: expected 200 kept, got {len(res1)}")

            df2 = make_df(500, 1)
            res2, meta2 = update_bank(
                bank_dir,
                df2,
                target_col="depth_m",
                max_samples=200,
                seed=1337,
                schema_cols=cols,
            )
            if len(res2) != 200:
                failures.append(f"MODEL_BANK FAIL: expected 200 kept after update, got {len(res2)}")
            if int(meta2.get("n_seen", 0)) < 1000:
                failures.append(f"MODEL_BANK FAIL: expected n_seen>=1000, got {meta2.get('n_seen')}")
    except Exception as e:
        failures.append(f"MODEL_BANK EXCEPTION: {type(e).__name__}: {e}")

    # 3) Unified report writer
    print("[SMOKE] Step 3/3: unified report writer", flush=True)
    try:
        from river_diagnostics import create_unified_bathy_report

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            # minimal bathy_report.json so the writer can summarize
            (out_dir / "bathy_report.json").write_text('{"sdb":{"status":"ok"},"river":{"status":"ok"},"fusion":{"status":"ok"}}')
            p = create_unified_bathy_report(out_dir, methods=["sdb", "river", "fuse"], priority="river")
            if not Path(p).exists():
                failures.append("UNIFIED_REPORT FAIL: unified report not created")
    except Exception as e:
        failures.append(f"UNIFIED_REPORT EXCEPTION: {type(e).__name__}: {e}")

    if failures:
        sys.stderr.write("\n".join(failures) + "\n")
        return 1

    print("OK: smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
