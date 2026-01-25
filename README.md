# open-bathy-workflows

Research workflows for satellite-derived bathymetry (SDB), river bathymetry inference, and provenance-aware intelligent gap filling.

This repository is intended to be a **research workflow repo** that can run standalone and also serve as a staging area for future upstream contributions to **CUDEM / waffles**.

## Quick start (standalone)

> Example (edit AOI/output paths as needed):

```bash
python workflow/bathy_main.py \
  --aoi="-88/-87.75/30.5/30.75" \
  --start 2025-01-01 \
  --end 2026-01-01 \
  --out-dir output/mobile_bay \
  --methods sdb,river \
  --priority sdb
```

## Repository layout

- `workflow/` — the current end-to-end pipeline (as provided in the workflow zip)
- `docs/` — architecture notes and design docs
- `configs/` — example configs (optional)
- `tests/` — lightweight tests (optional)

## Relationship to CUDEM

This repo is a sandbox for rapid iteration. Stable components may later be contributed to CUDEM as waffles modules.
