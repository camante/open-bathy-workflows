# Coastal and River Seam Stability Framework

This framework defines how to think about seam stability in the current workflow.
It is intentionally compatible with both the in-run `bathy_main.py` seam comparison and the standalone seam-stability tools in this directory.

## Core idea

A workflow is seam-stable when adjacent runs, adjacent tiles, and river-to-coast transitions do not create avoidable artificial steps beyond what the source data and domain differences justify.

## Seam types to review

### 1. Coastal–coastal

Compare neighboring SDB-driven products along their shared boundary region.

### 2. River–river

Compare neighboring river outputs while respecting the river/channel domain.

### 3. Coastal–river transition

Compare the estuarine or transition zone where SDB and river products overlap, hand off, or are fused.

## What to archive with every seam review

- the compared run manifests
- the compared run reports
- the seam-metric outputs
- any gate or summary JSON produced from those metrics

## Practical rule

If a seam problem is discovered, record whether it originates from:

- source-data disagreement
- mask/domain mismatch
- CRS or datum drift
- model-bank drift
- fusion priority behavior
- river-method geometry or hydraulic prioring

The goal is not just to measure a seam but to make the cause auditable.
