# Workflow script documentation style guide

Use this as a lightweight standard for scripts in `workflow/`.

## Module docstring template

"""
<Script name> — <one-line purpose>.

What it does
- Bullet list of major steps.

Inputs
- Key inputs (files/URLs/args) and formats.

Outputs
- Files produced, naming conventions, CRS assumptions, nodata rules.

Typical usage
- 1–3 examples (copy/paste runnable).

Notes
- Assumptions (datums, mask semantics, units).
"""

## CLI conventions

- `--help` should describe required inputs and outputs clearly.
- Prefer explicit units in arg names or help (e.g., `--spacing-m`).
- Include defaults in help text.

## Logging conventions

- Use `logging.getLogger(__name__)`
- Log major steps like:
  - "[STEP] ..."
  - "[WRITE] <path>"

## Geo conventions

- State CRS expectations (EPSG) for inputs + outputs.
- State nodata and mask semantics (e.g., water=0, land=1).
- If depths/elevations are involved, state sign convention and datum.
