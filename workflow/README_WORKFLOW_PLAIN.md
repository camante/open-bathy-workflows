# Open Bathy Workflows – Plain-language guide

This project makes “underwater maps” (bathymetry) using a mix of satellite images and river mapping data.

It can do two main jobs:

1) **Coastal / ocean bathymetry (SDB)**  
   Uses Sentinel‑2 satellite imagery to estimate water depth near the coast.

2) **River bathymetry**  
   Uses river polygons (NHDArea) to estimate where the river channel is, then predicts a smooth river bottom that is safe for modeling.

You can run either one, or both together.

---

## The most important idea: “Only predict in the right water”

The workflow keeps **separate areas (domains)** for the coast and for rivers:

- **Coast domain (SDB):** where ocean/coastal water is allowed  
- **River domain:** where river water is allowed (from NHDArea)

Sometimes these can overlap near river mouths. The workflow handles this in a predictable way.

### The rule the code follows

- If you run **SDB only**, final results are clipped to the **coast water mask**
- If you run **River only**, final results are clipped to the **river mask**
- If you run **both**, final results are clipped to the **combined “coast + river” mask**

This prevents weird outputs like:
- depth appearing on land
- depth appearing in lakes when you only want rivers
- depth appearing outside your area of interest

---

## What file should I look at?

After the run finishes, the main output is usually:

- `combined/bathy_depth_final_epsg4269.tif`

If you ran the river method, you will also get:

- `river/river_depth_final_epsg4269.tif`
- `river/river_bottom_navd88_final_epsg4269.tif`

---

## One command to run both SDB + River

```bash
PYTHONUNBUFFERED=1 python -u bathy_main.py \
  --aoi="-71.25/-71.00/42.75/43.00" \
  --start="2025-01-01" --end="2026-01-01" \
  --methods=sdb,river \
  --out-dir="output/my_run" \
  --cache-root="cache/my_run"
```

---

## Common problems

### “I see depths outside where I expected”
That usually means the correct mask wasn’t found or wasn’t applied.
Make sure:
- the waffles coastline masks were created
- the river channel mask (`river_channel_mask.tif`) exists
- your `--methods` setting matches what you intended

### “The summary metrics say no channel mask found”
The metrics script needs a channel mask to measure the results.
It can now also read `bathy_report.json` to find the cached mask automatically.

### “Weird circles at stream junctions”
Those were caused by how junction smoothing blended values from small tributaries into the main channel.
The river method now uses a “main channel width proxy” approach so the main river stays continuous.

