# Phase 1 — Data Foundation (Chennai Grid + Synthetic Data)

## What this does

1. **`config.py`** — every constant lives here (city boundary, grid size,
   area-type mix, calibration targets). Edit this file first when you want
   to change anything.

2. **`01_build_grid.py`** — divides Chennai's bounding box into a grid of
   ~0.2 sq km cells (matching the base paper's 0.077 sq. mile cells).
   Assigns each cell an area type (residential/market/bus_stand/etc.),
   a simulated streetlight score, and distance to nearest police station.
   → outputs `outputs/grid_cells.csv`

3. **`02_generate_synthetic_data.py`** — generates:
   - `outputs/mobility_footfall.csv` — simulated foot traffic per cell,
     per day, per 12-hour block (day/night), following realistic patterns
     per area type (e.g. markets busy in the day, bars busy at night).
   - `outputs/crime_events.csv` — simulated crime occurrence (0/1) per
     cell per block. The probability of crime is driven by footfall,
     area-type risk, night-time multiplier, streetlight score, and
     distance to police — then **calibrated** so the city-wide yearly
     total matches `TARGET_ANNUAL_CRIMES_CITYWIDE` in `config.py`.
   - Also flags a subset of crimes as `women_safety_relevant`.

4. **`03_generate_sociodemographic.py`** — generates a lightweight
   population density / literacy / income layer per cell.
   → outputs `outputs/sociodemographic.csv`

5. **`04_visualize.py`** — sanity-check plots: area types, streetlight
   coverage + police stations, night-time crime heatmap, women-safety
   crime heatmap.
   → outputs `outputs/phase1_overview.png`

## How to run

```bash
pip install numpy pandas matplotlib
python3 01_build_grid.py
python3 02_generate_synthetic_data.py
python3 03_generate_sociodemographic.py
python3 04_visualize.py
```

## ⚠️ Important — what to say honestly in your report/viva

This data is **synthetic**, not real police records (real FIR-level,
geotagged crime data isn't publicly available for Tamil Nadu — see our
earlier discussion). But it isn't arbitrary noise either:

- The crime probability model is grounded in **Routine Activity Theory**
  (the same theory the base paper cites) — risk scales with footfall,
  time-of-day, and guardianship factors (lighting, police proximity).
- It's **calibrated**: the city-wide annual total is forced to match a
  real, citable number (`TARGET_ANNUAL_CRIMES_CITYWIDE` in `config.py`)
  — replace the placeholder with the actual NCRB/TN Police figure for
  Chennai once you've pulled it from the sources we found earlier
  (data.opencity.in, NCRB "Crime in India" report).
- The resulting sparsity (~1.4% positive rate) matches the real cities
  in the base paper (0.6-2%), which is a good realism check.

**Next real-data step**: once you have the actual NCRB/TN Police
district-level yearly numbers, update `TARGET_ANNUAL_CRIMES_CITYWIDE`
and `WOMEN_SAFETY_CRIME_SHARE` in `config.py` with the cited figures —
this is a 2-line edit and instantly makes your calibration defensible
with a source.

## Next: Phase 2

Phase 2 will take these three CSVs and reshape them into the
"stacked frames" structure the base paper uses (like Fig. 2 in the
paper) — one N×N×channels frame per 12-hour block — ready to feed into
an LSTM/ConvLSTM in Phase 3.
