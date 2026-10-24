# Phase 2 — From Flat CSVs to Training Sequences

## What this does

Phase 1 gave you flat CSVs. A ConvLSTM/LSTM can't read CSVs — it needs
data shaped like a "video": `(time, height, width, channels)`. Phase 2
builds exactly that, then slices it into ready-to-train (X, y) sequences.

### `config_phase2.py`
All the settings for this phase (look-back periods, sub-grid size, etc.)

### `05_build_tensor.py`
Merges `grid_cells.csv` + `crime_events.csv` + `sociodemographic.csv`
into one big 4D array:

```
tensor[t, row, col, channel]
```

- **t** = a 12-hour block (day or night)
- **row, col** = position in the city grid
- **channel** = one of 15 features per cell per block:
  `crime_occurred, footfall, streetlight_score, dist_to_police_km,
  residential, commercial, market, bus_stand, entertainment, educational,
  industrial, hospital` (one-hot area type), `population_density_per_sqkm,
  literacy_rate_pct, income_band`

This mirrors the base paper's Figure 2 exactly — just with our own
(simplified, synthetic-calibrated) feature set instead of their US-city
Advan mobility + ACS census data.

→ Outputs `outputs/city_tensor.npy`, `outputs/channel_names.json`,
`outputs/time_index.csv`

### `06_build_sequences.py`
Takes the big tensor and builds actual training examples, following the
paper's method:

1. **Sliding window**: use the last `T` frames to predict the next one
   (`T=28` → 14-day look-back, `T=4` → 2-day look-back — same two values
   the paper compares)
2. **Random 16×16 sub-grid cropping**: our full grid (100×44) is too big
   to train a ConvLSTM on directly in a student project — so, like the
   paper, we randomly crop smaller 16×16 patches. This also multiplies
   your training examples (data augmentation).
3. **Balancing**: crime is rare (~1.4% of cells/blocks), so a random crop
   is usually empty. We only keep crops with at least
   `MIN_CRIMES_IN_SUBGRID` crimes in the input window — same fix the
   paper uses.
4. **Chronological train/test split** (90/10) with a buffer removed to
   prevent data leakage — the model is always tested on the *future*
   relative to what it trained on.

→ Outputs, per look-back period:
```
outputs/14_days/X_train.npy   shape (n_sequences, 28, 16, 16, 15)
outputs/14_days/y_train.npy   shape (n_sequences, 16, 16)
outputs/14_days/X_test.npy
outputs/14_days/y_test.npy
outputs/2_days/  (same structure, T=4)
```

### `07_visualize_sequence.py`
Sanity-check plot: shows how the crime channel and footfall channel look
across a real training sequence, plus the target frame it's trying to
predict. Good to include as a figure in your report — it's the direct
Tamil Nadu-project equivalent of the paper's Figure 2.

## How to run

```bash
cd phase2
pip install numpy pandas matplotlib
python3 05_build_tensor.py
python3 06_build_sequences.py
python3 07_visualize_sequence.py
```

Make sure `phase1/outputs/` (from Phase 1) sits next to `phase2/` — i.e.
your folder structure should look like:

```
your_project/
├── phase1/
│   └── outputs/
│       ├── grid_cells.csv
│       ├── crime_events.csv
│       └── sociodemographic.csv
└── phase2/
    ├── config_phase2.py
    ├── 05_build_tensor.py
    ├── 06_build_sequences.py
    ├── 07_visualize_sequence.py
    └── outputs/   (created automatically)
```

## Important settings to know about

- **`QUICK_TEST_MODE = True`** in `config_phase2.py` — currently only
  uses 220 days of data so scripts run fast while you're testing. Once
  everything works, **set this to `False`** to build the full dataset
  (all 730 days) for your actual model training in Phase 3. This will
  take longer and produce bigger files, so only do it once you're ready.

- **`MIN_CRIMES_IN_SUBGRID = 2`** — if you get the "not enough
  crime-positive sub-grids found" warning, either lower this number,
  increase `QUICK_TEST_MAX_DAYS`, or (once ready) turn off quick-test
  mode entirely.

## What you should verify before moving to Phase 3

Open `outputs/example_sequence_frames.png` and `outputs/example_target_frame.png`
and check that:
- Crime dots appear in *different* positions/counts across frames (not
  frozen/identical every frame — would mean a bug)
- Footfall shows spatial structure (some areas consistently busier than
  others — not pure random noise)

If both look reasonable, you're ready for Phase 3: building the actual
ConvLSTM model (and the LR/RF/LSTM baselines, like the paper) that learns
to predict the target frame from the input sequence.
