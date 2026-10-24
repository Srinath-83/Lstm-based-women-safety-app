"""
02_generate_synthetic_data.py
------------------------------
Generates two datasets for every cell x day x time-block combination:

  1. mobility_footfall.csv  -- simulated foot traffic per cell per block
  2. crime_events.csv       -- simulated crime occurrence (0/1) per cell
                                per block, calibrated so the CITY-WIDE
                                yearly total roughly matches a real,
                                citable NCRB/TN Police figure.

This mirrors the base paper's approach: crime probability is driven by
footfall + area-type risk + guardianship factors (streetlights, distance
to police), not just random noise -- so the LSTM/ConvLSTM has real
patterns to learn later in Phase 3.

Run this AFTER 01_build_grid.py.
"""

import numpy as np
import pandas as pd
import config as cfg

np.random.seed(cfg.RANDOM_SEED)

# ---------------------------------------------------------------------------
# Hourly footfall "shape" per area type, collapsed into day/night blocks.
# Values are relative weights, not absolute counts (0-1 scale then scaled).
# ---------------------------------------------------------------------------
FOOTFALL_PROFILE = {
    # (day_footfall, night_footfall) as relative intensity 0-1
    "residential":   (0.45, 0.35),
    "commercial":    (0.80, 0.25),
    "market":        (0.90, 0.20),
    "bus_stand":     (0.85, 0.55),
    "entertainment": (0.40, 0.90),
    "educational":   (0.85, 0.05),
    "industrial":    (0.70, 0.20),
    "hospital":      (0.75, 0.60),
}


def generate_mobility(grid, num_days):
    """
    For each cell and each 12-hour block over `num_days`, generate a
    footfall value. We add day-of-week effects (weekends busier for
    market/entertainment) and random daily noise.
    """
    records = []
    cell_ids = grid["cell_id"].to_numpy()
    area_types = grid["area_type"].to_numpy()

    for day in range(num_days):
        day_of_week = day % 7
        is_weekend = day_of_week in (5, 6)

        for block in cfg.TIME_BLOCKS:
            for c_id, a_type in zip(cell_ids, area_types):
                base_day, base_night = FOOTFALL_PROFILE[a_type]
                base = base_day if block == "day" else base_night

                weekend_boost = 1.0
                if is_weekend and a_type in ("market", "entertainment"):
                    weekend_boost = 1.3
                elif is_weekend and a_type == "educational":
                    weekend_boost = 0.3  # schools/colleges quiet on weekends

                noise = np.random.normal(1.0, 0.15)
                footfall = max(0.0, base * weekend_boost * noise)

                records.append((day, day_of_week, block, c_id, footfall))

    mobility = pd.DataFrame(
        records,
        columns=["day", "day_of_week", "time_block", "cell_id", "footfall"]
    )
    return mobility


def generate_crime(grid, mobility):
    """
    Generate binary crime occurrence per cell per time block.

    Risk model (illustrative, documented assumption):
        risk_score = area_risk_multiplier
                     * (0.5 + footfall)             # more people -> more risk
                     * night_multiplier              # night riskier than day
                     * (1.2 - streetlight_score)     # poor lighting -> more risk
                     * police_distance_factor        # far from police -> more risk

    The risk_score is then converted to a probability and scaled so the
    CITY-WIDE yearly total matches cfg.TARGET_ANNUAL_CRIMES_CITYWIDE.
    """
    grid_lookup = grid.set_index("cell_id")[
        ["area_type", "streetlight_score", "dist_to_police_km"]
    ]

    df = mobility.merge(grid_lookup, on="cell_id", how="left")

    area_risk = df["area_type"].map(cfg.AREA_TYPE_RISK_MULTIPLIER)
    night_mult = np.where(df["time_block"] == "night", 1.6, 1.0)
    light_factor = 1.2 - df["streetlight_score"]  # darker -> higher factor
    # cap the "far from police" effect so it doesn't blow up in outer cells
    police_factor = 1 + np.clip(df["dist_to_police_km"] / 10.0, 0, 0.5)

    raw_risk = area_risk * (0.5 + df["footfall"]) * night_mult * light_factor * police_factor

    # --- Calibration step ---------------------------------------------------
    # Scale raw_risk into a probability such that:
    #   sum(probability over all cell-blocks in one year) ~= TARGET_ANNUAL_CRIMES
    num_days_per_year = 365
    blocks_per_year = num_days_per_year * len(cfg.TIME_BLOCKS)
    n_cells = grid["cell_id"].nunique()

    # average raw_risk per cell-block in the dataset (proxy for a "typical" year)
    mean_raw_risk = raw_risk.mean()
    total_cell_blocks_per_year = n_cells * blocks_per_year

    # scaling constant k such that: sum(k * raw_risk) over one year == target
    k = cfg.TARGET_ANNUAL_CRIMES_CITYWIDE / (mean_raw_risk * total_cell_blocks_per_year)

    probability = np.clip(raw_risk * k, 0, 0.9)  # cap at 0.9 to avoid certainty
    crime_occurred = (np.random.rand(len(df)) < probability).astype(int)

    df["crime_probability"] = probability
    df["crime_occurred"] = crime_occurred

    # Flag a subset of crimes as "women's safety relevant" per the configured
    # share -- weighted so night-time / entertainment / bus_stand cells are
    # more likely to be tagged this way (again: documented assumption).
    women_relevant_boost = np.where(
        (df["time_block"] == "night") &
        (df["area_type"].isin(["entertainment", "bus_stand", "market"])),
        1.8, 1.0
    )
    women_flag_prob = np.clip(
        cfg.WOMEN_SAFETY_CRIME_SHARE * women_relevant_boost, 0, 1
    )
    df["women_safety_relevant"] = np.where(
        df["crime_occurred"] == 1,
        (np.random.rand(len(df)) < women_flag_prob).astype(int),
        0
    )

    return df[[
        "day", "day_of_week", "time_block", "cell_id",
        "footfall", "crime_probability", "crime_occurred",
        "women_safety_relevant"
    ]]


if __name__ == "__main__":
    import os
    grid = pd.read_csv("outputs/grid_cells.csv")

    print(f"Generating {cfg.NUM_DAYS} days of synthetic mobility data "
          f"for {len(grid)} cells...")
    mobility = generate_mobility(grid, cfg.NUM_DAYS)
    print(f"Mobility dataset shape: {mobility.shape}")

    print("Generating calibrated synthetic crime data...")
    crime = generate_crime(grid, mobility)
    print(f"Crime dataset shape: {crime.shape}")

    # Sanity-check the calibration
    days_generated = cfg.NUM_DAYS
    scale_to_one_year = 365 / days_generated
    total_crimes_generated = crime["crime_occurred"].sum()
    implied_annual = total_crimes_generated * scale_to_one_year
    print(f"\nTotal synthetic crimes generated ({days_generated} days): "
          f"{total_crimes_generated}")
    print(f"Implied annual rate: {implied_annual:.0f} "
          f"(target was {cfg.TARGET_ANNUAL_CRIMES_CITYWIDE})")

    women_crimes = crime["women_safety_relevant"].sum()
    print(f"Of which flagged women-safety-relevant: {women_crimes} "
          f"({100 * women_crimes / total_crimes_generated:.1f}%)")

    os.makedirs("outputs", exist_ok=True)
    mobility.to_csv("outputs/mobility_footfall.csv", index=False)
    crime.to_csv("outputs/crime_events.csv", index=False)
    print("\nSaved outputs/mobility_footfall.csv and outputs/crime_events.csv")
