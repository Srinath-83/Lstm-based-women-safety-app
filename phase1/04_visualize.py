"""
04_visualize.py
----------------
Sanity-check plots for Phase 1:
    1. Map of area types across the grid
    2. Map of streetlight coverage
    3. Heatmap of night-time crime density (aggregated over all days)
    4. Heatmap of women-safety-relevant crime density

Run this AFTER 01, 02, and 03.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import config as cfg

grid = pd.read_csv("outputs/grid_cells.csv")
crime = pd.read_csv("outputs/crime_events.csv")

n_rows = grid["row"].max() + 1
n_cols = grid["col"].max() + 1

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# --- Plot 1: Area types -----------------------------------------------------
area_type_codes = {t: i for i, t in enumerate(cfg.AREA_TYPES)}
area_grid = np.full((n_rows, n_cols), np.nan)
for _, row in grid.iterrows():
    area_grid[int(row["row"]), int(row["col"])] = area_type_codes[row["area_type"]]

cmap = plt.get_cmap("tab10", len(cfg.AREA_TYPES))
im0 = axes[0, 0].imshow(area_grid, cmap=cmap, origin="lower")
axes[0, 0].set_title("Area Type per Grid Cell")
cbar0 = fig.colorbar(im0, ax=axes[0, 0], ticks=range(len(cfg.AREA_TYPES)))
cbar0.ax.set_yticklabels(cfg.AREA_TYPES)

# --- Plot 2: Streetlight coverage ------------------------------------------
light_grid = np.full((n_rows, n_cols), np.nan)
for _, row in grid.iterrows():
    light_grid[int(row["row"]), int(row["col"])] = row["streetlight_score"]

im1 = axes[0, 1].imshow(light_grid, cmap="YlOrRd_r", origin="lower")
axes[0, 1].set_title("Simulated Streetlight Coverage (darker = poorly lit)")
fig.colorbar(im1, ax=axes[0, 1])

# mark police stations
police_cells = grid[grid["is_police_station"] == True]
axes[0, 1].scatter(police_cells["col"], police_cells["row"],
                    marker="*", s=120, c="blue", edgecolors="white",
                    label="Police station")
axes[0, 1].legend(loc="upper right")

# --- Plot 3: Night-time crime density (aggregated) --------------------------
night_crime = crime[crime["time_block"] == "night"]
crime_counts = night_crime.groupby("cell_id")["crime_occurred"].sum().reset_index()
crime_counts = grid[["cell_id", "row", "col"]].merge(crime_counts, on="cell_id", how="left")
crime_counts["crime_occurred"] = crime_counts["crime_occurred"].fillna(0)

crime_grid = np.full((n_rows, n_cols), np.nan)
for _, row in crime_counts.iterrows():
    crime_grid[int(row["row"]), int(row["col"])] = row["crime_occurred"]

im2 = axes[1, 0].imshow(crime_grid, cmap="inferno", origin="lower")
axes[1, 0].set_title(f"Night-time Crime Count over {cfg.NUM_DAYS} Days (all cells)")
fig.colorbar(im2, ax=axes[1, 0])

# --- Plot 4: Women-safety-relevant crime density -----------------------------
women_crime = crime.groupby("cell_id")["women_safety_relevant"].sum().reset_index()
women_crime = grid[["cell_id", "row", "col"]].merge(women_crime, on="cell_id", how="left")
women_crime["women_safety_relevant"] = women_crime["women_safety_relevant"].fillna(0)

women_grid = np.full((n_rows, n_cols), np.nan)
for _, row in women_crime.iterrows():
    women_grid[int(row["row"]), int(row["col"])] = row["women_safety_relevant"]

im3 = axes[1, 1].imshow(women_grid, cmap="magma", origin="lower")
axes[1, 1].set_title(f"Women-Safety-Relevant Crime Count over {cfg.NUM_DAYS} Days")
fig.colorbar(im3, ax=axes[1, 1])

for ax in axes.flat:
    ax.set_xlabel("Grid column (West -> East)")
    ax.set_ylabel("Grid row (South -> North)")

plt.tight_layout()
plt.savefig("outputs/phase1_overview.png", dpi=150)
print("Saved outputs/phase1_overview.png")

# --- Quick printed summary --------------------------------------------------
print("\n--- Phase 1 Summary ---")
print(f"City: {cfg.CITY_NAME}")
print(f"Grid size: {n_rows} x {n_cols} = {n_rows * n_cols} cells "
      f"(~{cfg.CELL_AREA_SQKM} sq km each)")
print(f"Days simulated: {cfg.NUM_DAYS}")
print(f"Total crime-positive cell-blocks: {crime['crime_occurred'].sum()}")
print(f"Overall positive rate: "
      f"{100 * crime['crime_occurred'].mean():.3f}% "
      f"(base paper's cities were ~0.6-2% -- comparable order of magnitude)")
