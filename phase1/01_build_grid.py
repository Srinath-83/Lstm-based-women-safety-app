"""
01_build_grid.py
-----------------
Builds the spatial grid over Chennai: divides the city's bounding box into
small square cells (matching the base paper's 0.2 sq km cell size), and
assigns each cell:
    - a unique cell_id
    - its center latitude/longitude
    - a dominant "area_type" (residential, market, bus_stand, etc.)
    - a simulated "streetlight_score" (0-1, higher = better lit)
    - distance to the nearest simulated police station

Run this first. It creates: outputs/grid_cells.csv
"""

import numpy as np
import pandas as pd
import config as cfg

np.random.seed(cfg.RANDOM_SEED)


def build_grid():
    """Create the N x N grid of cells covering the city bounding box."""
    lats = np.arange(cfg.LAT_MIN, cfg.LAT_MAX, cfg.CELL_SIZE_DEG_LAT)
    lons = np.arange(cfg.LON_MIN, cfg.LON_MAX, cfg.CELL_SIZE_DEG_LON)

    rows = []
    cell_id = 0
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            rows.append({
                "cell_id": cell_id,
                "row": i,
                "col": j,
                "lat_min": lat,
                "lat_max": lat + cfg.CELL_SIZE_DEG_LAT,
                "lon_min": lon,
                "lon_max": lon + cfg.CELL_SIZE_DEG_LON,
                "center_lat": lat + cfg.CELL_SIZE_DEG_LAT / 2,
                "center_lon": lon + cfg.CELL_SIZE_DEG_LON / 2,
            })
            cell_id += 1

    grid = pd.DataFrame(rows)
    print(f"Built a grid of {len(grid)} cells "
          f"({grid['row'].nunique()} rows x {grid['col'].nunique()} cols) "
          f"over {cfg.CITY_NAME}.")
    return grid


def assign_area_types(grid):
    """
    Assign a dominant area_type to each cell.

    We don't just assign types uniformly at random -- real cities have
    *clusters* (e.g., a market area is usually several adjacent cells, not
    one isolated cell). We simulate this by picking a handful of "seed"
    cells for each non-residential type and letting nearby cells have a
    higher chance of inheriting that type too. Anything left over defaults
    to residential/commercial by the configured weights.
    """
    n = len(grid)
    area_type = np.array(["residential"] * n, dtype=object)

    coords = grid[["center_lat", "center_lon"]].to_numpy()

    non_residential_types = [t for t in cfg.AREA_TYPES if t not in ("residential",)]
    for a_type in non_residential_types:
        weight = cfg.AREA_TYPE_WEIGHTS[a_type]
        n_seeds = max(1, int(weight * n / 8))  # each seed "claims" ~8 cells worth
        seed_idx = np.random.choice(n, size=n_seeds, replace=False)

        for s in seed_idx:
            seed_coord = coords[s]
            dist = np.sqrt(((coords - seed_coord) ** 2).sum(axis=1))
            # cells within a small radius have a chance to take this area type
            radius = cfg.CELL_SIZE_DEG_LAT * np.random.uniform(1.5, 3.5)
            nearby = np.where(dist < radius)[0]
            claim_prob = np.clip(1 - dist[nearby] / radius, 0.1, 0.9)
            claims = np.random.rand(len(nearby)) < claim_prob
            area_type[nearby[claims]] = a_type

    grid["area_type"] = area_type

    # Convert any leftover "residential" cells: some become "commercial"
    # per the configured weight, to avoid an unrealistically dominant
    # residential class.
    resid_idx = np.where(area_type == "residential")[0]
    commercial_target = int(cfg.AREA_TYPE_WEIGHTS["commercial"] * n)
    if len(resid_idx) > commercial_target:
        flip_idx = np.random.choice(resid_idx, size=commercial_target, replace=False)
        grid.loc[flip_idx, "area_type"] = "commercial"

    print("Area type distribution:")
    print(grid["area_type"].value_counts())
    return grid


def assign_streetlights_and_police(grid):
    """
    Simulate a streetlight coverage score per cell (0-1) and each cell's
    distance to the nearest police station.

    Assumption: commercial/market/bus_stand/entertainment areas tend to be
    better lit than purely residential backstreets (documented assumption,
    replace with real municipal streetlight data if you can obtain it later).
    """
    n = len(grid)

    base_light = {
        "residential":   0.45,
        "commercial":    0.75,
        "market":        0.70,
        "bus_stand":     0.80,
        "entertainment": 0.85,
        "educational":   0.65,
        "industrial":    0.55,
        "hospital":      0.85,
    }
    grid["streetlight_score"] = grid["area_type"].map(base_light) + \
        np.random.normal(0, 0.08, n)
    grid["streetlight_score"] = grid["streetlight_score"].clip(0.05, 1.0)

    # Simulate ~25 police stations scattered across the city, then compute
    # each cell's distance (km) to the nearest one.
    n_stations = 25
    station_idx = np.random.choice(n, size=n_stations, replace=False)
    stations = grid.loc[station_idx, ["center_lat", "center_lon"]].to_numpy()
    grid["is_police_station"] = grid["cell_id"].isin(grid.loc[station_idx, "cell_id"])

    coords = grid[["center_lat", "center_lon"]].to_numpy()
    dists_km = []
    for c in coords:
        d = np.sqrt(((stations - c) ** 2).sum(axis=1))
        d_km = d.min() * 111.0  # rough deg -> km conversion
        dists_km.append(d_km)
    grid["dist_to_police_km"] = dists_km

    print(f"Placed {n_stations} simulated police stations.")
    print(f"Median distance to nearest police station: "
          f"{grid['dist_to_police_km'].median():.2f} km")
    return grid


if __name__ == "__main__":
    grid = build_grid()
    grid = assign_area_types(grid)
    grid = assign_streetlights_and_police(grid)

    out_path = "outputs/grid_cells.csv"
    import os
    os.makedirs("outputs", exist_ok=True)
    grid.to_csv(out_path, index=False)
    print(f"\nSaved grid to {out_path}")
    print(grid.head())
