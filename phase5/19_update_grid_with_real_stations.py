"""
19_update_grid_with_real_stations.py
----------------------------------------
Fixes a deeper version of the "markers in water" bug: Phase 1's
dist_to_police_km column (used by Phase 4 to compute safety scores) was
calculated against RANDOMLY PLACED synthetic police stations -- some of
which could be in the Bay of Bengal, same root cause as the map-marker
issue. This meant a cell could be scored as "safer" for being close to
a police station that doesn't actually exist there in reality.

This script recomputes dist_to_police_km for every grid cell using
Phase 5's REAL OpenStreetMap police station locations instead, and
updates which cells are flagged is_police_station (used by Phase 4's
grid-based routing demo, 14_safe_routing.py).

*** Run this AFTER Phase 5's REAL 15_download_road_network.py. ***

*** After running this, you MUST re-run, in order:
      phase4/12_generate_safety_map.py   (recomputes safety scores)
      phase4/14_safe_routing.py          (optional, if you use this demo)
      phase6/18_export_frontend_data.py  (re-exports map data)
    ...since all of them depend on the values this script updates. ***
"""

import numpy as np
import pandas as pd

PHASE1_OUTPUTS_DIR = "../phase1/outputs"
PHASE5_OUTPUTS_DIR = "../phase5/outputs"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    grid = pd.read_csv(f"{PHASE1_OUTPUTS_DIR}/grid_cells.csv")
    pois = pd.read_csv(f"{PHASE5_OUTPUTS_DIR}/hospitals_police.csv")
    stations = pois[pois["amenity"] == "police"].dropna(subset=["lat", "lon"])

    if len(stations) == 0:
        raise ValueError(
            "No police stations found in hospitals_police.csv. Make sure "
            "you ran the REAL phase5/15_download_road_network.py (not the "
            "15b mock version) with internet access."
        )

    print(f"Recomputing distances against {len(stations)} REAL police stations...")

    station_lats = stations["lat"].to_numpy()
    station_lons = stations["lon"].to_numpy()

    new_distances = []
    for _, cell in grid.iterrows():
        dists = haversine_km(cell["center_lat"], cell["center_lon"], station_lats, station_lons)
        new_distances.append(dists.min())

    old_mean = grid["dist_to_police_km"].mean() if "dist_to_police_km" in grid.columns else None
    grid["dist_to_police_km"] = new_distances

    # re-flag is_police_station: True only for the grid cell closest to
    # each real station (used by Phase 4's grid-based routing demo)
    grid["is_police_station"] = False
    for station_pos in range(len(stations)):
        cell_dists = haversine_km(
            grid["center_lat"], grid["center_lon"],
            station_lats[station_pos], station_lons[station_pos]
        )
        closest_cell_idx = cell_dists.idxmin()
        grid.loc[closest_cell_idx, "is_police_station"] = True

    grid.to_csv(f"{PHASE1_OUTPUTS_DIR}/grid_cells.csv", index=False)

    print(f"Updated dist_to_police_km: old mean={old_mean:.2f} km -> "
          f"new mean={grid['dist_to_police_km'].mean():.2f} km")
    print(f"Flagged {grid['is_police_station'].sum()} grid cells as "
          f"is_police_station=True (one per real station)")
    print(f"\nSaved updated {PHASE1_OUTPUTS_DIR}/grid_cells.csv")
    print("\nNEXT STEPS -- re-run these in order:")
    print("  1. phase4/12_generate_safety_map.py")
    print("  2. phase4/14_safe_routing.py   (if you use this demo)")
    print("  3. phase6/18_export_frontend_data.py")


if __name__ == "__main__":
    main()
