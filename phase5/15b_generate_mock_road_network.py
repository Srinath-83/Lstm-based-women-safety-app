"""
15b_generate_mock_road_network.py
------------------------------------
OPTIONAL FALLBACK: generates a synthetic, Manhattan-style street grid
over the same Chennai bounding box as Phase 1 -- so you can test/develop
the rest of Phase 5 (safety-score merging, offline routing, folium
visualization) WITHOUT needing internet access to OpenStreetMap.

Use this if:
  - You want to test the Phase 5 pipeline quickly before doing the real
    (larger, slower) OSM download in 15_download_road_network.py, or
  - You genuinely don't have internet access on your development
    machine right now.

This is NOT a substitute for the real thing in your final report --
real streets aren't a perfect grid. Swap to 15_download_road_network.py
(run with real internet) before generating your final results/figures.
"""

import numpy as np
import networkx as nx
import osmnx as ox
import pandas as pd
import config_phase5 as cfg

np.random.seed(cfg.RANDOM_SEED)

# same bounding box as Phase 1 (see phase1/config.py)
LAT_MIN, LAT_MAX = 12.83, 13.23
LON_MIN, LON_MAX = 80.13, 80.31

# finer spacing than the analysis grid, to mimic real street density
# (real city blocks are often 80-150m)
INTERSECTION_SPACING_KM = 0.12
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians((LAT_MIN + LAT_MAX) / 2))
SPACING_DEG_LAT = INTERSECTION_SPACING_KM / KM_PER_DEG_LAT
SPACING_DEG_LON = INTERSECTION_SPACING_KM / KM_PER_DEG_LON


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_mock_network():
    lats = np.arange(LAT_MIN, LAT_MAX, SPACING_DEG_LAT)
    lons = np.arange(LON_MIN, LON_MAX, SPACING_DEG_LON)
    n_rows, n_cols = len(lats), len(lons)
    print(f"Building mock street grid: {n_rows} x {n_cols} = "
          f"{n_rows * n_cols} intersections")

    G = nx.MultiDiGraph(crs="epsg:4326")

    node_id_grid = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            node_id = int(node_id_grid[i, j])
            G.add_node(node_id, y=lat, x=lon)

    # connect each intersection to its right and up neighbor (avoids adding
    # each edge twice); randomly DROP a fraction of edges to mimic the fact
    # that real streets aren't a perfect uninterrupted grid
    drop_probability = 0.15
    for i in range(n_rows):
        for j in range(n_cols):
            node_id = int(node_id_grid[i, j])
            lat, lon = lats[i], lons[j]

            if j + 1 < n_cols and np.random.rand() > drop_probability:
                right_id = int(node_id_grid[i, j + 1])
                length = haversine_m(lat, lon, lat, lons[j + 1])
                G.add_edge(node_id, right_id, key=0, length=length)
                G.add_edge(right_id, node_id, key=0, length=length)

            if i + 1 < n_rows and np.random.rand() > drop_probability:
                up_id = int(node_id_grid[i + 1, j])
                length = haversine_m(lat, lon, lats[i + 1], lon)
                G.add_edge(node_id, up_id, key=0, length=length)
                G.add_edge(up_id, node_id, key=0, length=length)

    print(f"Mock network: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


def generate_mock_pois(n_hospitals=15, n_police=25):
    rows = []
    for i in range(n_hospitals):
        rows.append({
            "amenity": "hospital", "name": f"Mock Hospital {i+1}",
            "lat": np.random.uniform(LAT_MIN, LAT_MAX),
            "lon": np.random.uniform(LON_MIN, LON_MAX),
        })
    for i in range(n_police):
        rows.append({
            "amenity": "police", "name": f"Mock Police Station {i+1}",
            "lat": np.random.uniform(LAT_MIN, LAT_MAX),
            "lon": np.random.uniform(LON_MIN, LON_MAX),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    G = build_mock_network()
    ox.save_graphml(G, cfg.ROAD_NETWORK_FILE)
    print(f"Saved mock road network to {cfg.ROAD_NETWORK_FILE}")

    pois = generate_mock_pois()
    pois.to_csv(cfg.POI_FILE, index=False)
    print(f"Saved {len(pois)} mock hospital/police locations to {cfg.POI_FILE}")

    print("\nYou can now test 16_merge_safety_scores.py and "
          "17_offline_safe_route.py using this mock data.")
    print("Remember to re-run 15_download_road_network.py (with real "
          "internet) before generating your final report results.")
