"""
18_export_frontend_data.py  (UPDATED — real hospitals/police)
------------------------------------------------------------------
Same as the original version, EXCEPT: hospitals and police stations now
come from Phase 5's REAL OpenStreetMap download
(phase5/outputs/hospitals_police.csv) instead of Phase 1's randomly
placed synthetic markers.

WHY THIS CHANGED: Phase 1 placed "hospital"/"police station" cells by
picking random grid cells inside a rectangular bounding box, with no
awareness of the coastline -- so some landed in the Bay of Bengal. Real
OpenStreetMap data doesn't have this problem, since it reflects actual
hospitals/police stations that obviously exist on land.

Each real POI is also snapped to its nearest grid cell (reusing the
exact same lat/lon -> cell math as Phase 1 and Phase 5) purely so we can
show that cell's model-predicted safety score in the marker's popup --
the actual marker POSITION on the map still uses the POI's real lat/lon,
not the cell center, so it displays in the correct real-world spot.

*** IMPORTANT: run this AFTER Phase 5's REAL 15_download_road_network.py
(NOT the 15b mock fallback). Mock data is randomly generated and can
still land in water -- the exact same problem you're trying to fix. ***

Run this AFTER Phase 4's 12_generate_safety_map.py.
"""

import json
import numpy as np
import pandas as pd

PHASE1_OUTPUTS_DIR = "../phase1/outputs"
PHASE4_OUTPUTS_DIR = "../phase4/outputs"
PHASE5_OUTPUTS_DIR = "../phase5/outputs"
OUT_DIR = "app/frontend_data"

SAFETY_AVOIDANCE_STRENGTH = 3.0


def export_safety_cells(scores_df):
    features = []
    for _, row in scores_df.iterrows():
        poly = [[
            [row["lon_min"], row["lat_min"]],
            [row["lon_max"], row["lat_min"]],
            [row["lon_max"], row["lat_max"]],
            [row["lon_min"], row["lat_max"]],
            [row["lon_min"], row["lat_min"]],
        ]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": poly},
            "properties": {
                "cell_id": int(row["cell_id"]),
                "safety_score": round(float(row["safety_score"]), 1),
                "crime_probability": round(float(row["crime_probability"]), 3),
                "area_type": row["area_type"],
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(f"{OUT_DIR}/safety_cells.geojson", "w") as f:
        json.dump(geojson, f)
    print(f"Saved safety_cells.geojson ({len(features)} cells)")


def latlon_to_cell_id(lat, lon, lat_min, lon_min, cell_size_lat, cell_size_lon):
    """Same grid math as Phase 1 / Phase 5 -- keeps everything aligned."""
    row = int(np.floor((lat - lat_min) / cell_size_lat))
    col = int(np.floor((lon - lon_min) / cell_size_lon))
    return row, col


def export_real_pois(scores_df, amenity_type, filename, label):
    """
    NEW: builds hospital/police GeoJSON from Phase 5's real OSM download,
    instead of Phase 1's synthetic in-grid markers.
    """
    pois = pd.read_csv(f"{PHASE5_OUTPUTS_DIR}/hospitals_police.csv")
    subset = pois[pois["amenity"] == amenity_type].dropna(subset=["lat", "lon"])

    lat_min = scores_df["lat_min"].min()
    lon_min = scores_df["lon_min"].min()
    cell_size_lat = scores_df["lat_max"].iloc[0] - scores_df["lat_min"].iloc[0]
    cell_size_lon = scores_df["lon_max"].iloc[0] - scores_df["lon_min"].iloc[0]

    score_lookup = scores_df.set_index(["row", "col"])["safety_score"].to_dict()
    city_avg_score = scores_df["safety_score"].mean()

    features = []
    n_outside = 0
    for _, poi in subset.iterrows():
        row, col = latlon_to_cell_id(
            poi["lat"], poi["lon"], lat_min, lon_min, cell_size_lat, cell_size_lon
        )
        safety_score = score_lookup.get((row, col))
        if safety_score is None:
            safety_score = city_avg_score
            n_outside += 1

        name = poi["name"] if pd.notna(poi.get("name")) else label
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [poi["lon"], poi["lat"]]},
            "properties": {
                "name": name,
                "safety_score": round(float(safety_score), 1),
                "label": label,
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(f"{OUT_DIR}/{filename}", "w") as f:
        json.dump(geojson, f)
    print(f"Saved {filename} ({len(features)} REAL {label.lower()} locations"
          f"{f', {n_outside} outside grid bounds (used city-avg score)' if n_outside else ''})")


def build_routing_graph(scores_df, n_rows, n_cols):
    score_lookup = scores_df.set_index("cell_id")["safety_score"].to_dict()
    cell_id_grid = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)

    nodes = []
    for _, row in scores_df.iterrows():
        nodes.append({
            "id": int(row["cell_id"]),
            "row": int(row["row"]),
            "col": int(row["col"]),
            "lat": round(float(row["center_lat"]), 6),
            "lon": round(float(row["center_lon"]), 6),
            "safety_score": round(float(row["safety_score"]), 1),
        })

    edges = []
    seen = set()
    for r in range(n_rows):
        for c in range(n_cols):
            cell_id = int(cell_id_grid[r, c])
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    neighbor_id = int(cell_id_grid[rr, cc])
                    edge_key = tuple(sorted((cell_id, neighbor_id)))
                    if edge_key in seen:
                        continue
                    seen.add(edge_key)

                    physical_dist = round(float(np.sqrt(dr ** 2 + dc ** 2)), 4)
                    dest_risk = 1 - (score_lookup[neighbor_id] / 100)
                    safety_weight = round(
                        physical_dist + SAFETY_AVOIDANCE_STRENGTH * dest_risk, 4
                    )
                    edges.append([cell_id, neighbor_id, safety_weight, physical_dist])

    graph = {"nodes": nodes, "edges": edges}
    with open(f"{OUT_DIR}/graph.json", "w") as f:
        json.dump(graph, f)
    print(f"Saved graph.json ({len(nodes)} nodes, {len(edges)} edges)")


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    grid = pd.read_csv(f"{PHASE1_OUTPUTS_DIR}/grid_cells.csv")
    scores = pd.read_csv(f"{PHASE4_OUTPUTS_DIR}/safety_scores.csv")

    scores = scores.merge(
        grid[["cell_id", "area_type"]],
        on="cell_id", how="left", suffixes=("", "_grid")
    )
    if "area_type_grid" in scores.columns:
        scores["area_type"] = scores["area_type_grid"]

    n_rows = int(scores["row"].max()) + 1
    n_cols = int(scores["col"].max()) + 1
    print(f"Grid: {n_rows} x {n_cols} = {len(scores)} cells")

    export_safety_cells(scores)
    export_real_pois(scores, "police", "police_stations.geojson", "Police Station")
    export_real_pois(scores, "hospital", "hospitals.geojson", "Hospital")
    build_routing_graph(scores, n_rows, n_cols)

    print("\nAll frontend data exported to app/frontend_data/")


if __name__ == "__main__":
    main()
