"""
21_build_coimbatore_real_graph.py
----------------------------------
Replaces Coimbatore's MOCK road graph (shipped so the app works offline in
a sandbox with no internet) with a REAL OpenStreetMap street network, and
upgrades the police/hospital markers from approximate locality coordinates
to the real geocoded ones OSM returns.

WHY THIS IS ONE NEW SCRIPT INSTEAD OF EDITING THE CHENNAI SCRIPTS:
Chennai's equivalent process (phase5/16_merge_safety_scores.py +
phase6/20_export_real_road_graph.py) reads a `safety_scores.csv` produced
by a trained ConvLSTM model (Phase 3/4) -- Coimbatore never had one of
those (out of scope for this project; see README_coimbatore.md). Coimbatore
instead uses a simple, transparent formula: a flat NCRB-2017-derived
citywide baseline + real distance to the nearest real police station. So
this script reimplements that SAME formula from frontend_data/coimbatore's
original generator, but now feeding it REAL road nodes and REAL police
coordinates instead of the mock grid -- copying Chennai's two scripts
line-for-line would silently crash looking for a CSV that doesn't exist.

BEFORE RUNNING THIS: run 15_download_road_network_coimbatore.py first (on
your own machine, with internet). It must have produced, in this same
`phase6/` folder:
    coimbatore_road_network.graphml
    coimbatore_hospitals_police.csv

HOW TO RUN (from the phase6/ folder):
    py 21_build_coimbatore_real_graph.py

WHAT IT OVERWRITES (only Coimbatore's files -- Chennai is untouched):
    app/frontend_data/coimbatore/graph.json
    app/frontend_data/coimbatore/safety_cells.geojson
    app/frontend_data/coimbatore/police_stations.geojson
    app/frontend_data/coimbatore/hospitals.geojson

After it finishes, just hard-refresh the browser (Ctrl+Shift+R) -- no
changes to index.html are needed, since it already reads these exact
files in this exact format.
"""

import json
import math
import os

import osmnx as ox
import pandas as pd

PHASE6_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(PHASE6_DIR, "app")
CITIES_JSON = os.path.join(APP_DIR, "frontend_data", "cities.json")
ROAD_NETWORK_FILE = os.path.join(PHASE6_DIR, "coimbatore_road_network.graphml")
POI_FILE = os.path.join(PHASE6_DIR, "coimbatore_hospitals_police.csv")
OUT_DIR = os.path.join(APP_DIR, "frontend_data", "coimbatore")

# Same formula/constants as the original Coimbatore data generator
# (see README_coimbatore.md for the full justification): a flat citywide
# baseline from NCRB's CII-2017 report, combined with real distance to
# the nearest REAL police station. There is still no trained per-cell
# crime model for Coimbatore -- this script does not change that, it
# only upgrades the road network and POI coordinates from mock/approximate
# to real.
CRIME_BASELINE_RISK = 0.32
MAX_RELEVANT_POLICE_DIST_KM = 3.0
W_CRIME_BASE = 0.6
W_POLICE = 0.4
SAFETY_AVOIDANCE_STRENGTH = 3.0
CELL_SIDE_KM = 0.45  # same grid resolution as the original mock heatmap


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def safety_score_from_police_dist(dist_km):
    police_risk = min(dist_km / MAX_RELEVANT_POLICE_DIST_KM, 1.0)
    risk = W_CRIME_BASE * CRIME_BASELINE_RISK + W_POLICE * police_risk
    return round(100 * (1 - risk), 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Reading Coimbatore's boundary from cities.json...")
    with open(CITIES_JSON) as f:
        cities = json.load(f)
    boundary = cities["coimbatore"]["boundary"]
    lat_min, lat_max = boundary["latMin"], boundary["latMax"]
    lon_min, lon_max = boundary["lonMin"], boundary["lonMax"]

    print("Loading real police/hospital locations from OSM download...")
    pois = pd.read_csv(POI_FILE)
    police_rows = pois[pois["amenity"] == "police"].dropna(subset=["lat", "lon"])
    hospital_rows = pois[pois["amenity"] == "hospital"].dropna(subset=["lat", "lon"])
    police_points = list(zip(police_rows["lat"], police_rows["lon"], police_rows["name"]))
    print(f"  {len(police_points)} real police stations, {len(hospital_rows)} real hospitals")

    if not police_points:
        raise SystemExit(
            "No police stations found in coimbatore_hospitals_police.csv -- "
            "can't compute safety scores without at least one real reference point. "
            "Check that 15_download_road_network_coimbatore.py actually found some."
        )

    def nearest_police_km(lat, lon):
        return min(haversine_km(lat, lon, plat, plon) for plat, plon, _ in police_points)

    print("Loading real road network...")
    G = ox.load_graphml(ROAD_NETWORK_FILE)
    print(f"  {len(G.nodes)} intersections, {len(G.edges)} street segments")

    print("Scoring every road node by real distance to the nearest real police station...")
    node_scores = {}
    for node_id, data in G.nodes(data=True):
        lat, lon = float(data["y"]), float(data["x"])
        dist = nearest_police_km(lat, lon)
        node_scores[node_id] = safety_score_from_police_dist(dist)

    nodes = [
        {"id": int(nid), "lat": round(float(G.nodes[nid]["y"]), 6),
         "lon": round(float(G.nodes[nid]["x"]), 6), "safety_score": score}
        for nid, score in node_scores.items()
    ]

    # De-duplicate parallel edges, same approach as Chennai's
    # 20_export_real_road_graph.py: keep the cheapest edge between any
    # two nodes for routing purposes.
    best_edges = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        length_m = float(data.get("length", 1.0))
        avg_safety = (node_scores[u] + node_scores[v]) / 2
        risk = 1 - (avg_safety / 100)
        safety_weight = length_m * (1 + SAFETY_AVOIDANCE_STRENGTH * risk)
        edge_key = (int(u), int(v))
        if edge_key not in best_edges or safety_weight < best_edges[edge_key][0]:
            best_edges[edge_key] = (safety_weight, length_m)
    edges = [[u, v, sw, length] for (u, v), (sw, length) in best_edges.items()]

    graph = {"nodes": nodes, "edges": edges}
    with open(os.path.join(OUT_DIR, "graph.json"), "w") as f:
        json.dump(graph, f)
    print(f"  Wrote graph.json: {len(nodes)} nodes, {len(edges)} edges")

    # Rebuild the heatmap grid over the SAME boundary, using real police
    # distances instead of the mock list.
    print("Rebuilding the safety heatmap grid with real police distances...")
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians((lat_min + lat_max) / 2))
    n_rows = int((lat_max - lat_min) * km_per_deg_lat / CELL_SIDE_KM)
    n_cols = int((lon_max - lon_min) * km_per_deg_lon / CELL_SIDE_KM)
    lat_step = (lat_max - lat_min) / n_rows
    lon_step = (lon_max - lon_min) / n_cols

    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            lat0, lon0 = lat_min + r * lat_step, lon_min + c * lon_step
            lat1, lon1 = lat0 + lat_step, lon0 + lon_step
            clat, clon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
            dist = nearest_police_km(clat, clon)
            score = safety_score_from_police_dist(dist)
            cells.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[
                    [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]
                ]]},
                "properties": {
                    "cell_id": r * n_cols + c,
                    "safety_score": score,
                    "dist_to_police_km": round(dist, 2),
                    "data_level": "zone-aggregate (no trained crime model for Coimbatore yet)",
                },
            })
    with open(os.path.join(OUT_DIR, "safety_cells.geojson"), "w") as f:
        json.dump({"type": "FeatureCollection", "features": cells}, f)
    print(f"  Wrote safety_cells.geojson: {len(cells)} cells")

    # Rebuild POI files from the REAL OSM download.
    def poi_feature(lat, lon, name):
        score = safety_score_from_police_dist(nearest_police_km(lat, lon))
        return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"name": name if isinstance(name, str) else "Police Station",
                                "safety_score": score}}

    police_geojson = {"type": "FeatureCollection", "features": [
        poi_feature(lat, lon, name) for lat, lon, name in police_points
    ]}
    hospitals_geojson = {"type": "FeatureCollection", "features": [
        poi_feature(row["lat"], row["lon"], row["name"]) for _, row in hospital_rows.iterrows()
    ]}
    with open(os.path.join(OUT_DIR, "police_stations.geojson"), "w") as f:
        json.dump(police_geojson, f)
    with open(os.path.join(OUT_DIR, "hospitals.geojson"), "w") as f:
        json.dump(hospitals_geojson, f)
    print(f"  Wrote police_stations.geojson ({len(police_points)}) and "
          f"hospitals.geojson ({len(hospital_rows)}) from real OSM data")

    print("\nDone. Hard-refresh the app in your browser (Ctrl+Shift+R) to see real Coimbatore streets.")


if __name__ == "__main__":
    main()
