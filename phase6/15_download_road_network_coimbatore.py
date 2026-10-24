"""
15_download_road_network_coimbatore.py
---------------------------------------
Downloads Coimbatore's REAL walking street network (nodes = intersections,
edges = actual streets, with real distances) plus real hospital and police
station locations, using OpenStreetMap data via the `osmnx` library --
the exact same approach as phase5/15_download_road_network.py for Chennai.

*** THIS SCRIPT NEEDS INTERNET AND MUST BE RUN ON YOUR OWN MACHINE. ***
(It queries OpenStreetMap's Overpass API, which isn't reachable from
inside the sandboxed environment this project was built in -- that's
expected, not a bug.)

Run it once. It writes:
  coimbatore_road_network.graphml
  coimbatore_hospitals_police.csv

After that, adapt phase5/16_merge_safety_scores.py and
phase6/20_export_real_road_graph.py (swap the Chennai bbox/config for the
Coimbatore one below) to turn this into a new, more accurate
frontend_data/coimbatore/graph.json -- replacing the mock grid graph that
ships with this phase.
"""

import osmnx as ox
import pandas as pd

# Coimbatore Corporation's approximate extent (see README_coimbatore.md
# for why this is an approximate rectangle, not an official boundary).
LAT_MIN, LAT_MAX = 10.95, 11.07
LON_MIN, LON_MAX = 76.89, 77.05
NETWORK_TYPE = "walk"

ROAD_NETWORK_FILE = "coimbatore_road_network.graphml"
POI_FILE = "coimbatore_hospitals_police.csv"


def download_road_network():
    print(f"Downloading '{NETWORK_TYPE}' street network for Coimbatore, "
          f"bbox lat=[{LAT_MIN}, {LAT_MAX}], lon=[{LON_MIN}, {LON_MAX}] ...")
    bbox = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)  # (west, south, east, north)
    G = ox.graph_from_bbox(bbox=bbox, network_type=NETWORK_TYPE)
    print(f"Downloaded network: {len(G.nodes)} intersections, {len(G.edges)} street segments")
    ox.save_graphml(G, ROAD_NETWORK_FILE)
    print(f"Saved to {ROAD_NETWORK_FILE}")
    return G


def download_hospitals_and_police():
    print("\nDownloading hospital and police station locations...")
    bbox = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    tags = {"amenity": ["hospital", "police"]}
    pois = ox.features_from_bbox(bbox=bbox, tags=tags)
    print(f"Found {len(pois)} hospitals/police stations")

    pois = pois.reset_index()

    # Reproject to a local metre-based CRS before taking the centroid --
    # a centroid computed directly in lat/lon degrees distorts irregular
    # building-outline polygons and can land outside the shape (e.g. just
    # offshore for a waterfront campus). Simple point-mapped POIs are
    # unaffected (a point's centroid is itself).
    utm_crs = pois.estimate_utm_crs()
    centroids = pois.geometry.to_crs(utm_crs).centroid.to_crs(pois.crs)
    pois["lat"] = centroids.y
    pois["lon"] = centroids.x
    pois_clean = pois[["amenity", "name", "lat", "lon"]].dropna(subset=["lat", "lon"])

    pois_clean.to_csv(POI_FILE, index=False)
    print(f"Saved to {POI_FILE}")
    return pois_clean


if __name__ == "__main__":
    G = download_road_network()
    pois = download_hospitals_and_police()
    print("\n--- Done ---")
    print("Next: adapt phase5/16_merge_safety_scores.py and "
          "phase6/20_export_real_road_graph.py for Coimbatore's bbox/files "
          "to regenerate frontend_data/coimbatore/graph.json from real streets.")