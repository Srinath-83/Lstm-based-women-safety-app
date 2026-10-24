"""
15_download_road_network.py
------------------------------
Downloads Chennai's REAL walking street network (nodes = intersections,
edges = actual streets, with real distances) plus real hospital and
police station locations, using OpenStreetMap data via the `osmnx`
library.

NOTE ON HOW THIS WORKS: you do NOT need to manually download any .osm
or .pbf file yourself. osmnx does this live over the internet: it looks
up the requested area, queries OpenStreetMap's Overpass API for the
street data inside it, and hands it back to you as a ready-to-use graph
-- this script then saves that graph to a local file so every later
script can use it OFFLINE.

We download by BOUNDING BOX (not by place name like "Chennai, India")
so the road network's extent exactly matches Phase 1's analysis grid --
this means every road intersection is guaranteed to fall inside a grid
cell with a real safety score, instead of some falling outside a
mismatched city-boundary polygon.

*** THIS SCRIPT NEEDS INTERNET AND MUST BE RUN ON YOUR OWN MACHINE. ***
(It queries OpenStreetMap's Overpass API, which isn't reachable from
inside this sandboxed environment -- that's expected, not a bug.)

Run it ONCE. After that, everything is saved locally, and every other
script in Phase 5 (16 and 17) works completely OFFLINE by reading these
saved files -- no internet needed again. This is the same idea real
offline-maps apps use: download the city once, route locally forever.
"""

import osmnx as ox
import pandas as pd
import config_phase5 as cfg
import config as cfg1   # Phase 1's config -- gives us the exact same bounding box


def download_road_network():
    print(f"Downloading '{cfg.NETWORK_TYPE}' street network for the bounding "
          f"box lat=[{cfg1.LAT_MIN}, {cfg1.LAT_MAX}], "
          f"lon=[{cfg1.LON_MIN}, {cfg1.LON_MAX}] ... "
          f"(this can take a minute or two -- Chennai is a big city)")

    # osmnx wants bbox as (left, bottom, right, top) = (west, south, east, north)
    bbox = (cfg1.LON_MIN, cfg1.LAT_MIN, cfg1.LON_MAX, cfg1.LAT_MAX)
    G = ox.graph_from_bbox(bbox=bbox, network_type=cfg.NETWORK_TYPE)
    print(f"Downloaded network: {len(G.nodes)} intersections, "
          f"{len(G.edges)} street segments")

    ox.save_graphml(G, cfg.ROAD_NETWORK_FILE)
    print(f"Saved to {cfg.ROAD_NETWORK_FILE} "
          f"-- from now on this loads offline, no internet needed")
    return G


def download_hospitals_and_police():
    print("\nDownloading hospital and police station locations...")
    bbox = (cfg1.LON_MIN, cfg1.LAT_MIN, cfg1.LON_MAX, cfg1.LAT_MAX)
    tags = {"amenity": ["hospital", "police"]}
    pois = ox.features_from_bbox(bbox=bbox, tags=tags)
    print(f"Found {len(pois)} hospitals/police stations")

    # keep just what we need: name, amenity type, and a representative point
    pois = pois.reset_index()

    # THE FIX for markers landing in the sea: many hospitals/police in OSM
    # are mapped as building/ground OUTLINES (polygons), not single points.
    # Taking a "centroid" directly in latitude/longitude DEGREES distorts
    # irregular shapes -- for a non-convex waterfront campus outline, the
    # degree-based centroid can fall outside the shape entirely, sometimes
    # just offshore. (This is exactly what osmnx's own CRS warning flags.)
    # The correct way: reproject to a local metre-based CRS, take the
    # centroid there (geometrically accurate), then reproject back to
    # lat/lon. For POIs that are already simple points, this is a no-op --
    # a point's centroid is itself, so real point-mapped hospitals/police
    # are unaffected.
    utm_crs = pois.estimate_utm_crs()
    centroids = pois.geometry.to_crs(utm_crs).centroid.to_crs(pois.crs)
    pois["lat"] = centroids.y
    pois["lon"] = centroids.x
    pois_clean = pois[["amenity", "name", "lat", "lon"]].dropna(subset=["lat", "lon"])

    pois_clean.to_csv(cfg.POI_FILE, index=False)
    print(f"Saved to {cfg.POI_FILE}")
    return pois_clean


if __name__ == "__main__":
    G = download_road_network()
    pois = download_hospitals_and_police()

    print("\n--- Done ---")
    print("You can now run 16_merge_safety_scores.py and "
          "17_offline_safe_route.py fully offline.")