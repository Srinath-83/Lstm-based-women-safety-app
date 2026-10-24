"""
17_offline_safe_route.py
---------------------------
The actual "in an emergency, with no internet, get me somewhere safe"
feature. Given a starting point (simulating GPS coordinates from a
phone) and a destination TYPE ("hospital" or "police"), this:

  1. Finds the nearest matching POI (fully offline, from the local
     hospitals_police.csv saved in Phase 5's step 15/15b)
  2. Finds the nearest road-network node to both the start point and
     that POI
  3. Computes TWO routes using Dijkstra's algorithm (via networkx):
       - the safety-weighted route (real streets, prefers safer areas)
       - the true shortest route (real streets, ignores safety)
  4. Saves an interactive folium map showing both routes on real streets

Everything in this script reads only local files -- no internet calls
anywhere. This is the offline behavior your project promises: once
15/15b and 16 have been run once (with internet, to download/prepare
the data), this script works on a phone with airplane mode on.

Run this AFTER 16_merge_safety_scores.py.
"""

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import folium
import config_phase5 as cfg


def load_everything():
    print("Loading safety-weighted road network (offline)...")
    G = ox.load_graphml(cfg.ROAD_NETWORK_WITH_SAFETY_FILE)
    # osmnx saves edge attributes as strings in graphml; cast the ones we need
    for u, v, k, data in G.edges(keys=True, data=True):
        data["length"] = float(data["length"])
        data["safety_weight"] = float(data["safety_weight"])
    for n, data in G.nodes(data=True):
        data["y"] = float(data["y"])
        data["x"] = float(data["x"])
        data["safety_score"] = float(data["safety_score"])

    pois = pd.read_csv(f"{cfg.PHASE5_OUTPUTS_DIR}/hospitals_police.csv"
                        if False else cfg.POI_FILE)
    print(f"  Road network: {len(G.nodes)} nodes, {len(G.edges)} edges")
    print(f"  POIs available: {len(pois)} ({pois['amenity'].value_counts().to_dict()})")
    return G, pois


def nearest_node(G, lat, lon):
    """Find the road-network node closest to a raw lat/lon (e.g. a GPS fix)."""
    return ox.distance.nearest_nodes(G, X=lon, Y=lat)


def nearest_poi(pois, lat, lon, amenity_type):
    subset = pois[pois["amenity"] == amenity_type].copy()
    if len(subset) == 0:
        raise ValueError(f"No POIs of type '{amenity_type}' found.")
    # simple Euclidean approx in degrees -- fine at city scale for "nearest" ranking
    subset["dist"] = np.sqrt((subset["lat"] - lat) ** 2 + (subset["lon"] - lon) ** 2)
    nearest = subset.loc[subset["dist"].idxmin()]
    return nearest


def route_stats(G, path, weight_attr):
    total_length_m = sum(
        G[path[i]][path[i + 1]][0]["length"] for i in range(len(path) - 1)
    )
    avg_safety = np.mean([G.nodes[n]["safety_score"] for n in path])
    min_safety = np.min([G.nodes[n]["safety_score"] for n in path])
    return {
        "n_nodes": len(path),
        "total_distance_m": round(total_length_m, 1),
        "avg_safety_score": round(float(avg_safety), 1),
        "min_safety_score": round(float(min_safety), 1),
    }


def find_routes(G, start_lat, start_lon, destination_type, pois):
    poi = nearest_poi(pois, start_lat, start_lon, destination_type)
    print(f"\nNearest {destination_type}: {poi['name']} "
          f"at ({poi['lat']:.5f}, {poi['lon']:.5f})")

    start_node = nearest_node(G, start_lat, start_lon)
    end_node = nearest_node(G, poi["lat"], poi["lon"])

    safe_path = nx.shortest_path(G, start_node, end_node, weight="safety_weight")
    short_path = nx.shortest_path(G, start_node, end_node, weight="length")

    safe_stats = route_stats(G, safe_path, "safety_weight")
    short_stats = route_stats(G, short_path, "length")

    print(f"Safety-weighted route: {safe_stats}")
    print(f"Shortest route:        {short_stats}")

    return safe_path, short_path, poi, safe_stats, short_stats


def visualize_on_map(G, safe_path, short_path, start_lat, start_lon, poi, out_path):
    m = folium.Map(location=[start_lat, start_lon], zoom_start=15,
                    tiles="cartodbpositron")

    def path_to_latlon(path):
        return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]

    folium.PolyLine(
        path_to_latlon(short_path), color="black", weight=4, opacity=0.7,
        dash_array="8", tooltip="Shortest route (ignores safety)"
    ).add_to(m)

    folium.PolyLine(
        path_to_latlon(safe_path), color="blue", weight=5, opacity=0.9,
        tooltip="Safety-weighted route"
    ).add_to(m)

    folium.Marker(
        [start_lat, start_lon], icon=folium.Icon(color="black", icon="user", prefix="fa"),
        popup="Start (current location)"
    ).add_to(m)

    folium.Marker(
        [poi["lat"], poi["lon"]], icon=folium.Icon(color="blue", icon="shield", prefix="fa"),
        popup=f"{poi['name']} ({poi['amenity']})"
    ).add_to(m)

    m.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    G, pois = load_everything()

    # --- Demo: simulate a "current location" in a genuinely RISKY area,
    # so the safety-weighted route actually has something worth avoiding
    # (a purely random start/end pair is often too short/direct for the
    # two routes to meaningfully diverge)
    np.random.seed(cfg.RANDOM_SEED)
    all_nodes = list(G.nodes(data=True))
    sorted_by_safety = sorted(all_nodes, key=lambda n: n[1]["safety_score"])
    riskiest_nodes = sorted_by_safety[:int(len(all_nodes) * 0.05)]  # riskiest 5%
    sample_node = riskiest_nodes[np.random.randint(len(riskiest_nodes))]
    start_lat, start_lon = sample_node[1]["y"], sample_node[1]["x"]
    print(f"Simulated current location (deliberately picked from a "
          f"lower-safety area for a more illustrative demo): "
          f"({start_lat:.5f}, {start_lon:.5f}), "
          f"safety_score={sample_node[1]['safety_score']:.1f}")

    safe_path, short_path, poi, safe_stats, short_stats = find_routes(
        G, start_lat, start_lon, destination_type="police", pois=pois
    )

    visualize_on_map(
        G, safe_path, short_path, start_lat, start_lon, poi,
        f"{cfg.PHASE5_OUTPUTS_DIR}/offline_route_map.html"
    )

    summary = pd.DataFrame([
        {"route": "safety_weighted", **safe_stats},
        {"route": "shortest", **short_stats},
    ])
    summary.to_csv(f"{cfg.PHASE5_OUTPUTS_DIR}/route_summary.csv", index=False)
    print(f"Saved {cfg.PHASE5_OUTPUTS_DIR}/route_summary.csv")
