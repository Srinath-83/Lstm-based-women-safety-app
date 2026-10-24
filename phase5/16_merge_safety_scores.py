"""
16_merge_safety_scores.py
----------------------------
Takes the road network (from 15_download_road_network.py, or the mock
version from 15b) and attaches a safety score to every intersection, by
figuring out which of Phase 4's grid cells each intersection falls
inside (using the exact same grid math as Phase 1, so it lines up
perfectly).

Then computes a SAFETY-WEIGHTED length for every street segment:
    safety_weight = real_length_in_meters * (1 + STRENGTH * risk)

...so a routing algorithm minimizing safety_weight will naturally
prefer streets that pass through safer areas, while still fundamentally
respecting real distances (a street through a very safe area is barely
penalized; a street through a very unsafe area costs much more to use).

This whole script runs OFFLINE -- it only reads local files.

Run this AFTER 15 (or 15b) and after Phase 4 has produced
safety_scores.csv.
"""

import numpy as np
import pandas as pd
import osmnx as ox
import config_phase5 as cfg
import config as cfg1   # Phase 1's config -- gives us the grid math


def load_safety_scores():
    df = pd.read_csv(f"{cfg.PHASE4_OUTPUTS_DIR}/safety_scores.csv")
    return df


def latlon_to_cell_id(lat, lon, n_cols):
    """
    Reproduces Phase 1's exact grid indexing so road-network nodes map
    to the SAME cell IDs used in safety_scores.csv.
    """
    row = np.floor((lat - cfg1.LAT_MIN) / cfg1.CELL_SIZE_DEG_LAT).astype(int)
    col = np.floor((lon - cfg1.LON_MIN) / cfg1.CELL_SIZE_DEG_LON).astype(int)
    return row, col


def attach_safety_to_nodes(G, safety_df):
    n_rows = safety_df["row"].max() + 1
    n_cols = safety_df["col"].max() + 1

    # fast lookup: (row, col) -> safety_score
    score_grid = np.full((n_rows, n_cols), safety_df["safety_score"].mean())
    for _, r in safety_df.iterrows():
        score_grid[int(r["row"]), int(r["col"])] = r["safety_score"]

    n_outside = 0
    for node_id, data in G.nodes(data=True):
        lat, lon = data["y"], data["x"]
        row, col = latlon_to_cell_id(lat, lon, n_cols)
        if 0 <= row < n_rows and 0 <= col < n_cols:
            data["safety_score"] = float(score_grid[row, col])
        else:
            # node falls outside our analysis grid's bounding box --
            # fall back to the city-wide average so routing still works
            data["safety_score"] = float(safety_df["safety_score"].mean())
            n_outside += 1

    if n_outside > 0:
        print(f"  Note: {n_outside} nodes fell outside the analysis grid "
              f"and were given the city average safety score.")


def compute_safety_weighted_edges(G):
    for u, v, key, data in G.edges(keys=True, data=True):
        length_m = data.get("length", 1.0)
        safety_u = G.nodes[u]["safety_score"]
        safety_v = G.nodes[v]["safety_score"]
        avg_safety = (safety_u + safety_v) / 2
        risk = 1 - (avg_safety / 100)

        safety_weight = length_m * (1 + cfg.SAFETY_AVOIDANCE_STRENGTH * risk)
        data["safety_weight"] = safety_weight


def main():
    print("Loading road network (offline, from local file)...")
    G = ox.load_graphml(cfg.ROAD_NETWORK_FILE)
    print(f"  {len(G.nodes)} nodes, {len(G.edges)} edges")

    print("Loading Phase 4 safety scores...")
    safety_df = load_safety_scores()
    print(f"  {len(safety_df)} grid cells")

    print("Attaching safety scores to road-network nodes...")
    attach_safety_to_nodes(G, safety_df)

    print("Computing safety-weighted edge costs...")
    compute_safety_weighted_edges(G)

    ox.save_graphml(G, cfg.ROAD_NETWORK_WITH_SAFETY_FILE)
    print(f"\nSaved {cfg.ROAD_NETWORK_WITH_SAFETY_FILE}")

    scores = [d["safety_score"] for _, d in G.nodes(data=True)]
    print(f"\nNode safety score summary: min={min(scores):.1f}, "
          f"max={max(scores):.1f}, mean={np.mean(scores):.1f}")


if __name__ == "__main__":
    main()
