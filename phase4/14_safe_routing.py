"""
14_safe_routing.py
---------------------
Builds a routing graph over the grid (each cell connects to its
neighbors), where the "cost" of moving through a cell depends on its
safety score -- then uses Dijkstra's algorithm (via networkx) to find
the SAFEST path between two points, and compares it against the plain
SHORTEST path.

Beginner note: this is deliberately a simplified proof-of-concept using
straight-line grid-cell connections, NOT real roads. Phase 5 will replace
this with an actual road network (via OpenStreetMap + GraphHopper) so
routes follow real streets -- but the core idea ("prefer safer cells,
even if the path is a bit longer") is identical, and this is much
easier to understand, debug, and visualize first.

Run this AFTER 12_generate_safety_map.py.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import config_phase4 as cfg


def build_grid_graph(scores_df, n_rows, n_cols):
    """
    Each cell is a node. Connect each cell to its up/down/left/right/
    diagonal neighbors (8-connectivity, same idea as the paper's
    "neighboring cells" used in their NN metric).

    Edge weight = a combination of physical distance (1 for straight
    moves, sqrt(2) for diagonal moves -- like a chessboard) and a
    safety PENALTY based on how risky the destination cell is. Higher
    SAFETY_AVOIDANCE_STRENGTH makes the router avoid unsafe cells more
    aggressively, even if it means a longer path.
    """
    G = nx.Graph()
    score_lookup = scores_df.set_index("cell_id")["safety_score"].to_dict()
    row_lookup = scores_df.set_index("cell_id")["row"].to_dict()
    col_lookup = scores_df.set_index("cell_id")["col"].to_dict()
    cell_id_grid = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)

    for r in range(n_rows):
        for c in range(n_cols):
            cell_id = int(cell_id_grid[r, c])
            G.add_node(cell_id, row=r, col=c, safety_score=score_lookup[cell_id])

    for r in range(n_rows):
        for c in range(n_cols):
            cell_id = int(cell_id_grid[r, c])
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    neighbor_id = int(cell_id_grid[rr, cc])
                    if G.has_edge(cell_id, neighbor_id):
                        continue

                    physical_dist = np.sqrt(dr ** 2 + dc ** 2)  # 1 or sqrt(2)

                    # risk penalty for walking THROUGH the destination
                    # cell: low safety score -> high penalty
                    dest_risk = 1 - (score_lookup[neighbor_id] / 100)
                    safety_penalty = cfg.SAFETY_AVOIDANCE_STRENGTH * dest_risk

                    weight = physical_dist + safety_penalty
                    G.add_edge(cell_id, neighbor_id,
                               weight=weight, physical_dist=physical_dist)

    return G


def find_nearest_police_station(scores_df, from_cell_id):
    police = scores_df[scores_df["is_police_station"] == True]
    from_row = scores_df.loc[scores_df["cell_id"] == from_cell_id, "row"].values[0]
    from_col = scores_df.loc[scores_df["cell_id"] == from_cell_id, "col"].values[0]

    dists = np.sqrt((police["row"] - from_row) ** 2 + (police["col"] - from_col) ** 2)
    nearest_idx = dists.idxmin()
    return int(police.loc[nearest_idx, "cell_id"])


def get_route(G, source, target, weight_key):
    """
    weight_key options:
      "weight"        -> safety-weighted route (physical distance + safety penalty)
      "physical_dist" -> TRUE shortest route (pure distance, ignores safety)
      None            -> fewest hops (not physically meaningful on a grid
                          with diagonal moves -- not used here, kept only
                          for reference)
    """
    path = nx.shortest_path(G, source=source, target=target, weight=weight_key)
    return path


def route_stats(G, path):
    total_physical_dist = sum(
        G[path[i]][path[i + 1]]["physical_dist"] for i in range(len(path) - 1)
    )
    avg_safety = np.mean([G.nodes[n]["safety_score"] for n in path])
    min_safety = np.min([G.nodes[n]["safety_score"] for n in path])
    return {
        "n_cells": len(path),
        "total_physical_distance_cells": round(total_physical_dist, 2),
        "avg_safety_score": round(avg_safety, 1),
        "min_safety_score": round(min_safety, 1),
    }


def visualize_routes(safety_score_grid, G, safe_path, short_path, source, target,
                      n_rows, n_cols, out_path):
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(safety_score_grid, cmap="RdYlGn", origin="lower", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="Safety Score")

    def path_to_coords(path):
        rows = [G.nodes[n]["row"] for n in path]
        cols = [G.nodes[n]["col"] for n in path]
        return cols, rows  # x=col, y=row for imshow

    sx, sy = path_to_coords(short_path)
    ax.plot(sx, sy, color="black", linewidth=2, linestyle="--",
            label="Shortest path (ignores safety)", alpha=0.8)

    fx, fy = path_to_coords(safe_path)
    ax.plot(fx, fy, color="blue", linewidth=2.5,
            label="Safety-weighted path", alpha=0.9)

    src_r, src_c = G.nodes[source]["row"], G.nodes[source]["col"]
    tgt_r, tgt_c = G.nodes[target]["row"], G.nodes[target]["col"]
    ax.scatter([src_c], [src_r], color="black", marker="o", s=150,
               zorder=5, label="Start", edgecolors="white")
    ax.scatter([tgt_c], [tgt_r], color="blue", marker="*", s=250,
               zorder=5, label="Destination (police station)", edgecolors="white")

    ax.set_title("Safety-Weighted Route vs. Shortest Route")
    ax.legend(loc="upper right")
    ax.set_xlabel("Grid column (West -> East)")
    ax.set_ylabel("Grid row (South -> North)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    scores_df = pd.read_csv(f"{cfg.PHASE4_OUTPUTS_DIR}/safety_scores.csv")
    safety_score_grid = np.load(f"{cfg.PHASE4_OUTPUTS_DIR}/safety_score_grid.npy")
    n_rows, n_cols = safety_score_grid.shape

    print("Building routing graph...")
    G = build_grid_graph(scores_df, n_rows, n_cols)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # --- Demo: pick a random starting cell, route to nearest police station ---
    np.random.seed(cfg.RANDOM_SEED)
    # bias toward picking a "risky-ish" starting cell so the safety-weighted
    # route actually has something interesting to avoid
    risky_cells = scores_df.nsmallest(200, "safety_score")
    source = int(risky_cells.sample(1, random_state=cfg.RANDOM_SEED)["cell_id"].values[0])
    target = find_nearest_police_station(scores_df, source)

    print(f"\nRouting from cell {source} to nearest police station (cell {target})")

    safe_path = get_route(G, source, target, weight_key="weight")
    short_path = get_route(G, source, target, weight_key="physical_dist")

    safe_stats = route_stats(G, safe_path)
    short_stats = route_stats(G, short_path)

    print(f"\nSafety-weighted route: {safe_stats}")
    print(f"Shortest (unweighted) route: {short_stats}")

    visualize_routes(
        safety_score_grid, G, safe_path, short_path, source, target,
        n_rows, n_cols, f"{cfg.PHASE4_OUTPUTS_DIR}/route_comparison.png"
    )

    # save both routes as CSVs (cell-by-cell, with lat/lon for later mapping)
    def path_to_df(path, label):
        rows = []
        for cell_id in path:
            cell_row = scores_df[scores_df["cell_id"] == cell_id].iloc[0]
            rows.append({
                "route": label, "cell_id": cell_id,
                "center_lat": cell_row["center_lat"], "center_lon": cell_row["center_lon"],
                "safety_score": cell_row["safety_score"],
            })
        return pd.DataFrame(rows)

    routes_df = pd.concat([
        path_to_df(safe_path, "safety_weighted"),
        path_to_df(short_path, "shortest"),
    ])
    routes_df.to_csv(f"{cfg.PHASE4_OUTPUTS_DIR}/route_comparison.csv", index=False)
    print(f"Saved {cfg.PHASE4_OUTPUTS_DIR}/route_comparison.csv")


if __name__ == "__main__":
    main()
