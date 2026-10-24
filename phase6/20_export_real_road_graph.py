"""
20_export_real_road_graph.py
--------------------------------
THE FIX for routes not following real streets: Phase 6's app was
routing on a 4,400-node GRID graph (straight lines between grid cell
centers), not on real streets. Phase 5 already built the real street
network with real routing weights -- this script converts that into
the same lightweight JSON format the browser already knows how to
route on, so it's a drop-in replacement.

Reads: phase5/outputs/road_network_with_safety.graphml (built by
       Phase 5's 16_merge_safety_scores.py -- already has real streets,
       real distances, and safety-weighted costs precomputed)
Writes: phase6/app/frontend_data/graph.json (OVERWRITES the old
        grid-based routing graph -- the safety heatmap in
        safety_cells.geojson is untouched, since that's a separate,
        still-useful grid-based visualization layer)

*** Run this AFTER Phase 5's 16_merge_safety_scores.py. ***
*** After running this, no changes to index.html are needed -- it
    already reads graph.json in this exact format. Just re-run this
    script and hard-refresh your browser. ***
"""

import json
import osmnx as ox

PHASE5_OUTPUTS_DIR = "../phase5/outputs"
OUT_DIR = "app/frontend_data"


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading real road network (with safety weights)...")
    G = ox.load_graphml(f"{PHASE5_OUTPUTS_DIR}/road_network_with_safety.graphml")
    print(f"  {len(G.nodes)} intersections, {len(G.edges)} street segments")

    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            "id": int(node_id),
            "lat": round(float(data["y"]), 6),
            "lon": round(float(data["x"]), 6),
            "safety_score": round(float(data["safety_score"]), 1),
        })

    # de-duplicate: a MultiDiGraph can have more than one edge between the
    # same two intersections (e.g. divided roads) -- keep the cheapest one
    # for routing purposes, same simplification Phase 5's Python routing
    # already effectively does via networkx's shortest_path
    best_edges = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_key = (int(u), int(v))
        length = float(data["length"])
        safety_weight = float(data["safety_weight"])
        if edge_key not in best_edges or safety_weight < best_edges[edge_key][0]:
            best_edges[edge_key] = (safety_weight, length)

    edges = [[u, v, sw, length] for (u, v), (sw, length) in best_edges.items()]

    graph = {"nodes": nodes, "edges": edges}
    out_path = f"{OUT_DIR}/graph.json"
    with open(out_path, "w") as f:
        json.dump(graph, f)

    import os as _os
    size_mb = _os.path.getsize(out_path) / 1e6
    print(f"\nSaved {out_path} ({len(nodes)} nodes, {len(edges)} edges, "
          f"{size_mb:.1f} MB)")
    print("Routes in the app will now follow real streets. "
          "Hard-refresh your browser (Ctrl+Shift+R) to see the change.")


if __name__ == "__main__":
    main()
