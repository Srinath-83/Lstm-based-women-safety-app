# Phase 4 — Women's Safety Score + Safe Routing

## What this does

Turns your ConvLSTM's raw crime prediction into your project's actual
differentiator: a Safety Score per grid cell, and a route-finder that
prefers safer paths over strictly shortest ones.

### `config_phase4.py`
Settings: which trained model to use, the safety score formula's
weights, and how strongly routing avoids unsafe cells.

### `12_generate_safety_map.py`
This is the clever bit: your ConvLSTM was trained on 16×16 crops, but
it's a **fully convolutional** network — every operation inside it
(the convolutions in each gate, the classifier head) just slides a
small kernel across whatever image size you give it. So we feed it
the **entire city grid** (100×44) in one shot and get a full citywide
crime-probability map, with no retraining needed.

Then it combines that with two "guardianship" factors (from Routine
Activity Theory, the same theory the base paper cites):
```
risk = W_CRIME  * predicted_crime_probability
     + W_LIGHT  * (1 - streetlight_score)
     + W_POLICE * normalized_distance_to_police

safety_score = 100 * (1 - risk)     # 0 = risky, 100 = safe
```

→ Outputs `outputs/safety_scores.csv` (per-cell, with lat/lon),
`safety_score_grid.npy`, `crime_probability_grid.npy`, `risk_grid.npy`

### `13_visualize_safety_map.py`
- A static matplotlib heatmap (good for your report)
- An interactive folium map (`safety_map_interactive.html`) — open it
  in a browser, it's the same style of map you already set up in
  Phase 1

### `14_safe_routing.py`
Builds a graph where each grid cell connects to its 8 neighbors, then
uses Dijkstra's algorithm (via `networkx`) to find:
- The **safety-weighted route** (prefers safer cells, even if longer)
- The **true shortest route** (pure distance, ignores safety) — as a
  fair baseline to compare against

→ Outputs `route_comparison.png` (visual) and `route_comparison.csv`

## How to run

```bash
cd phase4
pip install torch pandas matplotlib networkx folium
python3 12_generate_safety_map.py
python3 13_visualize_safety_map.py
python3 14_safe_routing.py
```

Needs `phase1/outputs/`, `phase2/outputs/`, and `phase3/outputs/`
(specifically the trained `convlstm_model.pt`) all present as sibling
folders — same structure as before.

## ⚠️ Important limitations to state honestly in your report

1. **The safety score weights are assumptions, not fitted from real
   data.** `W_CRIME=0.5, W_LIGHT=0.3, W_POLICE=0.2` are a reasonable
   starting point grounded in the theory the base paper cites, but
   they're not calibrated against real incident reports. Say this
   plainly rather than presenting the score as empirically validated.

2. **This routing is grid-based, not road-based.** Right now, a "path"
   is a sequence of grid cells connected in straight lines — it doesn't
   know where actual streets are. This is a good proof-of-concept for
   the *idea* (prefer safer areas), but Phase 5 replaces it with real
   OpenStreetMap road data so routes actually follow streets.

3. **The current crime probability map looks fairly flat/uniform**
   (compare `safety_map_static.png`'s left panel) — this is because
   we're still using the quick-test ConvLSTM (limited data, few
   epochs, from Phase 3). Once you retrain ConvLSTM at full scale
   (see Phase 3's README), rerun `12_generate_safety_map.py` and you
   should see much sharper hotspots, since the model will have
   actually learned area-type/time-of-day risk patterns instead of
   still being mostly uninformed.

## Things worth experimenting with (good "extra analysis" for your report)

- **`SAFETY_AVOIDANCE_STRENGTH`** in `config_phase4.py` — try setting
  it much higher (e.g., 10 or 20) and rerun `14_safe_routing.py`. You
  should see the safety-weighted route take a visibly bigger detour to
  avoid unsafe cells — a good way to demonstrate the trade-off your
  system makes, and to justify a specific value in your final report.
- **`W_CRIME`, `W_LIGHT`, `W_POLICE`** — try shifting more weight onto
  `W_LIGHT` (streetlights) vs `W_CRIME` and see how differently the
  safety map looks. This is a good sensitivity-analysis figure for
  your report/viva.

## Next: Phase 5

Phase 5 replaces this grid-based routing with real road-network
routing (OpenStreetMap + a routing engine like GraphHopper), so a
"safe route" actually follows real streets — and, per your original
requirement, works **offline** once the map data and safety scores are
downloaded to the device.
