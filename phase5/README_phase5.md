# Phase 5 — Real Road Routing, Working Offline

## What this does

Phase 4's routing used straight lines between grid cells. Phase 5 replaces
that with **real streets** (via OpenStreetMap), and makes the whole thing
work **fully offline** after a one-time download — this is the feature
you originally asked about (routing to a hospital/police station with no
internet).

## The core idea

1. **Download once** (needs internet): the real street network for
   Chennai, plus real hospital/police station locations.
2. **Save it locally** as files on disk.
3. **From then on, every route calculation reads only those local
   files** — no internet calls, ever. This is exactly how offline map
   apps (Google Maps offline mode, Maps.me, OsmAnd) work.

### Files

| File | What it does | Needs internet? |
|---|---|---|
| `config_phase5.py` | Settings (city, network type, file paths) | — |
| `15_download_road_network.py` | Downloads real Chennai streets + hospitals/police via OpenStreetMap (`osmnx` library) | **Yes — run this on your own machine** |
| `15b_generate_mock_road_network.py` | Fallback: generates a synthetic Manhattan-style grid instead, for testing without internet | No |
| `16_merge_safety_scores.py` | Attaches Phase 4's safety scores to every road intersection | No (offline) |
| `17_offline_safe_route.py` | Finds safety-weighted vs. shortest routes to the nearest hospital/police station, on real streets | No (offline) |

## ⚠️ Important: script 15 needs to run on YOUR machine

I tested this whole pipeline using the mock generator (`15b`), because
this sandboxed environment can't reach OpenStreetMap's servers (network
restrictions on my end, not a bug in the code). **You need to run `15`
yourself**, with normal internet access, before your final results.

```bash
cd phase5
pip install osmnx networkx folium pandas numpy
python3 15_download_road_network.py      # needs internet, run once
python3 16_merge_safety_scores.py        # offline from here on
python3 17_offline_safe_route.py         # offline
```

If you don't have internet handy right now and just want to test/develop
the rest of the pipeline, run `15b_generate_mock_road_network.py` instead
of `15` — it produces files in the exact same format, so `16` and `17`
work identically either way. Just remember to switch to the real `15`
before generating your final report figures (mock streets are a perfect
grid; real streets aren't, and your report should show the real thing).

## How the safety-score merging works

Phase 4 gave you a safety score per **grid cell** (0.2 sq km squares).
Phase 5's road network has intersections at exact lat/lon points. So
`16_merge_safety_scores.py`:
1. For every road intersection, figures out which grid cell it falls
   inside (using the *exact same* grid math as Phase 1 — so cell
   boundaries line up perfectly, no approximation)
2. Assigns that cell's safety score to the intersection
3. For every street segment, computes a `safety_weight` = real distance
   × (1 + penalty based on how unsafe the segment's endpoints are)

Routing with `weight="safety_weight"` then naturally prefers streets
through safer areas, while `weight="length"` gives the ignore-safety
baseline for comparison — same idea as Phase 4, just on real streets now.

## What "offline" really means here

`17_offline_safe_route.py` never makes a network request. Everything it
needs — the road graph, safety scores, hospital/police locations — was
already saved to disk by scripts 15/15b and 16. This is genuinely how
you'd deploy this on a phone: download the city's data once (e.g., over
WiFi before a trip), and the emergency routing feature keeps working even
in a dead zone or with data turned off.

**For a real Android app** (beyond this Python prototype), the
equivalent production tools are:
- **GraphHopper** or **Valhalla** instead of `networkx` — much faster
  routing engines with proper mobile SDKs, same core idea (load a local
  graph, route offline)
- The `.graphml` files here would become GraphHopper's own optimized
  offline graph format instead

This Python version proves the *concept* end-to-end (which is what
matters for your project/report) — swapping the routing engine for a
production one is an implementation detail, not a design change.

## What to check once you run the real download (script 15)

- `outputs/chennai_road_network.graphml` should be noticeably larger and
  messier-looking than the mock version if you inspect it — real Chennai
  street data, not a perfect grid
- `outputs/hospitals_police.csv` should show real hospital/police names
  from OpenStreetMap, not "Mock Hospital 1", "Mock Police Station 1" etc.
- Re-run `16` and `17` after the real download — don't mix mock and real
  files

## Next: Phase 6

Phase 6 is the frontend — wiring the safety map, safe routing, and an
SOS button into an actual app interface (Leaflet map + basic UI), so
this stops being a collection of scripts and starts looking like the
product you're presenting.
