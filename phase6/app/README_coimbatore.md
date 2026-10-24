# Coimbatore city data — sources, methodology, and limitations

Phase 3 adds Coimbatore as a second selectable city, using the exact same
file shapes Chennai's app already consumes (`safety_cells.geojson`,
`graph.json`, `police_stations.geojson`, `hospitals.geojson`), now under
`frontend_data/coimbatore/`. `frontend_data/cities.json` tells `index.html`
where each city's files live and what its default map center/boundary is.

## 1. Boundary used

An approximate rectangle around Coimbatore Corporation's extent:
`lat 10.95–11.07, lon 76.89–77.05` (~13 km × ~18 km), centered on
Coimbatore's published coordinates (11°00′N, 76°58′E). This is **not** an
official administrative boundary polygon — it's a practical analysis box,
the same approach Chennai's Phase 1 bbox took.

## 2. Police stations and hospitals — REAL, named, verified institutions

**Police stations** (18, all real): Coimbatore City Police's own four ACP
divisions (West, East, Central, South) and their published station list —
e.g. R.S. Puram, Ukkadam, Gandhipuram All Women PS, Race Course, Singanallur,
Podanur, and others.
Source: [Coimbatore City Police — Wikipedia](https://en.wikipedia.org/wiki/Coimbatore_City_Police).

**Hospitals** (5, all real, well-known public/major hospitals): Coimbatore
Medical College Hospital (CMCH), Kovai Medical Center and Hospital (KMCH),
PSG Hospitals, Ganga Hospital, Government ESI Hospital Singanallur.

**Important honesty caveat:** the coordinates attached to these real,
named institutions are **approximate locality-center estimates**, not
precision rooftop geocodes (no mapping/geocoding API was available in this
environment). No incident, institution, or location was invented — the
institutions themselves and their general areas are real and verifiable —
but before relying on this for anything beyond a class project/demo,
replace these coordinates with precise geocoded coordinates (e.g. from
Google Maps or OpenStreetMap's Nominatim).

## 3. Crime/safety data — what it is and, importantly, what it is NOT

There is **no trained crime-prediction model for Coimbatore** (that would
require a real Coimbatore incident dataset and a retrain — explicitly out
of scope for this phase, same as the instructions for Phase 1 and 2).

So `safety_score` for Coimbatore is **not** a per-cell ML prediction like
Chennai's ConvLSTM output. It is a transparent, documented formula:

```
risk = 0.6 * CRIME_BASELINE_RISK + 0.4 * police_distance_risk
safety_score = 100 * (1 - risk)
```

- `CRIME_BASELINE_RISK = 0.32` — a flat, citywide constant. It is set
  lower than a generic big-city default because NCRB's **Crime in India
  2017** report placed Coimbatore among the **lowest** of the 19 large
  Indian cities it compared on violent crime (7.9 per lakh population)
  and crimes against women (7.1 per lakh women).
  Source: NCRB CII-2017, as summarized by
  [Factly: "What is the status of crime in 19 of the large Indian cities"](https://factly.in/what-is-the-status-of-crime-in-19-of-the-large-indian-cities).
  **This is a single, coarse, city-level statistic from 2017 — not a
  recent, fine-grained, or official Tamil Nadu Police dataset.** No
  ward-level, police-station-level, or year-by-year Coimbatore crime
  breakdown was found to be freely available in this session; data.gov.in
  and the Tamil Nadu Police open-data pages were not reachable from this
  sandboxed environment (no internet access to third-party sites besides
  package registries), so this could not be directly queried here.
- `police_distance_risk` — the one genuinely spatial, real signal: each
  grid cell's real haversine distance to the nearest of the 18 real
  police stations above, scaled to [0,1] over a 3 km reference distance.

**No individual crime incidents, coordinates, or counts were invented
anywhere.** Every cell's score is explicitly labeled in its GeoJSON
properties as `"data_level": "zone-aggregate (no trained crime model for
Coimbatore yet)"` so this is never confused with Chennai's model-based
scores.

### What to do before using this for a real report/viva claim about Coimbatore
Get the actual Tamil Nadu Police / NCRB district or station-wise crime
counts (data.gov.in's "Crime in India" tables, or a Right-to-Information
request to Coimbatore City Police) for a specific year, and recompute
`CRIME_BASELINE_RISK` — or a per-station factor — from that real number
instead of the single 2017 city-level ranking used here as a placeholder.

## 4. Road network — mock grid, pending a real OSM download

Exactly like Chennai's Phase 5 fallback (`15b_generate_mock_road_network.py`),
`graph.json` here is a **synthetic 4-connected grid with ~15% of edges
randomly dropped**, because this sandboxed environment has no internet
access to OpenStreetMap's Overpass API. It is good enough to exercise the
full app (routing, safety-weighted vs. shortest comparison, Online/Offline
modes) end to end, but it does **not** follow Coimbatore's real streets.

`15_download_road_network_coimbatore.py` (same pattern as Chennai's
`phase5/15_download_road_network.py`) is provided for you to run **on your
own machine, with internet**, to replace this mock graph with Coimbatore's
real OSM road network and real hospital/police coordinates from
OpenStreetMap.

## 5. Files

- `frontend_data/cities.json` — city registry (name, map center, data file
  paths, boundary, documented data notes) read by `index.html`.
- `frontend_data/coimbatore/safety_cells.geojson` — 1,102 grid cells.
- `frontend_data/coimbatore/graph.json` — 1,102 nodes / 1,830 mock edges.
- `frontend_data/coimbatore/police_stations.geojson` — 18 real stations.
- `frontend_data/coimbatore/hospitals.geojson` — 5 real hospitals.
- `15_download_road_network_coimbatore.py` — run on your own machine to
  replace the mock graph with real OSM data later.
