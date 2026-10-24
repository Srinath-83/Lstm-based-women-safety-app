# SafeRoute – Crime-Aware Women Safety Navigation (Chennai & Coimbatore)

SafeRoute is a browser-based women-safety navigation prototype. It detects whether the user is in Chennai or Coimbatore, shows a safety heatmap with police stations and hospitals, and compares a **safety-weighted walking route** with the **shortest route**. A small Flask + SQLite backend provides registration, login, location history and SOS records.

> **Status: working prototype, not a validated safety product.**
> Chennai safety scores come from a **synthetic** dataset and assumed weights. Coimbatore uses a simple zone-level formula. See [Limitations](#limitations).

## Features

- **Online mode** – current position from the browser Geolocation API.
- **Offline mode** – reuses the last recorded location stored in the browser (`localStorage`). It is a user-selected setting, not tied to `navigator.onLine`.
- **City detection** – point-in-polygon geofence for Chennai and Coimbatore (rectangular boundaries, not official ones). A position outside both cities shows a message and no city is guessed.
- **Map layers** – safety heatmap, police stations and hospitals (Leaflet 1.9.4, bundled locally).
- **Routing** – Dijkstra's algorithm on an OpenStreetMap walking graph, run twice (safety-weighted cost and plain length).
- **Accounts** – registration (name, mobile, email, address, nationality, 3 emergency contacts), login, logout, hashed passwords.
- **Location history** – each recorded position is stored per user in SQLite.
- **SOS** – records the alert with coordinates, tries SMS through Twilio, and shows manual WhatsApp/SMS links as a fallback.

## Project structure

| Folder | Purpose |
|---|---|
| `phase1/` | Synthetic footfall and crime data for the Chennai grid (100 × 44 cells) |
| `phase2/` | Feature tensor and sliding-window sequences |
| `phase3/` | Baselines (LR, RF), LSTM and ConvLSTM training and comparison |
| `phase4/` | Safety score generation and a static prototype map |
| `phase5/` | OSM road network download and safety-weighted edge costs |
| `phase6/` | Browser app (`app/index.html`), Coimbatore data scripts |
| `phase7/` | Flask backend (`app.py`), `templates/`, `requirements.txt`, `sms_config.json` |

## Requirements

- Python 3.10+
- `flask` (backend)
- For the data pipeline: NumPy, pandas, scikit-learn, PyTorch, OSMnx, GeoPandas
- A modern browser; Geolocation needs a secure context (`localhost` counts as secure)

## Quick start (run the app)

```bash
cd phase7
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000**. The registration page appears first, and after registering or logging in you are taken to the map. The SQLite file `saferoute.db` is created on first run; inspect it with `sqlite3 saferoute.db` or any SQLite viewer.

## Preparing city data (only if the exported files are missing)

Run the phase scripts in order. Chennai and Coimbatore each need their exported files (`safety_cells.geojson`, `graph.json`, police and hospital files) in the front-end data folder, together with `cities.json`.

```bash
# Chennai (phases 1–5)
#   synthetic data -> sequences -> models -> safety scores -> OSM graph
python phase5/15_download_road_network.py
python phase5/16_merge_safety_scores.py

# Coimbatore
python phase6/15_download_road_network_coimbatore.py
python phase6/21_build_coimbatore_real_graph.py
```

Downloads need internet access and take a while for large cities. After regenerating data, hard-refresh the browser (the app fetches data with `cache: 'no-store'`).

## Regenerating large data files

These files are excluded from the repository due to GitHub's size limits and 
should be regenerated locally before running later-phase scripts:

| File(s) | Regenerate with |
|---|---|
| phase1/outputs/crime_events.csv, mobility_footfall.csv, chennai_safety_map.html | `python phase1/02_generate_synthetic_data.py` → `python phase1/04_visualize_chennai_map.py` |
| phase2/outputs/**/*.npy | `python phase2/05_build_tensor.py` → `python phase2/06_build_sequences.py` |
| phase3/outputs/*.pt, *.npy | `python phase3/09_train_lstm.py`, `python phase3/10_train_convlstm.py` |
| phase4/outputs/*.npy | `python phase4/12_generate_safety_map.py` |
| phase5/outputs/*.graphml, phase6/coimbatore_road_network.graphml | `python phase5/15_download_road_network.py`, `python phase6/15_download_road_network_coimbatore.py` |

Twilio SMS credentials: copy `phase7/sma_config_placeholder.json` to `phase7/sms_config.json` 
and fill in your own Account SID, Auth Token, and sender number.

## How the safety score works

For grid cell *i*:

```
R_i = 0.5·P_i + 0.3·(1 − L_i) + 0.2·D_i        (clipped to [0, 1])
S_i = 100·(1 − R_i)
edge cost = length_m · (1 + k·ρ),  ρ = 1 − (S_u + S_v)/200,  k = 3.0
```

- `P_i` crime probability (Chennai: model output), `L_i` street lighting, `D_i` normalised distance to the nearest police station.
- The weights and `k` are **design assumptions**, not fitted values.
- **Coimbatore:** `R = 0.6·0.32 + 0.4·min(d/3 km, 1)`, where 0.32 is a baseline derived from the city's NCRB *Crime in India 2017* ranking (as summarised by Factly) and `d` is the distance to the nearest police station.

## SOS alerts and SMS

- Automatic SMS uses the Twilio REST API. Put your credentials in `phase7/sms_config.json` (or in the `TWILIO_*` environment variables). Keep this file private and never commit real credentials.
- **Twilio trial accounts cannot message unverified numbers** (errors 572002 / 572006 were seen in testing). Use a paid account, or verify the recipient numbers.
- If automatic delivery fails, the SOS pop-up shows pre-filled WhatsApp and SMS links for each contact. The user has to tap send.
- E-mail alerts and automatic WhatsApp delivery are **not implemented**.

## Results summary (synthetic data, quick test)

2-day look-back, 220 days of data, 10 epochs:

| Model | Recall | Precision | F1 |
|---|---|---|---|
| Logistic Regression | 0.5261 | 0.0169 | 0.0327 |
| Random Forest | 0.0000 | 0.0000 | 0.0000 |
| LSTM | 0.6648 | 0.0159 | 0.0311 |
| ConvLSTM | 0.4341 | 0.0155 | 0.0299 |

The ConvLSTM did **not** outperform the LSTM or logistic regression in this run. These numbers come from synthetic data and say nothing about real-world prediction quality.

## Limitations

- Chennai crime data is synthetic; the 45,000 incidents per year target and the 28 % women-safety share are placeholders, not official statistics.
- Coimbatore uses one city-wide baseline and has no trained model.
- City boundaries are rectangles, not official administrative polygons.
- Offline mode reuses a saved position only; there are no offline maps or offline routing.
- Walking routes only; no turn-by-turn guidance.
- Tested with automated headless-browser scripts and simulated GPS; no testing on real devices yet.

## Future work

Real, ethically sourced crime data and validation; official city boundaries and more cities; automated alerts (WhatsApp Business Cloud API, Telegram or a paid SMS provider); longer model training with better class-imbalance handling; time-of-day scores, offline map tiles and a user study.

## Data sources and credits

- Map data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, accessed via [OSMnx](https://github.com/gboeing/osmnx)
- [Leaflet](https://leafletjs.com) for map rendering
- NCRB *Crime in India 2017* (via Factly) for the Coimbatore baseline
- Flask and SQLite for the backend



## Licence**

Academic project – all rights reserved"**
