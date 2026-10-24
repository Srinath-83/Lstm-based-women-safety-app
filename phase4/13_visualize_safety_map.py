"""
13_visualize_safety_map.py
-----------------------------
Two visualizations of your safety scores:
  1. A static matplotlib heatmap (good for your report)
  2. An interactive folium map (good for demos -- consistent with the
     folium setup you added in Phase 1)

Run this AFTER 12_generate_safety_map.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
import config_phase4 as cfg

safety_score = np.load(f"{cfg.PHASE4_OUTPUTS_DIR}/safety_score_grid.npy")
crime_prob = np.load(f"{cfg.PHASE4_OUTPUTS_DIR}/crime_probability_grid.npy")
scores_df = pd.read_csv(f"{cfg.PHASE4_OUTPUTS_DIR}/safety_scores.csv")

# --- Matplotlib heatmap -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

im0 = axes[0].imshow(crime_prob, cmap="Reds", origin="lower", vmin=0, vmax=1)
axes[0].set_title("Predicted Crime Probability")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(safety_score, cmap="RdYlGn", origin="lower", vmin=0, vmax=100)
axes[1].set_title("Women's Safety Score (0=risky, 100=safe)")
fig.colorbar(im1, ax=axes[1])

for ax in axes:
    ax.set_xlabel("Grid column (West -> East)")
    ax.set_ylabel("Grid row (South -> North)")

plt.tight_layout()
plt.savefig(f"{cfg.PHASE4_OUTPUTS_DIR}/safety_map_static.png", dpi=150)
print(f"Saved {cfg.PHASE4_OUTPUTS_DIR}/safety_map_static.png")

# --- Interactive folium map -------------------------------------------------
center_lat = scores_df["center_lat"].mean()
center_lon = scores_df["center_lon"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")


def score_to_color(score):
    """0 (red, risky) -> 100 (green, safe), simple linear interpolation."""
    score = max(0, min(100, score))
    if score < 50:
        # red -> yellow
        ratio = score / 50
        r, g, b = 220, int(80 + ratio * 160), 60
    else:
        # yellow -> green
        ratio = (score - 50) / 50
        r, g, b = int(220 - ratio * 170), 220, int(60 + ratio * 20)
    return f"#{r:02x}{g:02x}{b:02x}"


# plotting every single cell as a rectangle can be slow/heavy in folium for
# 4400 cells -- subsample for a lighter demo map, or plot all if you don't
# mind a bigger HTML file. Here we plot all, but keep opacity/geometry simple.
for _, row in scores_df.iterrows():
    folium.Rectangle(
        bounds=[[row["lat_min"], row["lon_min"]], [row["lat_max"], row["lon_max"]]],
        color=None,
        fill=True,
        fill_color=score_to_color(row["safety_score"]),
        fill_opacity=0.6,
        weight=0,
    ).add_to(m)

# mark police stations
police_cells = scores_df[scores_df["is_police_station"] == True]
for _, row in police_cells.iterrows():
    folium.Marker(
        location=[row["center_lat"], row["center_lon"]],
        icon=folium.Icon(color="blue", icon="shield", prefix="fa"),
        popup="Police Station",
    ).add_to(m)

out_path = f"{cfg.PHASE4_OUTPUTS_DIR}/safety_map_interactive.html"
m.save(out_path)
print(f"Saved {out_path} (open this in a browser)")
