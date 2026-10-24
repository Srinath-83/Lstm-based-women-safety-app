"""
04_visualize.py
----------------
Phase 1 visualization for the Chennai women-safety project.

Instead of plotting the grid as an abstract matplotlib image, this version
places the generated data on an interactive Chennai basemap using Folium.

Outputs:
    outputs/chennai_safety_map.html
        Interactive map with:
        1. Night-time crime heatmap
        2. Women-safety-relevant crime heatmap
        3. Night-crime grid layer
        4. Women-safety grid layer
        5. Streetlight coverage
        6. Simulated police-station markers
        7. Area-type grid layer

Run this AFTER 01_build_grid.py, 02_generate_synthetic_data.py,
and 03_generate_sociodemographic.py.
"""

import os
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from branca.colormap import linear
import config as cfg


OUTPUT_DIR = "outputs"
GRID_FILE = os.path.join(OUTPUT_DIR, "grid_cells.csv")
CRIME_FILE = os.path.join(OUTPUT_DIR, "crime_events.csv")
MAP_FILE = os.path.join(OUTPUT_DIR, "chennai_safety_map.html")


def load_data():
    """Load the Phase 1 grid and crime data."""
    if not os.path.exists(GRID_FILE):
        raise FileNotFoundError(
            f"{GRID_FILE} not found. Run 01_build_grid.py first."
        )

    if not os.path.exists(CRIME_FILE):
        raise FileNotFoundError(
            f"{CRIME_FILE} not found. Run 02_generate_synthetic_data.py first."
        )

    grid = pd.read_csv(GRID_FILE)
    crime = pd.read_csv(CRIME_FILE)

    return grid, crime


def aggregate_crime(grid, crime):
    """
    Aggregate crime over the simulated period.

    Night crime:
        only time_block == 'night'

    Women-safety crime:
        all time blocks, using women_safety_relevant
    """
    night = crime[crime["time_block"] == "night"]

    night_counts = (
        night.groupby("cell_id")["crime_occurred"]
        .sum()
        .rename("night_crime_count")
    )

    women_counts = (
        crime.groupby("cell_id")["women_safety_relevant"]
        .sum()
        .rename("women_safety_count")
    )

    data = grid.copy()
    data = data.merge(night_counts, on="cell_id", how="left")
    data = data.merge(women_counts, on="cell_id", how="left")

    data["night_crime_count"] = data["night_crime_count"].fillna(0)
    data["women_safety_count"] = data["women_safety_count"].fillna(0)

    return data


def add_grid_heatmap(
    fmap,
    data,
    value_column,
    layer_name,
    colormap,
    fill_opacity=0.62,
):
    """
    Add the actual grid cells as colored rectangles on top of the
    OpenStreetMap basemap.

    Each rectangle corresponds to one generated ~0.2 sq km cell.
    """
    layer = folium.FeatureGroup(name=layer_name, show=False)

    values = data[value_column].astype(float)
    vmax = max(float(values.max()), 1.0)

    for _, cell in data.iterrows():
        value = float(cell[value_column])

        # Convert the numeric value into a color.
        color = colormap(value / vmax)

        popup = folium.Popup(
            f"""
            <b>Cell ID:</b> {int(cell['cell_id'])}<br>
            <b>Area type:</b> {cell['area_type']}<br>
            <b>Night crime:</b> {int(cell['night_crime_count'])}<br>
            <b>Women-safety crime:</b> {int(cell['women_safety_count'])}<br>
            <b>Streetlight score:</b> {cell['streetlight_score']:.2f}<br>
            <b>Distance to police:</b> {cell['dist_to_police_km']:.2f} km
            """,
            max_width=300,
        )

        folium.Rectangle(
            bounds=[
                [cell["lat_min"], cell["lon_min"]],
                [cell["lat_max"], cell["lon_max"]],
            ],
            color=None,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            weight=0,
            popup=popup,
        ).add_to(layer)

    layer.add_to(fmap)
    return layer


def add_smooth_heatmap(
    fmap,
    data,
    value_column,
    layer_name,
    gradient,
    radius=18,
    blur=22,
):
    """
    Add a smooth Leaflet heatmap using the center of every grid cell.

    The cell value is used as the heatmap weight, so high-crime cells
    appear hotter on the real Chennai basemap.
    """
    layer = folium.FeatureGroup(name=layer_name, show=True)

    values = data[value_column].astype(float)
    vmax = float(values.max())

    if vmax <= 0:
        layer.add_to(fmap)
        return layer

    heat_data = [
        [
            float(row["center_lat"]),
            float(row["center_lon"]),
            float(row[value_column]) / vmax,
        ]
        for _, row in data.iterrows()
        if float(row[value_column]) > 0
    ]

    HeatMap(
        heat_data,
        radius=radius,
        blur=blur,
        min_opacity=0.20,
        max_zoom=14,
        gradient=gradient,
    ).add_to(layer)

    layer.add_to(fmap)
    return layer


def add_streetlight_layer(fmap, data):
    """Show streetlight quality as a map layer."""
    layer = folium.FeatureGroup(
        name="Streetlight Coverage",
        show=False,
    )

    # Red = poor lighting, green = good lighting.
    for _, cell in data.iterrows():
        score = float(cell["streetlight_score"])

        # Use a simple red-yellow-green interpolation.
        if score < 0.5:
            ratio = score / 0.5
            color = "#{:02x}{:02x}{:02x}".format(
                255,
                int(80 + 175 * ratio),
                0,
            )
        else:
            ratio = (score - 0.5) / 0.5
            color = "#{:02x}{:02x}{:02x}".format(
                int(255 * (1 - ratio)),
                220,
                int(80 * ratio),
            )

        folium.Rectangle(
            bounds=[
                [cell["lat_min"], cell["lon_min"]],
                [cell["lat_max"], cell["lon_max"]],
            ],
            color=None,
            fill=True,
            fill_color=color,
            fill_opacity=0.45,
            weight=0,
            popup=folium.Popup(
                f"""
                <b>Cell ID:</b> {int(cell['cell_id'])}<br>
                <b>Streetlight score:</b> {score:.2f}<br>
                <b>Area type:</b> {cell['area_type']}
                """,
                max_width=250,
            ),
        ).add_to(layer)

    layer.add_to(fmap)


def add_area_type_layer(fmap, data):
    """Show the simulated dominant area type per grid cell."""
    colors = {
        "residential": "#377eb8",
        "commercial": "#ff7f00",
        "market": "#2ca02c",
        "bus_stand": "#9467bd",
        "entertainment": "#8c564b",
        "educational": "#7f7f7f",
        "industrial": "#bcbd22",
        "hospital": "#17becf",
    }

    layer = folium.FeatureGroup(
        name="Area Types",
        show=False,
    )

    for _, cell in data.iterrows():
        area_type = cell["area_type"]

        folium.Rectangle(
            bounds=[
                [cell["lat_min"], cell["lon_min"]],
                [cell["lat_max"], cell["lon_max"]],
            ],
            color="white",
            weight=0.2,
            fill=True,
            fill_color=colors.get(area_type, "#000000"),
            fill_opacity=0.55,
            popup=folium.Popup(
                f"""
                <b>Cell ID:</b> {int(cell['cell_id'])}<br>
                <b>Area type:</b> {area_type}<br>
                <b>Streetlight:</b> {cell['streetlight_score']:.2f}<br>
                <b>Police distance:</b> {cell['dist_to_police_km']:.2f} km
                """,
                max_width=250,
            ),
        ).add_to(layer)

    layer.add_to(fmap)


def add_police_stations(fmap, data):
    """Add the 25 simulated police-station cells as markers."""
    layer = folium.FeatureGroup(
        name="Simulated Police Stations",
        show=True,
    )

    police = data[data["is_police_station"] == True]

    for _, station in police.iterrows():
        folium.Marker(
            location=[
                station["center_lat"],
                station["center_lon"],
            ],
            popup=folium.Popup(
                f"""
                <b>Simulated Police Station</b><br>
                Cell ID: {int(station['cell_id'])}<br>
                Area type: {station['area_type']}
                """,
                max_width=250,
            ),
            tooltip="Simulated Police Station",
            icon=folium.Icon(
                color="blue",
                icon="shield",
                prefix="fa",
            ),
        ).add_to(layer)

    layer.add_to(fmap)


def add_legend(fmap):
    """Add a short project disclaimer/legend to the map."""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        background-color: white;
        padding: 12px 16px;
        border: 2px solid #555;
        border-radius: 6px;
        font-size: 13px;
        max-width: 330px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    ">
        <b>Chennai Women-Safety Project</b><br>
        🔴 Hotter areas = higher simulated crime concentration<br>
        🔵 Police markers = simulated stations<br><br>
        <b>Important:</b> Crime, streetlight and police-location
        layers are synthetic Phase 1 data, not live police records.
    </div>
    """

    fmap.get_root().html.add_child(
        folium.Element(legend_html)
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grid, crime = load_data()
    data = aggregate_crime(grid, crime)

    # Center the map on the configured Chennai bounding box.
    center_lat = (cfg.LAT_MIN + cfg.LAT_MAX) / 2
    center_lon = (cfg.LON_MIN + cfg.LON_MAX) / 2

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    # Fit exactly to the generated project bounding box.
    fmap.fit_bounds(
        [
            [cfg.LAT_MIN, cfg.LON_MIN],
            [cfg.LAT_MAX, cfg.LON_MAX],
        ]
    )

    # ------------------------------------------------------------------
    # Main heatmaps
    # ------------------------------------------------------------------

    # Smooth night-crime heatmap.
    add_smooth_heatmap(
        fmap,
        data,
        "night_crime_count",
        "Night Crime Heatmap",
        gradient={
            "0.15": "#440154",
            "0.35": "#31688e",
            "0.55": "#35b779",
            "0.75": "#fde725",
            "1.0": "#ff0000",
        },
        radius=18,
        blur=22,
    )

    # Smooth women's-safety heatmap.
    add_smooth_heatmap(
        fmap,
        data,
        "women_safety_count",
        "Women-Safety Crime Heatmap",
        gradient={
            "0.15": "#440154",
            "0.35": "#9e0142",
            "0.55": "#f46d43",
            "0.75": "#fee08b",
            "1.0": "#ffffff",
        },
        radius=18,
        blur=22,
    )

    # ------------------------------------------------------------------
    # Cell-level layers
    # ------------------------------------------------------------------

    night_colormap = linear.YlOrRd_09.scale(
        0,
        max(float(data["night_crime_count"].max()), 1),
    )
    night_colormap.caption = "Night-time crime count"

    women_colormap = linear.PuRd_09.scale(
        0,
        max(float(data["women_safety_count"].max()), 1),
    )
    women_colormap.caption = "Women-safety-relevant crime count"

    add_grid_heatmap(
        fmap,
        data,
        "night_crime_count",
        "Night Crime — Grid Cells",
        night_colormap,
        fill_opacity=0.55,
    )

    add_grid_heatmap(
        fmap,
        data,
        "women_safety_count",
        "Women Safety — Grid Cells",
        women_colormap,
        fill_opacity=0.55,
    )

    add_streetlight_layer(fmap, data)
    add_area_type_layer(fmap, data)
    add_police_stations(fmap, data)

    # Show layer controls so the user can switch between datasets.
    folium.LayerControl(collapsed=False).add_to(fmap)

    # Project title.
    title_html = """
    <div style="
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: white;
        padding: 8px 18px;
        border-radius: 6px;
        border: 1px solid #777;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.25);
    ">
        Chennai Women-Safety Crime Risk Map
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))

    add_legend(fmap)

    # Save interactive map.
    fmap.save(MAP_FILE)

    print("\n--- Chennai Map Summary ---")
    print(f"City: {cfg.CITY_NAME}")
    print(f"Grid: {grid['row'].nunique()} x {grid['col'].nunique()} "
          f"= {len(grid)} cells")
    print(f"Simulated period: {cfg.NUM_DAYS} days")
    print(f"Night crimes: {int(data['night_crime_count'].sum())}")
    print(f"Women-safety-relevant crimes: "
          f"{int(data['women_safety_count'].sum())}")
    print(f"Simulated police stations: "
          f"{int(data['is_police_station'].sum())}")
    print(f"\nSaved interactive map to: {MAP_FILE}")


if __name__ == "__main__":
    main()
