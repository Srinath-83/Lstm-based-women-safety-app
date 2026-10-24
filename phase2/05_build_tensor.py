"""
05_build_tensor.py
-------------------
Turns the flat CSVs from Phase 1 into a single 4D array shaped like the
base paper's Figure 2:

    tensor[t, row, col, channel]

...i.e. a "video" of the city, where each frame (t) is a 12-hour block,
and each frame has multiple channels (like RGB channels in an image, but
here: crime, footfall, area-type, streetlights, distance to police,
population density, literacy, income).

WHY this shape? Because ConvLSTM (used in Phase 3) is built for video-like
data: (time, height, width, channels) -- exactly what CNNs use for images,
plus a time dimension for the LSTM part to learn from.

Run this AFTER Phase 1's scripts have produced:
    ../phase1/outputs/grid_cells.csv
    ../phase1/outputs/crime_events.csv   (already contains footfall, merged
                                           in during Phase 1 step 02)
    ../phase1/outputs/sociodemographic.csv

Outputs (into outputs/):
    city_tensor.npy       -- the 4D array described above
    channel_names.json    -- what each channel index means
    time_index.csv        -- maps tensor time-index t -> (day, time_block)
"""

import json
import numpy as np
import pandas as pd
import config_phase2 as cfg

np.random.seed(cfg.RANDOM_SEED)

AREA_TYPES = [
    "residential", "commercial", "market", "bus_stand",
    "entertainment", "educational", "industrial", "hospital",
]


def load_static_features():
    """Per-cell features that don't change over time: area type (one-hot),
    streetlight score, distance to police, and sociodemographics."""
    grid = pd.read_csv(f"{cfg.PHASE1_OUTPUTS_DIR}/grid_cells.csv")
    grid = grid.sort_values("cell_id").reset_index(drop=True)
    n_rows = int(grid["row"].max()) + 1
    n_cols = int(grid["col"].max()) + 1

    # sanity check: cell_id should equal row * n_cols + col (row-major),
    # which is what lets us reshape a flat per-cell array straight into
    # a (n_rows, n_cols) grid without an expensive pivot operation.
    expected = grid["row"] * n_cols + grid["col"]
    assert (grid["cell_id"].to_numpy() == expected.to_numpy()).all(), \
        "cell_id ordering assumption broken -- check 01_build_grid.py"

    # one-hot encode area_type
    area_onehot = pd.get_dummies(grid["area_type"]).reindex(
        columns=AREA_TYPES, fill_value=0
    ).astype(np.float32)

    static_df = pd.concat([
        grid[["streetlight_score", "dist_to_police_km"]],
        area_onehot,
    ], axis=1)

    socio = pd.read_csv(f"{cfg.PHASE1_OUTPUTS_DIR}/sociodemographic.csv")
    socio = socio.sort_values("cell_id").reset_index(drop=True)
    static_df = pd.concat([
        static_df,
        socio[["population_density_per_sqkm", "literacy_rate_pct", "income_band"]],
    ], axis=1)

    static_channel_names = list(static_df.columns)
    static_array = static_df.to_numpy(dtype=np.float32)  # shape (n_cells, n_static)

    return static_array, static_channel_names, n_rows, n_cols


def load_dynamic_features():
    """Per-cell, per-time-block features: crime_occurred and footfall."""
    usecols = ["day", "time_block", "cell_id", "footfall", "crime_occurred"]
    dtypes = {
        "day": "int32", "time_block": "category",
        "cell_id": "int32", "footfall": "float32", "crime_occurred": "int8",
    }
    df = pd.read_csv(
        f"{cfg.PHASE1_OUTPUTS_DIR}/crime_events.csv",
        usecols=usecols, dtype=dtypes,
    )

    if cfg.QUICK_TEST_MODE:
        df = df[df["day"] < cfg.QUICK_TEST_MAX_DAYS]
        print(f"[QUICK_TEST_MODE] Using only the first {cfg.QUICK_TEST_MAX_DAYS} days.")

    # global time index t: day*2 + (0 for day-block, 1 for night-block)
    block_offset = df["time_block"].map({"day": 0, "night": 1}).astype("int32")
    df["t"] = df["day"] * 2 + block_offset

    df = df.sort_values(["t", "cell_id"]).reset_index(drop=True)
    return df


def normalize(arr, axis=None):
    """Simple min-max scaling to [0, 1], like the base paper."""
    mn = arr.min(axis=axis, keepdims=True)
    mx = arr.max(axis=axis, keepdims=True)
    rng = np.where((mx - mn) == 0, 1, mx - mn)
    return (arr - mn) / rng


def build_tensor():
    static_array, static_names, n_rows, n_cols = load_static_features()
    n_cells = n_rows * n_cols
    print(f"Grid: {n_rows} x {n_cols} = {n_cells} cells, "
          f"{static_array.shape[1]} static channels: {static_names}")

    dyn_df = load_dynamic_features()
    n_timesteps = dyn_df["t"].nunique()
    print(f"Dynamic data: {n_timesteps} time steps "
          f"({n_timesteps} x 12-hour blocks)")

    # reshape dynamic features: (n_timesteps, n_cells) -> (n_timesteps, n_rows, n_cols)
    crime_flat = dyn_df["crime_occurred"].to_numpy(dtype=np.float32).reshape(
        n_timesteps, n_cells
    )
    footfall_flat = dyn_df["footfall"].to_numpy(dtype=np.float32).reshape(
        n_timesteps, n_cells
    )

    crime_grid = crime_flat.reshape(n_timesteps, n_rows, n_cols)
    footfall_grid = footfall_flat.reshape(n_timesteps, n_rows, n_cols)
    footfall_grid = normalize(footfall_grid)  # scale to [0,1]

    # normalize continuous static channels only (leave one-hot area types as-is)
    static_norm = static_array.copy()
    continuous_idx = [
        static_names.index(c) for c in
        ["streetlight_score", "dist_to_police_km", "population_density_per_sqkm",
         "literacy_rate_pct", "income_band"]
    ]
    for idx in continuous_idx:
        col = static_norm[:, idx]
        static_norm[:, idx] = (col - col.min()) / max(col.max() - col.min(), 1e-8)

    static_grid = static_norm.reshape(n_rows, n_cols, -1)  # (rows, cols, n_static)
    # broadcast static features across every timestep
    static_grid_over_time = np.broadcast_to(
        static_grid, (n_timesteps, n_rows, n_cols, static_grid.shape[-1])
    )

    # stack everything: [crime, footfall, *static channels]
    tensor = np.concatenate([
        crime_grid[..., np.newaxis],
        footfall_grid[..., np.newaxis],
        static_grid_over_time,
    ], axis=-1).astype(np.float32)

    channel_names = ["crime_occurred", "footfall"] + static_names

    print(f"\nFinal tensor shape: {tensor.shape} "
          f"(time, rows, cols, channels)")
    print(f"Channels ({len(channel_names)}): {channel_names}")
    print(f"Tensor size in memory: {tensor.nbytes / 1e6:.1f} MB")

    # time index lookup (t -> day, time_block) for reference/debugging
    time_index = dyn_df[["t", "day", "time_block"]].drop_duplicates().sort_values("t")

    return tensor, channel_names, time_index


if __name__ == "__main__":
    tensor, channel_names, time_index = build_tensor()

    np.save(f"{cfg.PHASE2_OUTPUTS_DIR}/city_tensor.npy", tensor)
    with open(f"{cfg.PHASE2_OUTPUTS_DIR}/channel_names.json", "w") as f:
        json.dump(channel_names, f, indent=2)
    time_index.to_csv(f"{cfg.PHASE2_OUTPUTS_DIR}/time_index.csv", index=False)

    print(f"\nSaved:")
    print(f"  {cfg.PHASE2_OUTPUTS_DIR}/city_tensor.npy")
    print(f"  {cfg.PHASE2_OUTPUTS_DIR}/channel_names.json")
    print(f"  {cfg.PHASE2_OUTPUTS_DIR}/time_index.csv")
