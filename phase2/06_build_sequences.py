"""
06_build_sequences.py
----------------------
Turns the big city_tensor.npy into actual (X, y) training examples for
the LSTM/ConvLSTM, following the base paper's method:

  1. SLIDING WINDOW: use T consecutive frames (t-T ... t-1) to predict
     frame t. Slide this window across the whole time series.

  2. SUB-GRID SAMPLING: our full city grid (100x44) is too big to feed
     directly into a ConvLSTM for a student project (slow, memory-heavy).
     The paper solves this by randomly cropping smaller MxM patches
     (M=16) out of the full grid for each time window -- this also acts
     as data augmentation (multiple training examples per time window).

  3. BALANCING: since crime is rare (~1-2% of cells have crime in any
     block), a randomly cropped 16x16 patch is often ALL empty. The paper
     only keeps crops that contain at least MIN_CRIMES_IN_SUBGRID crimes
     in the input window, so the model actually has something to learn
     from. We do the same.

  4. CHRONOLOGICAL TRAIN/TEST SPLIT: we split by TIME (not randomly) so
     the model is tested on the future relative to what it trained on --
     this is the only fair way to evaluate a forecasting model. We also
     drop a small buffer between train and test to avoid "leaking"
     information across the boundary (same fix the paper uses).

Run this AFTER 05_build_tensor.py.

Outputs (into outputs/<lookback_name>/):
    X_train.npy   shape: (n_train_sequences, T, 16, 16, n_channels)
    y_train.npy   shape: (n_train_sequences, 16, 16)
    X_test.npy    shape: (n_test_sequences, T, 16, 16, n_channels)
    y_test.npy    shape: (n_test_sequences, 16, 16)
"""

import json
import numpy as np
import config_phase2 as cfg

np.random.seed(cfg.RANDOM_SEED)


def load_tensor():
    tensor = np.load(f"{cfg.PHASE2_OUTPUTS_DIR}/city_tensor.npy")
    with open(f"{cfg.PHASE2_OUTPUTS_DIR}/channel_names.json") as f:
        channel_names = json.load(f)
    crime_channel_idx = channel_names.index("crime_occurred")
    return tensor, channel_names, crime_channel_idx


def sample_subgrid_sequences(tensor, crime_channel_idx, T, n_rows, n_cols):
    """
    For every valid sliding-window position, try to draw
    SAMPLES_PER_SEQUENCE random MxM crops that contain enough crime
    activity in the input window to be useful training examples.
    """
    M = cfg.SUBGRID_SIZE
    n_timesteps = tensor.shape[0]
    max_row_start = n_rows - M
    max_col_start = n_cols - M

    X_list, y_list = [], []

    # valid window start times: need T frames before t, and t itself
    for t in range(T, n_timesteps):
        input_frames = tensor[t - T:t]        # shape (T, n_rows, n_cols, C)
        target_frame_crime = tensor[t, :, :, crime_channel_idx]  # (n_rows, n_cols)

        found = 0
        attempts = 0
        while found < cfg.SAMPLES_PER_SEQUENCE and attempts < cfg.MAX_SUBGRID_ATTEMPTS:
            attempts += 1
            r0 = np.random.randint(0, max_row_start + 1)
            c0 = np.random.randint(0, max_col_start + 1)

            crop_input = input_frames[:, r0:r0 + M, c0:c0 + M, :]
            crime_count_in_crop = crop_input[..., crime_channel_idx].sum()

            if crime_count_in_crop >= cfg.MIN_CRIMES_IN_SUBGRID:
                crop_target = target_frame_crime[r0:r0 + M, c0:c0 + M]
                X_list.append(crop_input)
                y_list.append(crop_target)
                found += 1

    if len(X_list) == 0:
        return None, None

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    return X, y


def chronological_split(tensor, T):
    """
    Returns (train_tensor, test_tensor) split 90/10 by time, with a
    buffer of T removed from the end of train to prevent any overlap
    between a training window and a testing window (data leakage).
    """
    n_timesteps = tensor.shape[0]
    split_point = int(n_timesteps * cfg.TRAIN_FRACTION)

    train_tensor = tensor[:split_point - T]   # drop last T to avoid leakage
    test_tensor = tensor[split_point:]

    print(f"Chronological split: {split_point} / {n_timesteps} timesteps "
          f"-> train has {train_tensor.shape[0]} (after leakage buffer), "
          f"test has {test_tensor.shape[0]}")
    return train_tensor, test_tensor


def build_for_lookback(tensor, channel_names, crime_channel_idx, lookback_name, T):
    print(f"\n=== Building sequences for look-back = {lookback_name} (T={T}) ===")
    n_rows, n_cols = tensor.shape[1], tensor.shape[2]

    train_tensor, test_tensor = chronological_split(tensor, T)

    print("Sampling training sub-grid sequences...")
    X_train, y_train = sample_subgrid_sequences(
        train_tensor, crime_channel_idx, T, n_rows, n_cols
    )
    print("Sampling test sub-grid sequences...")
    X_test, y_test = sample_subgrid_sequences(
        test_tensor, crime_channel_idx, T, n_rows, n_cols
    )

    if X_train is None or X_test is None:
        print(f"  WARNING: not enough crime-positive sub-grids found for "
              f"T={T}. Try lowering MIN_CRIMES_IN_SUBGRID in config_phase2.py "
              f"or using more days of data (QUICK_TEST_MAX_DAYS / QUICK_TEST_MODE).")
        return

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape} "
          f"(positive rate: {y_train.mean() * 100:.2f}%)")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape} "
          f"(positive rate: {y_test.mean() * 100:.2f}%)")

    out_dir = f"{cfg.PHASE2_OUTPUTS_DIR}/{lookback_name}"
    import os
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/X_train.npy", X_train)
    np.save(f"{out_dir}/y_train.npy", y_train)
    np.save(f"{out_dir}/X_test.npy", X_test)
    np.save(f"{out_dir}/y_test.npy", y_test)
    with open(f"{out_dir}/channel_names.json", "w") as f:
        json.dump(channel_names, f, indent=2)

    print(f"  Saved to {out_dir}/")


if __name__ == "__main__":
    tensor, channel_names, crime_channel_idx = load_tensor()
    print(f"Loaded tensor: {tensor.shape}")

    for lookback_name, T in cfg.LOOKBACK_PERIODS.items():
        build_for_lookback(tensor, channel_names, crime_channel_idx,
                            lookback_name, T)
