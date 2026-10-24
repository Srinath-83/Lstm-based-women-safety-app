"""
12_generate_safety_map.py
---------------------------
Uses your trained ConvLSTM (from Phase 3) to predict crime probability
for the ENTIRE city grid at once -- not just the 16x16 crops it was
trained on.

This works because ConvLSTM is "fully convolutional": every operation
inside it (the convolutions in each gate, the final classifier) slides
a small kernel across whatever image size you give it. A model trained
on 16x16 patches will happily run on a 100x44 grid at inference time --
this is a standard, well-known property of CNN-based architectures, not
a hack.

Then combines that crime prediction with "guardianship" factors
(streetlight coverage, distance to police) into a single Safety Score
(0-100, higher = safer) per cell.

Run this AFTER Phase 3's 10_train_convlstm.py (or 09_train_lstm.py, if
MODEL_TO_USE = "lstm" -- though note the LSTM baseline has no spatial
convolution, so it can't be run on the full grid using this trick; see
the fallback logic below).
"""

import json
import numpy as np
import pandas as pd
import torch

import config_phase4 as cfg
import config_phase3 as cfg3
from models import ConvLSTM


def load_full_tensor():
    tensor = np.load(f"{cfg.PHASE2_OUTPUTS_DIR}/city_tensor.npy")
    with open(f"{cfg.PHASE2_OUTPUTS_DIR}/channel_names.json") as f:
        channel_names = json.load(f)
    time_index = pd.read_csv(f"{cfg.PHASE2_OUTPUTS_DIR}/time_index.csv")
    return tensor, channel_names, time_index


def load_convlstm_model(input_channels):
    model = ConvLSTM(
        input_channels=input_channels,
        hidden_channels=cfg3.CONVLSTM_HIDDEN_CHANNELS,
        num_layers=cfg3.CONVLSTM_NUM_LAYERS,
        kernel_size=cfg3.CONVLSTM_KERNEL_SIZE,
        dropout=cfg3.DROPOUT_RATE,
    )
    state_dict = torch.load(
        f"{cfg.PHASE3_OUTPUTS_DIR}/convlstm_model.pt", map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_full_city(model, tensor, T):
    """
    Feed the model the most recent T frames of the FULL city grid and get
    back a crime-probability map for the whole city in one shot.
    """
    input_frames = tensor[-T:]                        # (T, H, W, C)
    input_batch = torch.tensor(
        input_frames[np.newaxis, ...], dtype=torch.float32
    )                                                  # (1, T, H, W, C)
    logits = model(input_batch)                        # (1, H, W)
    probs = torch.sigmoid(logits).squeeze(0).numpy()    # (H, W)
    return probs


def compute_safety_score(crime_prob_grid, tensor, channel_names):
    """
    Combine crime probability with guardianship factors into a 0-100
    safety score per cell. streetlight_score and dist_to_police_km are
    already present (and already min-max normalized to [0,1]) as static
    channels in the tensor -- take them from the most recent frame.
    """
    last_frame = tensor[-1]  # (H, W, C) -- static channels are the same
                              # in every frame, so any frame works
    light_idx = channel_names.index("streetlight_score")
    police_idx = channel_names.index("dist_to_police_km")

    streetlight = last_frame[:, :, light_idx]       # already 0 (dark) - 1 (bright)
    dist_police_norm = last_frame[:, :, police_idx]  # already 0 (near) - 1 (far)

    risk = (
        cfg.W_CRIME * crime_prob_grid
        + cfg.W_LIGHT * (1 - streetlight)
        + cfg.W_POLICE * dist_police_norm
    )
    risk = np.clip(risk, 0, 1)
    safety_score = 100 * (1 - risk)
    return safety_score, risk


def main():
    tensor, channel_names, time_index = load_full_tensor()
    n_rows, n_cols = tensor.shape[1], tensor.shape[2]
    print(f"Loaded city tensor: {tensor.shape} "
          f"({n_rows} x {n_cols} grid, {tensor.shape[0]} time steps)")

    # look-back T comes from Phase 2's config (that's where LOOKBACK_PERIODS
    # is defined); Phase 3's config just tells us *which* one was used to
    # train the model we're loading.
    import config_phase2 as cfg2
    T = cfg2.LOOKBACK_PERIODS[cfg3.LOOKBACK_TO_USE]
    print(f"Using look-back T={T} ({cfg3.LOOKBACK_TO_USE}), model={cfg.MODEL_TO_USE}")

    if cfg.MODEL_TO_USE != "convlstm":
        raise NotImplementedError(
            "Full-grid inference relies on ConvLSTM's fully-convolutional "
            "property. The plain LSTM baseline processes each cell "
            "independently and technically *could* be run cell-by-cell "
            "across the full grid too, but that's not implemented here -- "
            "set MODEL_TO_USE = 'convlstm' in config_phase4.py."
        )

    model = load_convlstm_model(input_channels=tensor.shape[-1])
    print(f"Loaded trained ConvLSTM ({sum(p.numel() for p in model.parameters()):,} params)")

    crime_prob_grid = predict_full_city(model, tensor, T)
    print(f"Predicted crime probability grid: shape {crime_prob_grid.shape}, "
          f"min={crime_prob_grid.min():.3f}, max={crime_prob_grid.max():.3f}, "
          f"mean={crime_prob_grid.mean():.3f}")

    safety_score, risk = compute_safety_score(crime_prob_grid, tensor, channel_names)
    print(f"Safety score: min={safety_score.min():.1f}, max={safety_score.max():.1f}, "
          f"mean={safety_score.mean():.1f}")

    np.save(f"{cfg.PHASE4_OUTPUTS_DIR}/crime_probability_grid.npy", crime_prob_grid)
    np.save(f"{cfg.PHASE4_OUTPUTS_DIR}/safety_score_grid.npy", safety_score)
    np.save(f"{cfg.PHASE4_OUTPUTS_DIR}/risk_grid.npy", risk)

    # also save a per-cell CSV merged with lat/lon, for mapping / the routing step
    grid = pd.read_csv(f"{cfg.PHASE1_OUTPUTS_DIR}/grid_cells.csv")
    grid = grid.sort_values("cell_id").reset_index(drop=True)
    grid["crime_probability"] = crime_prob_grid.flatten()
    grid["risk_score"] = risk.flatten()
    grid["safety_score"] = safety_score.flatten()
    grid.to_csv(f"{cfg.PHASE4_OUTPUTS_DIR}/safety_scores.csv", index=False)

    print(f"\nSaved:")
    print(f"  {cfg.PHASE4_OUTPUTS_DIR}/crime_probability_grid.npy")
    print(f"  {cfg.PHASE4_OUTPUTS_DIR}/safety_score_grid.npy")
    print(f"  {cfg.PHASE4_OUTPUTS_DIR}/risk_grid.npy")
    print(f"  {cfg.PHASE4_OUTPUTS_DIR}/safety_scores.csv  (per-cell, with lat/lon)")


if __name__ == "__main__":
    main()
