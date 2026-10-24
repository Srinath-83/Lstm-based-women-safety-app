"""
07_visualize_sequence.py
--------------------------
Sanity check: pick one random training sequence and plot:
  - the crime channel across all T input frames (does crime move/cluster
    sensibly over time within the 16x16 patch?)
  - the footfall channel for the last input frame
  - the target frame (what we're trying to predict)

Run this AFTER 06_build_sequences.py.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import config_phase2 as cfg

LOOKBACK_TO_INSPECT = "2_days"   # change to "14_days" to inspect the other one

out_dir = f"{cfg.PHASE2_OUTPUTS_DIR}/{LOOKBACK_TO_INSPECT}"
X = np.load(f"{out_dir}/X_train.npy")
y = np.load(f"{out_dir}/y_train.npy")
with open(f"{out_dir}/channel_names.json") as f:
    channel_names = json.load(f)

crime_idx = channel_names.index("crime_occurred")
footfall_idx = channel_names.index("footfall")

# pick the sequence with the most crime in its input window, so the plot
# is actually informative rather than showing an all-empty patch
crime_totals = X[..., crime_idx].sum(axis=(1, 2, 3))
sample_idx = int(np.argmax(crime_totals))

sample_X = X[sample_idx]   # shape (T, 16, 16, C)
sample_y = y[sample_idx]   # shape (16, 16)
T = sample_X.shape[0]

n_frames_to_show = min(T, 6)
frame_indices = np.linspace(0, T - 1, n_frames_to_show).astype(int)

fig, axes = plt.subplots(2, n_frames_to_show, figsize=(3 * n_frames_to_show, 6))

for col, t in enumerate(frame_indices):
    axes[0, col].imshow(sample_X[t, :, :, crime_idx], cmap="Reds", vmin=0, vmax=1)
    axes[0, col].set_title(f"Crime, frame t-{T - t}")
    axes[0, col].axis("off")

    axes[1, col].imshow(sample_X[t, :, :, footfall_idx], cmap="Blues", vmin=0, vmax=1)
    axes[1, col].set_title(f"Footfall, frame t-{T - t}")
    axes[1, col].axis("off")

fig.suptitle(f"Example training sequence ({LOOKBACK_TO_INSPECT} look-back)\n"
             f"Top row = crime channel over time | Bottom row = footfall channel over time")
plt.tight_layout()
plt.savefig(f"{cfg.PHASE2_OUTPUTS_DIR}/example_sequence_frames.png", dpi=150)
print(f"Saved {cfg.PHASE2_OUTPUTS_DIR}/example_sequence_frames.png")

# separate plot: target frame (what the model should predict)
fig2, ax = plt.subplots(figsize=(5, 5))
ax.imshow(sample_y, cmap="Reds", vmin=0, vmax=1)
ax.set_title("Target frame (crime at time t, to be predicted)")
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{cfg.PHASE2_OUTPUTS_DIR}/example_target_frame.png", dpi=150)
print(f"Saved {cfg.PHASE2_OUTPUTS_DIR}/example_target_frame.png")

print(f"\nThis sequence's input window contains "
      f"{int(sample_X[..., crime_idx].sum())} total crime events, "
      f"and the target frame contains {int(sample_y.sum())} crime cell(s).")
