"""
config_phase3.py
-----------------
Settings for Phase 3: model training and evaluation.
"""

import os

# Which look-back dataset (from Phase 2) to train on.
# Options match the folder names created in phase2/outputs/: "2_days", "14_days"
LOOKBACK_TO_USE = "2_days"

PHASE2_OUTPUTS_DIR = f"../phase2/outputs/{LOOKBACK_TO_USE}"
PHASE3_OUTPUTS_DIR = "outputs"
os.makedirs(PHASE3_OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Training hyperparameters (kept modest so this trains fast on a CPU laptop;
# the base paper used a GPU and much larger data -- these are intentionally
# lighter for a student project, see README for how to scale up)
# ---------------------------------------------------------------------------
CONVLSTM_HIDDEN_CHANNELS = 16   # paper used 28; smaller = faster on CPU
CONVLSTM_NUM_LAYERS = 2         # paper used 3
CONVLSTM_KERNEL_SIZE = 3

LSTM_HIDDEN_SIZE = 32
LSTM_NUM_LAYERS = 2

LEARNING_RATE = 1e-3
BATCH_SIZE = 256        # kept large so CPU training stays fast (see README)
NUM_EPOCHS = 10
DROPOUT_RATE = 0.3

CLASSIFICATION_THRESHOLD = 0.5   # same as the paper (probability -> 0/1)

# Random Forest gets slow on the exploded per-cell dataset once you switch
# to the full (non-quick-test) data from Phase 2. If the training set has
# more rows than this, we randomly subsample down to this many rows just
# for RF (LR and the deep models still use the full data).
RF_MAX_TRAIN_SAMPLES = 300_000

RANDOM_SEED = 42
DEVICE = "cpu"   # change to "cuda" if you have a GPU available
