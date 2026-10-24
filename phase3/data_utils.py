"""
data_utils.py
-------------
Shared helper functions for Phase 3: loading Phase 2's sequence data and
reshaping it for each model type.

Two different "shapes" are needed:

  1. ConvLSTM wants the FULL 16x16 grid per sequence (it uses convolutions
     to look at spatial neighbors itself):
         X: (n_sequences, T, 16, 16, C)   y: (n_sequences, 16, 16)

  2. The baselines (Logistic Regression, Random Forest, plain LSTM) don't
     understand space at all -- they only look at one cell's own history.
     So we "explode" each 16x16 sequence into 256 independent per-cell
     samples:
         X: (n_sequences * 256, T, C)     y: (n_sequences * 256,)
     LR/RF additionally flatten the (T, C) history into one long vector:
         X: (n_sequences * 256, T * C)

This mirrors the base paper's design intent: LR/RF/plain-LSTM have no
spatial awareness, while ConvLSTM does -- exploding into per-cell samples
for the baselines is what makes that comparison meaningful.
"""

import numpy as np
import config_phase3 as cfg


def load_sequences(split):
    """split: 'train' or 'test'"""
    X = np.load(f"{cfg.PHASE2_OUTPUTS_DIR}/X_{split}.npy")
    y = np.load(f"{cfg.PHASE2_OUTPUTS_DIR}/y_{split}.npy")
    return X, y


def explode_to_percell(X, y):
    """
    X: (n_seq, T, H, W, C) -> (n_seq * H * W, T, C)
    y: (n_seq, H, W)       -> (n_seq * H * W,)
    """
    n_seq, T, H, W, C = X.shape
    # move H, W next to n_seq, then flatten all three into one sample axis
    X_cells = X.transpose(0, 2, 3, 1, 4).reshape(n_seq * H * W, T, C)
    y_cells = y.reshape(n_seq * H * W)
    return X_cells, y_cells


def flatten_for_sklearn(X_cells):
    """(n_samples, T, C) -> (n_samples, T*C) for LR/RF."""
    n_samples, T, C = X_cells.shape
    return X_cells.reshape(n_samples, T * C)
