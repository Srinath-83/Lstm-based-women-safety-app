"""
08_train_baselines.py
----------------------
Trains the two "traditional ML" baselines from the base paper:
  - Logistic Regression (LR)
  - Random Forest (RF)

Both are given the SAME information as the deep models (crime history +
footfall + area type + streetlights + distance to police + sociodemographics
over the look-back window), but flattened into a single feature vector per
grid cell, with no awareness of neighboring cells' spatial patterns. This
is exactly the point of comparison the paper makes: does spatial awareness
(ConvLSTM) actually help, or could a simpler model do just as well?

Run this AFTER Phase 2's 06_build_sequences.py.
"""

import time
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import config_phase3 as cfg
from data_utils import load_sequences, explode_to_percell, flatten_for_sklearn
from metrics import evaluate_predictions

np.random.seed(cfg.RANDOM_SEED)


def reshape_grid_preds(y_flat_pred, n_seq, H, W):
    """Turn per-cell predictions back into (n_seq, H, W) grids for the
    NN (nearest-neighbor) metric, which needs real spatial layout."""
    return y_flat_pred.reshape(n_seq, H, W)


def main():
    print(f"Using look-back dataset: {cfg.LOOKBACK_TO_USE}")
    X_train, y_train = load_sequences("train")
    X_test, y_test = load_sequences("test")
    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

    n_seq_test, _, H, W, _ = X_test.shape

    X_train_cells, y_train_cells = explode_to_percell(X_train, y_train)
    X_test_cells, y_test_cells = explode_to_percell(X_test, y_test)
    print(f"Exploded to per-cell samples: train={X_train_cells.shape}, "
          f"test={X_test_cells.shape}")

    X_train_flat = flatten_for_sklearn(X_train_cells)
    X_test_flat = flatten_for_sklearn(X_test_cells)

    results = {}

    # -------------------------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------------------------
    print("\nTraining Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(
        max_iter=400, class_weight="balanced"
    )
    lr.fit(X_train_flat, y_train_cells)
    print(f"  done in {time.time() - t0:.1f}s")

    lr_probs_flat = lr.predict_proba(X_test_flat)[:, 1]
    lr_probs_grid = reshape_grid_preds(lr_probs_flat, n_seq_test, H, W)
    y_test_grid = y_test  # already (n_seq, H, W)

    results["LogisticRegression"] = evaluate_predictions(
        y_test_grid, lr_probs_grid, threshold=cfg.CLASSIFICATION_THRESHOLD
    )
    print(f"  LR results: {results['LogisticRegression']}")

    # -------------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------------
    print("\nTraining Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        criterion="entropy", n_jobs=-1, random_state=cfg.RANDOM_SEED
    )

    if len(X_train_flat) > cfg.RF_MAX_TRAIN_SAMPLES:
        print(f"  Subsampling RF training set from {len(X_train_flat)} "
              f"to {cfg.RF_MAX_TRAIN_SAMPLES} rows (keeps training time "
              f"reasonable; LR and deep models still use the full data)")
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        idx = rng.choice(len(X_train_flat), cfg.RF_MAX_TRAIN_SAMPLES, replace=False)
        rf.fit(X_train_flat[idx], y_train_cells[idx])
    else:
        rf.fit(X_train_flat, y_train_cells)
    print(f"  done in {time.time() - t0:.1f}s")

    rf_probs_flat = rf.predict_proba(X_test_flat)[:, 1]
    rf_probs_grid = reshape_grid_preds(rf_probs_flat, n_seq_test, H, W)

    results["RandomForest"] = evaluate_predictions(
        y_test_grid, rf_probs_grid, threshold=cfg.CLASSIFICATION_THRESHOLD
    )
    print(f"  RF results: {results['RandomForest']}")

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    with open(f"{cfg.PHASE3_OUTPUTS_DIR}/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {cfg.PHASE3_OUTPUTS_DIR}/baseline_results.json")


if __name__ == "__main__":
    main()
