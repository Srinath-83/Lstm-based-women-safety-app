"""
metrics.py
----------
Evaluation metrics, following the base paper's methodology exactly:

  - Standard Recall / Precision / F1 (cell-by-cell exact match)
  - "NN" (nearest-neighbor) modified Recall / Precision / F1: a predicted
    crime is also counted as correct if a real crime happened in an
    ADJACENT cell (the paper's justification: crime spills over into
    nearby areas, and a police officer could still reasonably respond to
    a neighboring cell in a few minutes given how small each cell is).

These matter a lot for your project because, just like in the paper,
raw precision will look very low (crime is rare, so most positive
predictions will be "wrong" in the strict sense) -- the NN metric shows
the model is still spatially "in the right neighborhood" even when it
misses the exact cell.
"""

import numpy as np


def _confusion_counts(y_true, y_pred):
    """y_true, y_pred: same-shape binary arrays (any shape)."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()
    tn = np.logical_and(~y_true, ~y_pred).sum()
    return tp, fp, fn, tn


def standard_metrics(y_true, y_pred):
    """Plain recall/precision/F1 over flattened predictions."""
    tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"recall": recall, "precision": precision, "f1": f1}


def _has_neighbor_positive(true_grid, r, c):
    """Check the 8 neighboring cells (Chebyshev distance 1) for a true crime."""
    n_rows, n_cols = true_grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < n_rows and 0 <= cc < n_cols and true_grid[rr, cc] == 1:
                return True
    return False


def nn_metrics(y_true, y_pred):
    """
    Nearest-neighbor modified recall/precision/F1, exactly matching the
    base paper's definition (their Eq. 7-8): a false positive still
    "counts" as a hit if one of its 8 neighboring cells actually had a
    crime. y_true, y_pred are expected as arrays of shape
    (n_samples, H, W) -- i.e. one HxW grid per sample.
    """
    tp_total = fp_nn_total = fp_total = fn_total = 0

    for i in range(y_true.shape[0]):
        true_grid = y_true[i]
        pred_grid = y_pred[i]
        n_rows, n_cols = true_grid.shape

        for r in range(n_rows):
            for c in range(n_cols):
                t, p = true_grid[r, c], pred_grid[r, c]
                if t == 1 and p == 1:
                    tp_total += 1
                elif t == 1 and p == 0:
                    fn_total += 1
                elif t == 0 and p == 1:
                    if _has_neighbor_positive(true_grid, r, c):
                        fp_nn_total += 1   # "forgivable" false positive
                    else:
                        fp_total += 1

    recall_nn = (tp_total + fp_nn_total) / (tp_total + fp_nn_total + fn_total) \
        if (tp_total + fp_nn_total + fn_total) > 0 else 0.0
    precision_nn = (tp_total + fp_nn_total) / (tp_total + fp_nn_total + fp_total) \
        if (tp_total + fp_nn_total + fp_total) > 0 else 0.0
    f1_nn = (2 * precision_nn * recall_nn / (precision_nn + recall_nn)
             if (precision_nn + recall_nn) > 0 else 0.0)

    return {"recall_nn": recall_nn, "precision_nn": precision_nn, "f1_nn": f1_nn}


def evaluate_predictions(y_true, y_pred_probs, threshold=0.5):
    """
    Convenience wrapper: takes probability predictions, thresholds them,
    and returns both standard and NN metrics in one dict.
    y_true, y_pred_probs: shape (n_samples, H, W)
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    y_true = y_true.astype(int)

    results = standard_metrics(y_true, y_pred)
    results.update(nn_metrics(y_true, y_pred))
    return results
