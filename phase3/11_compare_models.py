"""
11_compare_models.py
----------------------
Loads the saved results from all four models (Logistic Regression, Random
Forest, plain LSTM, ConvLSTM) and produces:
  1. A summary table printed to console + saved as CSV
  2. A bar chart comparing Recall / Precision / F1 (standard vs. NN)
     across all four models -- the same comparison style as the base
     paper's Figure 5.

Run this AFTER 08, 09, and 10 have all been run (so their _results.json
files exist in outputs/).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config_phase3 as cfg


def load_all_results():
    files = {
        "Logistic Regression": "baseline_results.json",  # LR + RF share this file
        "Random Forest": "baseline_results.json",
        "LSTM": "lstm_results.json",
        "ConvLSTM": "convlstm_results.json",
    }

    all_results = {}
    with open(f"{cfg.PHASE3_OUTPUTS_DIR}/baseline_results.json") as f:
        baseline = json.load(f)
    all_results["Logistic Regression"] = baseline["LogisticRegression"]
    all_results["Random Forest"] = baseline["RandomForest"]

    with open(f"{cfg.PHASE3_OUTPUTS_DIR}/lstm_results.json") as f:
        all_results["LSTM"] = json.load(f)

    with open(f"{cfg.PHASE3_OUTPUTS_DIR}/convlstm_results.json") as f:
        all_results["ConvLSTM"] = json.load(f)

    return all_results


def main():
    results = load_all_results()

    df = pd.DataFrame(results).T
    df = df[["recall", "precision", "f1", "recall_nn", "precision_nn", "f1_nn"]]
    df.columns = ["Recall", "Precision", "F1", "Recall (NN)", "Precision (NN)", "F1 (NN)"]

    print(f"\n=== Model comparison (look-back = {cfg.LOOKBACK_TO_USE}) ===\n")
    print(df.round(4).to_string())

    df.to_csv(f"{cfg.PHASE3_OUTPUTS_DIR}/model_comparison.csv")
    print(f"\nSaved {cfg.PHASE3_OUTPUTS_DIR}/model_comparison.csv")

    # --- Plot: standard vs NN metrics, grouped like the paper's Fig 5 ------
    models = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_pairs = [("recall", "recall_nn", "Recall"),
                     ("precision", "precision_nn", "Precision"),
                     ("f1", "f1_nn", "F1 Score")]

    x = np.arange(len(models))
    width = 0.35

    for ax, (std_key, nn_key, title) in zip(axes, metric_pairs):
        std_vals = [results[m][std_key] for m in models]
        nn_vals = [results[m][nn_key] for m in models]

        ax.bar(x - width / 2, std_vals, width, label="Standard", color="#4C72B0")
        ax.bar(x + width / 2, nn_vals, width, label="Nearest-Neighbor", color="#DD8452")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.legend()

    fig.suptitle(f"Model Comparison — Chennai Synthetic Crime Data "
                 f"(look-back = {cfg.LOOKBACK_TO_USE})")
    plt.tight_layout()
    plt.savefig(f"{cfg.PHASE3_OUTPUTS_DIR}/model_comparison.png", dpi=150)
    print(f"Saved {cfg.PHASE3_OUTPUTS_DIR}/model_comparison.png")


if __name__ == "__main__":
    main()
