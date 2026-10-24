"""
09_train_lstm.py
------------------
Trains the "plain LSTM" baseline from the base paper: an LSTM that
processes each grid cell's own time history independently (no
convolution, so it has no idea what's happening in neighboring cells).
This is the deep-learning baseline that ConvLSTM needs to beat in order
to justify the extra architectural complexity.

Run this AFTER Phase 2's 06_build_sequences.py.
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config_phase3 as cfg
from data_utils import load_sequences, explode_to_percell
from models import SimpleLSTM
from metrics import evaluate_predictions

torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    """Runs the model and converts its raw logits into 0-1 probabilities."""
    model.eval()
    all_preds = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
    return np.concatenate(all_preds)


def main():
    device = torch.device(cfg.DEVICE)
    print(f"Using look-back dataset: {cfg.LOOKBACK_TO_USE}, device: {device}")

    X_train, y_train = load_sequences("train")
    X_test, y_test = load_sequences("test")
    n_seq_test, T, H, W, C = X_test.shape

    X_train_cells, y_train_cells = explode_to_percell(X_train, y_train)
    X_test_cells, y_test_cells = explode_to_percell(X_test, y_test)
    print(f"Per-cell samples: train={X_train_cells.shape}, test={X_test_cells.shape}")

    train_ds = TensorDataset(
        torch.tensor(X_train_cells, dtype=torch.float32),
        torch.tensor(y_train_cells, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_cells, dtype=torch.float32),
        torch.tensor(y_test_cells, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = SimpleLSTM(
        input_channels=C,
        hidden_size=cfg.LSTM_HIDDEN_SIZE,
        num_layers=cfg.LSTM_NUM_LAYERS,
        dropout=cfg.DROPOUT_RATE,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # Crime is rare (~1-2% positive). Without correcting for this, the
    # model can minimize loss just by predicting "no crime" everywhere.
    # pos_weight tells BCEWithLogitsLoss to penalize missed crimes (false
    # negatives) more heavily -- the deep-learning equivalent of sklearn's
    # class_weight="balanced" used for the LR/RF baselines.
    n_pos = y_train_cells.sum()
    n_neg = len(y_train_cells) - n_pos
    pos_weight_value = n_neg / max(n_pos, 1)
    print(f"Class balance: {n_pos:.0f} positive / {n_neg:.0f} negative "
          f"-> pos_weight={pos_weight_value:.1f}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )

    print(f"\nTraining SimpleLSTM for {cfg.NUM_EPOCHS} epochs...")
    for epoch in range(cfg.NUM_EPOCHS):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Epoch {epoch + 1}/{cfg.NUM_EPOCHS}  loss={loss:.4f}  "
              f"({time.time() - t0:.1f}s)")

    print("\nEvaluating on test set...")
    test_probs_flat = predict(model, test_loader, device)
    test_probs_grid = test_probs_flat.reshape(n_seq_test, H, W)

    results = evaluate_predictions(
        y_test, test_probs_grid, threshold=cfg.CLASSIFICATION_THRESHOLD
    )
    print(f"LSTM results: {results}")

    with open(f"{cfg.PHASE3_OUTPUTS_DIR}/lstm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), f"{cfg.PHASE3_OUTPUTS_DIR}/lstm_model.pt")
    print(f"\nSaved {cfg.PHASE3_OUTPUTS_DIR}/lstm_results.json and lstm_model.pt")


if __name__ == "__main__":
    main()
