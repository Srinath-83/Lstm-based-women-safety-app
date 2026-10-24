"""
10_train_convlstm.py
----------------------
Trains the MAIN model of your project: ConvLSTM. Unlike the baselines,
this model sees the FULL 16x16 grid at every time step and uses
convolutions to learn spatial patterns (e.g. "crime spreads to
neighboring cells", "risk clusters around bus stands") on top of the
temporal patterns LSTM already captures.

This is the model you'll compare against LR, RF, and plain LSTM to show
that spatial awareness genuinely helps -- exactly the comparison the
base paper makes (their Figure 5/6/7).

Run this AFTER Phase 2's 06_build_sequences.py.
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config_phase3 as cfg
from data_utils import load_sequences
from models import ConvLSTM
from metrics import evaluate_predictions

torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
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
    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

    C = X_train.shape[-1]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = ConvLSTM(
        input_channels=C,
        hidden_channels=cfg.CONVLSTM_HIDDEN_CHANNELS,
        num_layers=cfg.CONVLSTM_NUM_LAYERS,
        kernel_size=cfg.CONVLSTM_KERNEL_SIZE,
        dropout=cfg.DROPOUT_RATE,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvLSTM has {n_params:,} trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # Same class-imbalance correction as the LSTM baseline (see that
    # script's comment for why this matters).
    n_pos = y_train.sum()
    n_neg = y_train.size - n_pos
    pos_weight_value = n_neg / max(n_pos, 1)
    print(f"Class balance: {n_pos:.0f} positive / {n_neg:.0f} negative "
          f"-> pos_weight={pos_weight_value:.1f}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )

    print(f"\nTraining ConvLSTM for {cfg.NUM_EPOCHS} epochs...")
    for epoch in range(cfg.NUM_EPOCHS):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Epoch {epoch + 1}/{cfg.NUM_EPOCHS}  loss={loss:.4f}  "
              f"({time.time() - t0:.1f}s)")

    print("\nEvaluating on test set...")
    test_probs = predict(model, test_loader, device)  # (n_seq, H, W)

    results = evaluate_predictions(
        y_test, test_probs, threshold=cfg.CLASSIFICATION_THRESHOLD
    )
    print(f"ConvLSTM results: {results}")

    with open(f"{cfg.PHASE3_OUTPUTS_DIR}/convlstm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), f"{cfg.PHASE3_OUTPUTS_DIR}/convlstm_model.pt")
    np.save(f"{cfg.PHASE3_OUTPUTS_DIR}/convlstm_test_probs.npy", test_probs)
    print(f"\nSaved {cfg.PHASE3_OUTPUTS_DIR}/convlstm_results.json, "
          f"convlstm_model.pt, convlstm_test_probs.npy")


if __name__ == "__main__":
    main()
