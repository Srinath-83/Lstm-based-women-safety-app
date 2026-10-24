# Phase 3 — Training and Comparing the Models

## What this does

This is the heart of your project: training the same 4 models the base
paper compares (Logistic Regression, Random Forest, plain LSTM,
ConvLSTM), and evaluating them the same way (standard + nearest-neighbor
Recall/Precision/F1).

### Files

| File | What it does |
|---|---|
| `config_phase3.py` | All settings (which look-back to use, model sizes, epochs, etc.) |
| `models.py` | PyTorch model definitions: `ConvLSTM` (main model) and `SimpleLSTM` (baseline) |
| `metrics.py` | Standard + nearest-neighbor (NN) Recall/Precision/F1, matching the paper's Eq. 7-8 |
| `data_utils.py` | Reshapes Phase 2's sequences for each model type (see below) |
| `08_train_baselines.py` | Trains Logistic Regression + Random Forest (scikit-learn) |
| `09_train_lstm.py` | Trains the plain LSTM baseline (PyTorch) |
| `10_train_convlstm.py` | Trains the main ConvLSTM model (PyTorch) |
| `11_compare_models.py` | Combines all 4 results into a table + comparison chart |

## An important design choice: why two different data "shapes"

- **ConvLSTM** gets the **full 16×16 grid** per sequence — it uses
  convolutions to look at neighboring cells itself.
- **LR, RF, and plain LSTM** have **no spatial awareness** at all — so we
  "explode" each 16×16 sequence into 256 independent per-cell samples.
  Each one only sees its own cell's history (crime, footfall, area type,
  etc. over time), never its neighbors.

This exploding step happens in `data_utils.py` (`explode_to_percell`).
It's exactly what makes the final comparison meaningful: if ConvLSTM
beats the others, it's because spatial awareness genuinely helped — not
because it just had access to different information.

## An important fix you should understand: class imbalance

Crime is rare (~1.4% of cells/blocks have any crime). Without correcting
for this, all the deep models will "cheat" by predicting **zero crime
everywhere** — since that's right 98.6% of the time and minimizes the
loss function trivially. You'll see this yourself if you remove the
fix and retrain: recall drops straight to 0.0.

- **LR/RF**: fixed via `class_weight="balanced"` in scikit-learn
- **LSTM/ConvLSTM**: fixed via `pos_weight` in `BCEWithLogitsLoss` —
  this tells PyTorch to penalize a missed crime (false negative) much
  more heavily than a false alarm. Both training scripts compute this
  automatically from your training data's actual class balance and
  print it out (e.g. `pos_weight=70.7` means "treat one missed crime as
  costly as 70.7 false alarms").

This is the single most important concept to explain confidently in your
viva if asked "why does the model predict anything at all, given how
rare crime is in your data?"

## How to run

```bash
cd phase3
pip install torch scikit-learn pandas matplotlib
python3 08_train_baselines.py
python3 09_train_lstm.py
python3 10_train_convlstm.py
python3 11_compare_models.py
```

Make sure `phase2/outputs/2_days/` (or `14_days/`) exists first — set
which one to use via `LOOKBACK_TO_USE` in `config_phase3.py`.

## ⚠️ About the settings in this repo — read before your final run

To keep this fast enough to test and iterate on a normal laptop CPU, the
default settings are intentionally **lighter** than the base paper's:

| Setting | Base paper | This project's default | Why |
|---|---|---|---|
| ConvLSTM hidden channels | 28 | 16 | Faster on CPU |
| ConvLSTM layers | 3 | 2 | Faster on CPU |
| Epochs | 200 | 10 | Faster for testing |
| Data | 5 years, full city | 220 days (quick-test), sub-grid crops | Faster for testing |

**In our own quick test run** (220 days of data, 10 epochs), ConvLSTM
actually scored *lower* than plain LSTM — the opposite of the paper's
finding. This is expected and NOT a bug: with so little data and so few
epochs, ConvLSTM's extra parameters (spatial convolutions) don't have
enough signal to learn from yet, so its advantage doesn't show up. The
paper's own result — ConvLSTM beating LSTM — depends on having enough
data and training time for that spatial-awareness advantage to actually
pay off.

**Before your final results for the report**, do this:
1. In `phase1/config.py` and `phase2/config_phase2.py`, make sure you're
   using the full dataset (`QUICK_TEST_MODE = False` in Phase 2's
   config, and keep `NUM_DAYS = 365*2` in Phase 1's config, or increase
   it further)
2. In `config_phase3.py`, increase `NUM_EPOCHS` (try 50-100),
   `CONVLSTM_HIDDEN_CHANNELS` (try 28, matching the paper), and
   `CONVLSTM_NUM_LAYERS` (try 3)
3. Re-run all four training scripts
4. If ConvLSTM still doesn't beat LSTM after this, that's a genuinely
   interesting, honestly-reported finding for your discussion section —
   the paper itself found this pattern varies somewhat by city and crime
   type, so a different result on Chennai's synthetic data isn't
   unreasonable, as long as you report it truthfully rather than
   adjusting the comparison to force a particular conclusion.

## What "good" results look like (matching the paper's pattern)

- **Standard precision will look bad** (often under 5%) for every
  model — this is expected given how rare crime is, and the paper
  reports the same pattern (their precision was also consistently
  below 10%).
- **NN (nearest-neighbor) metrics should look much better** than
  standard metrics — this is the paper's central point: even when a
  model doesn't hit the exact cell, it's usually "in the neighborhood."
- **Random Forest usually performs worst**, sometimes near zero — this
  matches the paper's own finding almost exactly (they describe RF as
  "failing to capture meaningful patterns").
- **Recall matters more than precision** for this use case — a missed
  crime (false negative) is more costly than a false alarm, which is
  exactly why we used `pos_weight`/`class_weight` in training.

## Next: Phase 4

Phase 4 turns these raw crime-probability predictions into your project's
actual differentiator: a **women's safety score** per grid cell, and the
safe-route logic built on top of it.
