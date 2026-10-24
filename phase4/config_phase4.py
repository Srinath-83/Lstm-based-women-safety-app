"""
config_phase4.py
-----------------
Settings for Phase 4: safety scoring and safe routing.
"""

import os
import sys

# so we can import models.py and config_phase3.py from Phase 3
PHASE3_DIR = "../phase3"
sys.path.append(PHASE3_DIR)

# so we can import config_phase2.py (for LOOKBACK_PERIODS) from Phase 2
PHASE2_DIR = "../phase2"
sys.path.append(PHASE2_DIR)

PHASE2_OUTPUTS_DIR = "../phase2/outputs"
PHASE1_OUTPUTS_DIR = "../phase1/outputs"
PHASE3_OUTPUTS_DIR = "../phase3/outputs"
PHASE4_OUTPUTS_DIR = "outputs"
os.makedirs(PHASE4_OUTPUTS_DIR, exist_ok=True)

# Which trained model to use for generating the safety map.
# "convlstm" (recommended -- it's your main, spatially-aware model) or "lstm"
MODEL_TO_USE = "convlstm"

# ---------------------------------------------------------------------------
# Safety score formula weights (must sum to 1.0)
# ---------------------------------------------------------------------------
# safety_score = 100 * (1 - risk), where:
#   risk = W_CRIME * predicted_crime_probability
#        + W_LIGHT * (1 - streetlight_score)
#        + W_POLICE * normalized_distance_to_police
#
# These weights are a documented ASSUMPTION -- grounded in Routine Activity
# Theory (crime risk + lack of "capable guardianship" = lighting + police
# presence), but not fitted from real data. State this clearly in your
# report. Feel free to tune them and see how the resulting map changes.
W_CRIME = 0.5
W_LIGHT = 0.3
W_POLICE = 0.2

assert abs(W_CRIME + W_LIGHT + W_POLICE - 1.0) < 1e-6, \
    "Safety score weights must sum to 1.0"

# ---------------------------------------------------------------------------
# Routing settings
# ---------------------------------------------------------------------------
# How strongly the router avoids low-safety cells vs. just taking the
# shortest path. 0 = ignore safety entirely (pure shortest path).
# Higher = more willing to take a longer path to stay in safer cells.
SAFETY_AVOIDANCE_STRENGTH = 3.0

RANDOM_SEED = 42
