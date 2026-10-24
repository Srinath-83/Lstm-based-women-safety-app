"""
config_phase2.py
-----------------
Constants for Phase 2 (tensor building + sequence generation).
Mirrors the terminology used in the base paper (Albors Zumel et al., 2025)
so it's easy to cross-reference: T = look-back period, M = sub-grid size.
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative to wherever you run the phase2 scripts from)
# ---------------------------------------------------------------------------
PHASE1_OUTPUTS_DIR = "../phase1/outputs"   # adjust if your folders differ
PHASE2_OUTPUTS_DIR = "outputs"
os.makedirs(PHASE2_OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Look-back (T) periods, in number of 12-hour blocks.
# The base paper tested T=28 (14 days) and T=4 (2 days). We do the same.
# ---------------------------------------------------------------------------
LOOKBACK_PERIODS = {
    "14_days": 28,
    "2_days": 4,
}

# ---------------------------------------------------------------------------
# Sub-grid sampling (like the paper's M x M random crops, M=16)
# ---------------------------------------------------------------------------
SUBGRID_SIZE = 16          # M: each training sample is a 16x16 patch of the city
SAMPLES_PER_SEQUENCE = 5   # how many random subgrids to draw per time window
MIN_CRIMES_IN_SUBGRID = 2  # discard subgrid-sequences with fewer than this
                            # many crimes in the input window (paper's rule,
                            # keeps the dataset from being 100% empty cells)
MAX_SUBGRID_ATTEMPTS = 30  # give up trying to find a "busy enough" crop after this

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.9   # chronological split, like the paper (90/10)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# For a quick local test run before scaling up to the full dataset
# ---------------------------------------------------------------------------
QUICK_TEST_MODE = True     # set False once you're ready to build the full dataset
QUICK_TEST_MAX_DAYS = 220  # only use this many days of data when testing
                            # (needs to be large enough that even the 14-day
                            # look-back (T=28) has enough test-set timesteps
                            # left over after the 90/10 split)
