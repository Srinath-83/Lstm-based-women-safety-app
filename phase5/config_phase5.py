"""
config_phase5.py
-----------------
Settings for Phase 5: real road-network download + offline safe routing.
"""

import os

# so we can reuse Phase 1's exact grid-indexing math (LAT_MIN, LON_MIN,
# CELL_SIZE_DEG_LAT/LON) to map road-network nodes to grid cells
import sys
PHASE1_DIR = "../phase1"
sys.path.append(PHASE1_DIR)

CITY_QUERY = "Chennai, Tamil Nadu, India"   # not used by default anymore --
    # 15_download_road_network.py now downloads by BOUNDING BOX (matching
    # Phase 1's grid exactly) instead of by place name, for cleaner
    # alignment. Kept here in case you want to experiment with a
    # place-name query instead (e.g. "Greater Chennai Corporation, Tamil
    # Nadu, India" to match the administrative boundary used by some of
    # the real crime-data sources we found earlier).
NETWORK_TYPE = "walk"    # "walk" = pedestrian paths+roads (best fit for a
                          # women's-safety walking-route app). Use "drive"
                          # instead if you want car routing.

PHASE1_OUTPUTS_DIR = "../phase1/outputs"
PHASE4_OUTPUTS_DIR = "../phase4/outputs"
PHASE5_OUTPUTS_DIR = "outputs"
os.makedirs(PHASE5_OUTPUTS_DIR, exist_ok=True)

# Local files that make this phase work OFFLINE after the first download.
# Once these exist, 16_ and 17_ never need the internet again.
ROAD_NETWORK_FILE = f"{PHASE5_OUTPUTS_DIR}/chennai_road_network.graphml"
POI_FILE = f"{PHASE5_OUTPUTS_DIR}/hospitals_police.csv"
ROAD_NETWORK_WITH_SAFETY_FILE = f"{PHASE5_OUTPUTS_DIR}/road_network_with_safety.graphml"

# Same idea as Phase 4's SAFETY_AVOIDANCE_STRENGTH: how much extra
# "cost" a routing algorithm should assign to walking through unsafe
# areas, relative to just minimizing distance. 0 = ignore safety
# entirely (pure shortest path).
SAFETY_AVOIDANCE_STRENGTH = 3.0

RANDOM_SEED = 42
