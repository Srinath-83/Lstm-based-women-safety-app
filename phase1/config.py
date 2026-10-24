"""
config.py
---------
All the constants for Phase 1 live here, so you can tweak the project
(city boundary, grid size, area-type mix, calibration numbers) from a
single place without touching the logic in the other scripts.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. CITY BOUNDARY (Chennai, approximate bounding box)
# ---------------------------------------------------------------------------
# NOTE: This is a rough rectangle around Chennai city limits, not the exact
# Greater Chennai Corporation boundary. It's good enough to start the
# project. If you later get an official shapefile/GeoJSON of Chennai's
# boundary, we can swap this out for a precise polygon.
CITY_NAME = "Chennai"
LAT_MIN, LAT_MAX = 12.83, 13.23   # south to north
LON_MIN, LON_MAX = 80.13, 80.31   # west to east

# ---------------------------------------------------------------------------
# 2. GRID CELL SIZE
# ---------------------------------------------------------------------------
# The base paper (Albors Zumel et al., 2025) used 0.077 sq. miles = 0.2 sq km
# per cell. We match that here. A cell is treated as a small square, so each
# side is sqrt(0.2) km long.
CELL_AREA_SQKM = 0.2
CELL_SIDE_KM = np.sqrt(CELL_AREA_SQKM)          # ~0.447 km per side

# Convert km to degrees (approximate, fine for city-scale projects):
KM_PER_DEG_LAT = 111.0
# longitude degrees get "compressed" the further you are from the equator
CITY_CENTER_LAT = (LAT_MIN + LAT_MAX) / 2
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(CITY_CENTER_LAT))

CELL_SIZE_DEG_LAT = CELL_SIDE_KM / KM_PER_DEG_LAT
CELL_SIZE_DEG_LON = CELL_SIDE_KM / KM_PER_DEG_LON

# ---------------------------------------------------------------------------
# 3. TIME WINDOW
# ---------------------------------------------------------------------------
# Like the base paper, we predict crime in 12-hour blocks: "day" (06:00-18:00)
# and "night" (18:00-06:00). Night is generally higher-risk for women's safety.
TIME_BLOCKS = ["day", "night"]   # day = 06:00-18:00, night = 18:00-06:00
NUM_DAYS = 365 * 2                # 2 years of synthetic data (~730 days)

# ---------------------------------------------------------------------------
# 4. AREA TYPES
# ---------------------------------------------------------------------------
# Each grid cell gets assigned exactly one dominant "area type". This drives
# footfall patterns and baseline crime risk. Weights = how common each type
# is across the city (roughly reflecting a typical Indian metro's mix).
AREA_TYPES = [
    "residential",
    "commercial",
    "market",
    "bus_stand",
    "entertainment",   # bars, pubs, late-night eateries
    "educational",
    "industrial",
    "hospital",
]

AREA_TYPE_WEIGHTS = {
    "residential":   0.42,
    "commercial":    0.18,
    "market":        0.10,
    "bus_stand":     0.04,
    "entertainment": 0.06,
    "educational":   0.08,
    "industrial":    0.07,
    "hospital":      0.05,
}

# Relative baseline crime-risk multiplier per area type (1.0 = average).
# These are illustrative assumptions grounded loosely in routine activity
# theory (high footfall + low guardianship = higher risk), NOT real
# calibrated police statistics. Document this clearly as an assumption
# in your report/viva.
AREA_TYPE_RISK_MULTIPLIER = {
    "residential":   0.8,
    "commercial":    1.1,
    "market":        1.3,
    "bus_stand":     1.6,
    "entertainment": 1.8,
    "educational":   0.9,
    "industrial":    1.2,
    "hospital":      0.7,
}

# ---------------------------------------------------------------------------
# 5. CALIBRATION TARGET (tie synthetic data to a real, citable number)
# ---------------------------------------------------------------------------
# NCRB / TN Police reports put Chennai city's total cognizable crimes at
# roughly this order of magnitude per year (adjust this constant once you
# pull the exact figure from the NCRB "Crime in India" report or the TN
# Police / OpenCity dataset for your target year -- this is what makes your
# synthetic data "calibrated" rather than arbitrary).
TARGET_ANNUAL_CRIMES_CITYWIDE = 45000   # <-- placeholder, replace with a cited figure

# Fraction of crimes we treat as specifically relevant to women's safety
# (harassment, assault, stalking, etc.) -- again, replace with a cited
# NCRB "crimes against women" share for Tamil Nadu when you have it.
WOMEN_SAFETY_CRIME_SHARE = 0.28

# ---------------------------------------------------------------------------
# 6. RANDOM SEED
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
