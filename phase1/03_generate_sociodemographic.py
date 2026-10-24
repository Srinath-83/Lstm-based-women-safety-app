"""
03_generate_sociodemographic.py
--------------------------------
Generates a lightweight sociodemographic layer per cell: population
density, literacy rate, and a poverty-proxy income band. In the base
paper, this came from the US Census (American Community Survey). For
Chennai, the real equivalent is Census of India / TN government ward
data -- for now we simulate it, correlated loosely with area_type, so
Phase 3's model has a third feature layer to learn from (like the paper's
"S" feature set).

Run this AFTER 01_build_grid.py.
"""

import numpy as np
import pandas as pd
import config as cfg

np.random.seed(cfg.RANDOM_SEED + 1)

# Rough population density bands (people per sq km) by area type --
# illustrative, not sourced from Census yet.
DENSITY_RANGE = {
    "residential":   (12000, 28000),
    "commercial":    (4000, 9000),
    "market":        (6000, 12000),
    "bus_stand":     (3000, 7000),
    "entertainment": (3000, 6000),
    "educational":   (2000, 5000),
    "industrial":    (1000, 3000),
    "hospital":      (2000, 4000),
}

LITERACY_RANGE = {
    "residential":   (78, 94),
    "commercial":    (85, 96),
    "market":        (72, 88),
    "bus_stand":     (70, 85),
    "entertainment": (80, 95),
    "educational":   (88, 98),
    "industrial":    (68, 85),
    "hospital":      (85, 96),
}


def generate_sociodemographic(grid):
    rows = []
    for _, cell in grid.iterrows():
        lo, hi = DENSITY_RANGE[cell["area_type"]]
        pop_density = np.random.uniform(lo, hi)

        lo_l, hi_l = LITERACY_RANGE[cell["area_type"]]
        literacy_rate = np.random.uniform(lo_l, hi_l)

        # income band proxy, 1 (low) - 5 (high), loosely tied to literacy
        income_band = int(np.clip(np.round((literacy_rate - 65) / 7), 1, 5))

        rows.append({
            "cell_id": cell["cell_id"],
            "population_density_per_sqkm": round(pop_density, 1),
            "literacy_rate_pct": round(literacy_rate, 1),
            "income_band": income_band,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    grid = pd.read_csv("outputs/grid_cells.csv")
    sociodemo = generate_sociodemographic(grid)

    os.makedirs("outputs", exist_ok=True)
    sociodemo.to_csv("outputs/sociodemographic.csv", index=False)
    print(f"Saved outputs/sociodemographic.csv ({len(sociodemo)} rows)")
    print(sociodemo.describe())
