"""Generate FAKE data shaped like UCI / HPDmobile so you can test that the
pipeline runs before the real files are in place. Never use for results.

Usage:  python scripts/make_dummy_data.py
        python run_pipeline.py configs/smoke_test.yaml
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
out = Path("data/dummy")
out.mkdir(parents=True, exist_ok=True)


def make(n, shift, names):
    occ = rng.integers(0, 2, n)
    df = pd.DataFrame({
        names[0]: 21 + shift + occ * 1.0 + rng.normal(0, 0.8, n),
        names[1]: 27 + shift * 3 + rng.normal(0, 4, n),
        names[2]: occ * 400 + shift * 50 + rng.normal(0, 60, n).clip(-50),
        names[3]: 450 + occ * 300 + shift * 100 + rng.normal(0, 80, n),
        names[4]: occ,
    })
    return df


uci = ["Temperature", "Humidity", "Light", "CO2", "Occupancy"]
hpd = ["temperature", "rh", "light", "co2eq", "occupied"]
make(2000, 0, uci).to_csv(out / "source_train.csv", index=False)
make(600, 0, uci).to_csv(out / "source_test.csv", index=False)
make(1500, 1.5, hpd).to_csv(out / "target.csv", index=False)
print(f"Fake data written to {out}/")
