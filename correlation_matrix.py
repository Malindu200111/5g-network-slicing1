"""Correlation matrix of the synthetic Sri Lanka 5G traffic dataset (project root, like load_curves.py).
    python correlation_matrix.py            # hourly means (default), writes results/correlation_matrix.png and .csv
    python correlation_matrix.py --minute   # minute-level (noisier, 259,200 rows)

Variables: the three arrival rates plus calendar/pattern flags. Office-hours and night-batch flags come
from the generator design, so their high correlation with URLLC and mMTC is by construction (synthetic data).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import RESULTS_DIR

CSV = "data/processed/srilanka_5g_synthetic_traffic_6months.csv"
COLS = ["embb", "urllc", "mmtc", "office", "night", "weekend", "month"]
NAMES = ["eMBB rate", "URLLC rate", "mMTC rate", "Office hours\n(08-17)", "Night batch\n(00:30-03)", "Weekend", "Month index"]

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=CSV)
ap.add_argument("--minute", action="store_true", help="use minute-level rows instead of hourly means")
a = ap.parse_args()

d = pd.read_csv(a.csv)
d["hour"] = d.hour_of_day.astype(int)
d["office"] = ((d.hour >= 8) & (d.hour < 17)).astype(int)
d["night"] = ((d.hour_of_day >= 0.5) & (d.hour_of_day < 3)).astype(int)
d = d.rename(columns={"embb_arrival_rate": "embb", "urllc_arrival_rate": "urllc", "mmtc_arrival_rate": "mmtc",
                      "is_weekend": "weekend", "month_index": "month"})
if a.minute:
    t, unit = d, "minute"
else:
    t = d.groupby(["day_number", "hour"], as_index=False).agg(embb=("embb", "mean"), urllc=("urllc", "mean"), mmtc=("mmtc", "mean"),
                                                             office=("office", "max"), night=("night", "max"),
                                                             weekend=("weekend", "max"), month=("month", "max"))
    unit = "hourly"
c = t[COLS].corr().round(2)
print(f"Correlation matrix ({unit} data, n = {len(t):,})\n{c}")

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
suffix = "_minute" if a.minute else ""
c.to_csv(RESULTS_DIR / f"correlation_matrix{suffix}.csv")
fig, ax = plt.subplots(figsize=(8.2, 6.6))
im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(COLS))); ax.set_xticklabels(NAMES, rotation=40, ha="right", fontsize=10)
ax.set_yticks(range(len(COLS))); ax.set_yticklabels(NAMES, fontsize=10)
for i in range(len(COLS)):
    for j in range(len(COLS)):
        v = c.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white" if abs(v) > 0.6 else "black")
ax.set_title(f"Correlation matrix of the synthetic traffic dataset\n({unit} data, 180 days, n = {len(t):,})", fontsize=12)
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(RESULTS_DIR / f"correlation_matrix{suffix}.png", dpi=180); plt.close()
print(f"Saved results/correlation_matrix{suffix}.png and .csv")
