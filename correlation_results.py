"""Correlation matrix of the evaluation results (all policies x 5 seeds), isolated and priority modes.
    python correlation_results.py
Reads results/test_results_15min_seed*.csv (isolated) and results/test_results_15min_seed*_prio.csv (priority),
writes results/correlation_results_isolated.png/.csv and results/correlation_results_priority.png/.csv.

Each row is one policy in one seed (test days 151-180). Static policies do not change between seeds, so most of the
spread comes from differences between policies. Read the matrix as 'which metrics move together across policies',
not as a causal result. W (delay) is left out because W = Wq + 1/mu is almost the same as Wq.
"""
import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import RESULTS_DIR

COLS = ["PB_URLLC", "PB_eMBB", "PB_mMTC", "Wq_URLLC", "Wq_eMBB", "Wq_mMTC", "Lq_URLLC", "Lq_eMBB", "Lq_mMTC",
        "throughput", "util", "moves_per_day", "return_per_day"]
NAMES = ["P_B URLLC", "P_B eMBB", "P_B mMTC", "Wq URLLC", "Wq eMBB", "Wq mMTC", "Lq URLLC", "Lq eMBB", "Lq mMTC",
         "Throughput", "Utilisation", "Moves/day", "Reward/day"]


def load(prio):
    files = sorted(f for f in glob.glob(str(RESULTS_DIR / "test_results_15min_seed*.csv"))
                   if f.endswith("_prio.csv") == prio and re.search(r"seed\d+(_prio)?\.csv$", f))
    return pd.concat([pd.read_csv(f).assign(seed=f) for f in files], ignore_index=True), len(files)


for prio, label in ((False, "isolated"), (True, "priority")):
    d, n_files = load(prio)
    if d.empty:
        print(f"{label}: no test_results files found in {RESULTS_DIR}")
        continue
    c = d[COLS].corr().round(2)
    c.to_csv(RESULTS_DIR / f"correlation_results_{label}.csv")
    k = len(COLS)
    fig, ax = plt.subplots(figsize=(9.6, 8))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(k)); ax.set_xticklabels(NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(k)); ax.set_yticklabels(NAMES, fontsize=9)
    for i in range(k):
        for j in range(k):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > 0.6 else "black")
    ax.set_title(f"Correlation of evaluation metrics, {label} mode\n(4 policies x {n_files} seeds = {len(d)} runs, test days 151-180)", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"correlation_results_{label}.png", dpi=170); plt.close()
    print(f"\n{label}: {len(d)} runs\n{c[['moves_per_day', 'return_per_day']]}")
    print(f"Saved results/correlation_results_{label}.png and .csv")
