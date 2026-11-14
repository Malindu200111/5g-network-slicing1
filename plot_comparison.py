"""
Comparative bar charts (deliverable 5): every policy on the TEST days, mean +/- std over the available seeds.
    python plot_comparison.py                 # reads results/test_results_15min_seed*.csv  (DQN runs of main.py)
Writes results/comparison_bars.png and results/comparison_table.csv.
"""
import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SLICES

ap = argparse.ArgumentParser()
ap.add_argument("--priority", action="store_true", help="use the runs of main.py --priority (files ending _prio.csv)")
suffix = "_prio" if ap.parse_args().priority else ""
files = sorted(f for f in glob.glob(str(RESULTS_DIR / "test_results_15min_seed*.csv"))
               if f.endswith("_prio.csv") == bool(suffix))
if not files:
    raise SystemExit("no results/test_results_15min_seed*.csv found - run main.py first")
df = pd.concat([pd.read_csv(f) for f in files])
df["policy"] = df["policy"].str.replace(r"DQN.*", "DQN", regex=True)
order = [p for p in ["Static (mean)", "Static (peak)", "Proportional", "DQN"] if p in set(df.policy)]
g = df.groupby("policy")
mean, std = g.mean(numeric_only=True).loc[order], g.std(numeric_only=True).fillna(0.0).loc[order]
mean.round(4).to_csv(RESULTS_DIR / f"comparison_table{suffix}.csv")

panels = [("Packet loss (blocking) per slice", [f"PB_{s}" for s in SLICES], "P_B"),
          ("Mean waiting time Wq per slice", [f"Wq_{s}" for s in SLICES], "seconds"),
          ("Mean latency W per slice", [f"W_{s}" for s in SLICES], "seconds"),
          ("Mean queue length Lq per slice", [f"Lq_{s}" for s in SLICES], "packets"),
          ("Throughput", ["throughput"], "packets served"),
          ("Resource utilisation", ["util"], "served / capacity"),
          ("Moves per day", ["moves_per_day"], "moves"),
          ("Return per day (higher is better)", ["return_per_day"], "reward")]
fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
for ax, (title, cols, ylab) in zip(axes.ravel(), panels):
    cols = [c for c in cols if c in mean.columns]
    w = 0.8 / len(order)
    for i, p in enumerate(order):
        ax.bar(np.arange(len(cols)) + i * w, mean.loc[p, cols], w, yerr=std.loc[p, cols], capsize=2, label=p)
    ax.set_xticks(np.arange(len(cols)) + 0.4 - w / 2)
    ax.set_xticklabels([c.split("_")[-1] if "_" in c else c for c in cols])
    ax.set_title(title); ax.set_ylabel(ylab); ax.grid(True, axis="y", linestyle="--", alpha=0.6)
axes[0, 0].legend(fontsize=8)
fig.suptitle(f"Static baselines vs DRL (test days, {len(files)} seed file(s))")
plt.tight_layout(); plt.savefig(RESULTS_DIR / f"comparison_bars{suffix}.png", dpi=200)
print(mean.round(4).T)
print(f">> saved results/comparison_bars{suffix}.png and comparison_table{suffix}.csv ({len(files)} seed file(s))")