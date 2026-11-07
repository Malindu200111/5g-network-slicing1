"""
Check the analytical priority formula (src/priority_model.py) against the SimPy priority simulation.
    python verify_priority.py            # a few minutes
    python verify_priority.py --quick    # ~1 min, noisier
Output: results/priority_wq_validation.csv, results/priority_wq_validation.png
A large buffer (K = c + 300) is used so the infinite-buffer formula applies.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import queuing_model as qm
from src.config import RESULTS_DIR, SLICES
from src.priority_model import priority_wq_nonpreemptive

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
sim_time, warmup, n_seeds = (8000, 800, 1) if args.quick else (30000, 2000, 2)

mu, c = 1.0, 10
K = c + 300
# (lambda URLLC, lambda eMBB, lambda mMTC) -> total rho = 0.5, 0.7, 0.85, 0.95
scenarios = [(1.5, 2.5, 1.0), (2.0, 3.5, 1.5), (2.5, 4.0, 2.0), (3.0, 4.5, 2.0)]

rows = []
print(f"c={c}, mu={mu}, K={K} | non-preemptive priority URLLC > eMBB > mMTC")
print(f"{'rho':>5} {'class':<6}{'Wq formula':>12}{'Wq SimPy':>10}{'rel.err %':>10}   (FCFS Wq: formula)")
for lams in scenarios:
    rho = sum(lams) / (c * mu)
    wq_an, wq_fcfs = priority_wq_nonpreemptive(lams, mu, c)
    sims = [qm.simulate_priority(dict(zip(SLICES, lams)), mu, c, K, sim_time=sim_time, warmup=warmup, seed=s + 1)
            for s in range(n_seeds)]
    for i, s in enumerate(SLICES):
        wq_sim = float(np.mean([x[s]["Wq"] for x in sims]))
        err = 100 * abs(wq_an[i] - wq_sim) / wq_an[i]
        rows.append({"rho": rho, "class": s, "lam": lams[i], "Wq_formula": wq_an[i], "Wq_sim": wq_sim,
                     "rel_err_%": err, "Wq_FCFS_formula": wq_fcfs})
        print(f"{rho:>5.2f} {s:<6}{wq_an[i]:>12.4f}{wq_sim:>10.4f}{err:>10.1f}   ({wq_fcfs:.4f})")

df = pd.DataFrame(rows)
RESULTS_DIR.mkdir(exist_ok=True)
df.to_csv(RESULTS_DIR / "priority_wq_validation.csv", index=False)
print(f"\nmean relative error {df['rel_err_%'].mean():.1f} %, max {df['rel_err_%'].max():.1f} %")

fig, ax = plt.subplots(figsize=(7, 4.5))
for s in SLICES:
    d = df[df["class"] == s]
    line, = ax.plot(d.rho, d.Wq_formula, "-", label=f"{s} (formula)")
    ax.plot(d.rho, d.Wq_sim, "o", mfc="none", color=line.get_color(), label=f"{s} (SimPy)")
ax.plot(df.drop_duplicates("rho").rho, df.drop_duplicates("rho").Wq_FCFS_formula, "k--", label="FCFS (no priority)")
ax.set_xlabel(r"Total offered load $\rho$"); ax.set_ylabel("Mean waiting time $W_q$ (s)")
ax.set_title("Non-preemptive priority M/M/c: formula vs SimPy"); ax.grid(True, linestyle="--", alpha=0.6); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "priority_wq_validation.png", dpi=200)
plt.show()
