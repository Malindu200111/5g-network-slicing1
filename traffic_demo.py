"""Demo of the parameterised traffic generator: lambda(t), mu, c, K -> queue-length time series.
    python traffic_demo.py            # isolated slices and one shared priority pool, plot saved to results/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import RESULTS_DIR
from src.priority_model import priority_ctmc
from src.traffic_generator import SliceTraffic, run_generator

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
day = lambda base, amp: (lambda t: base + amp * max(0.0, np.sin(np.pi * (t % 3600) / 3600)))   # 1-hour demo profile
sl = {"URLLC": SliceTraffic(lam=day(4, 6), mu=2.0, c=10, K=20),
      "eMBB": SliceTraffic(lam=day(14, 16), mu=2.0, c=14, K=44),
      "mMTC": SliceTraffic(lam=day(3, 4), mu=2.0, c=6, K=26)}
for label, prio in (("isolated slices", False), ("shared pool + priority", True)):
    out = run_generator(sl, horizon=3600, sample_every=5, seed=3, priority=prio)
    print(f"\n{label}")
    for n, s in out["summary"].items():
        print(f"  {n:<6} arrivals {s['arrivals']:>6}  P_B {s['PB']:.4f}  Wq {s['Wq']:.3f}s  W {s['W']:.3f}s  Lq {s['Lq']:.2f}")
    plt.figure(figsize=(8, 3.2))
    for n in sl:
        plt.plot(out["t"] / 60, out["queue_len"][n], label=n, lw=0.9)
    plt.xlabel("minute"); plt.ylabel("queue length"); plt.title(f"Generated traffic: {label}"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"traffic_demo_{'prio' if prio else 'iso'}.png", dpi=120); plt.close()

# constant-rate check against the exact finite-buffer priority CTMC
const = {"URLLC": SliceTraffic(16, 2.0, 10, 20), "eMBB": SliceTraffic(38, 2.0, 10, 20), "mMTC": SliceTraffic(6, 2.0, 10, 20)}
sim = run_generator(const, horizon=30000, warmup=1000, seed=5, priority=True, pool_servers=30)["summary"]
ex = priority_ctmc([16, 38, 6], 2.0, 30, 60)
print("\nconstant lambda=[16,38,6], mu=2, shared c=30, K=60: simulation vs exact CTMC")
for i, n in enumerate(const):
    print(f"  {n:<6} PB {sim[n]['PB']:.4f}/{ex['PB'][i]:.4f}   Wq {sim[n]['Wq']:.3f}/{ex['Wq'][i]:.3f}")
