"""Run from the project root:  python main.py"""
import matplotlib.pyplot as plt
import numpy as np

from src.config import ASSUMED, DATA_PATH, RESULTS_DIR, SLICES
from src.environment import (SlicingEnv, evaluate, load_data, policy_proportional,
                             policy_static, print_table, proportional_alloc)

cfg = ASSUMED
if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {DATA_PATH}\nCopy it into data/processed/ first.")

days = lambda r: list(range(r[0] - 1, r[1]))      # 1-based inclusive -> 0-based indices
train, val, test = days(cfg["train_days"]), days(cfg["val_days"]), days(cfg["test_days"])

data = load_data(DATA_PATH, cfg["decision_minutes"])
lam = data["mean"]
print("CSV :", DATA_PATH)
print(f"decision interval: {cfg['decision_minutes']} min | lambda array (days, steps, slices): {lam.shape}")

# Static allocations are designed from TRAIN days only
static_mean = proportional_alloc(lam[train].mean(axis=(0, 1)), cfg["total_prbs"], cfg["min_alloc"])
static_peak = proportional_alloc(lam[train].max(axis=(0, 1)), cfg["total_prbs"], cfg["min_alloc"])
print(f"Static (mean) alloc {SLICES}: {static_mean}")
print(f"Static (peak) alloc {SLICES}: {static_peak}")

env = SlicingEnv(data, train, cfg)
for name, d in [("VALIDATION", val), ("TEST", test)]:
    res = {"Static (mean)": evaluate(env, policy_static, d, static_mean),
           "Static (peak)": evaluate(env, policy_static, d, static_peak),
           "Proportional": evaluate(env, policy_proportional, d, static_mean)}
    print(f"\n=== {name} days {d[0]+1}-{d[-1]+1} ===")
    print_table(res)

# ---- Sensitivity to the decision interval (metrics are minute-level, so intervals compare fairly)
print("\n=== SENSITIVITY: decision interval (TEST days, minute-level KPIs) ===")
print(f"{'interval':>9}{'steps/day':>10}{'Static PB URLLC':>17}{'Prop PB URLLC':>15}{'Prop PB eMBB':>14}{'Moves/day':>11}")
for dm in (5, 15, 60):
    c2 = dict(cfg); c2["decision_minutes"] = dm
    d2 = load_data(DATA_PATH, dm)
    sm = proportional_alloc(d2["mean"][train].mean(axis=(0, 1)), c2["total_prbs"], c2["min_alloc"])
    e2 = SlicingEnv(d2, train, c2)
    st, pr = evaluate(e2, policy_static, test, sm), evaluate(e2, policy_proportional, test, sm)
    print(f"{dm:>7}m{d2['mean'].shape[1]:>10}{st['PB'][0]:>17.4f}{pr['PB'][0]:>15.4f}{pr['PB'][1]:>14.4f}{pr['moves_per_day']:>11.1f}")

# ---- plot one test day
RESULTS_DIR.mkdir(exist_ok=True)
d0 = test[0]
obs, _ = env.reset(day=d0, init_alloc=static_mean)
allocs, done = [], False
while not done:
    obs, r, done, _, info = env.step(policy_proportional(env, obs))
    allocs.append(info["alloc"])
allocs = np.array(allocs)
hrs = np.arange(env.steps) * cfg["decision_minutes"] / 60
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for s in range(3):
    ax[0].plot(hrs, lam[d0, :, s], label=SLICES[s])
    ax[1].plot(hrs, allocs[:, s], label=SLICES[s])
ax[0].set_ylabel("lambda (PRB-Eq. pkts/s)"); ax[0].set_title(f"Day {d0+1}: lambda(t) from CSV")
ax[1].set_ylabel("Allocated PRBs"); ax[1].set_title("Allocation under proportional heuristic")
ax[1].set_xlabel("Hour of day")
for a in ax:
    a.grid(True, linestyle="--", alpha=0.6); a.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "env_day_example.png", dpi=200)
plt.show()
