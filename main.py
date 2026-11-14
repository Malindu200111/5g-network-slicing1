"""
Run from the project root:
    python main.py                      # full run: baselines + DQN training + evaluation + plots
    python main.py --episodes 300       # quick test (written to results/quick_ep300/)
    python main.py --minutes 5          # decision interval override (5, 15, 60 ...)
    python main.py --seed 1             # another seed (run 3-5 seeds for the thesis)
    python main.py --no-sensitivity     # skip the interval sensitivity table
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.agent import DQNAgent
from src.config import ASSUMED, DATA_PATH, SLICES
from src.environment import (SlicingEnv, evaluate, load_data, make_alloc, policy_proportional,
                             policy_static)
from train_hparams import FULL_EPISODES, HP, output_dir

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=FULL_EPISODES)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--minutes", type=int, default=ASSUMED["decision_minutes"])
parser.add_argument("--no-sensitivity", action="store_true")
parser.add_argument("--priority", action="store_true",
                    help="train/evaluate with the priority-sharing queue model (outputs get the suffix _prio)")
args = parser.parse_args()

cfg = dict(ASSUMED)
cfg["decision_minutes"] = args.minutes
cfg["priority_sharing"] = bool(args.priority)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
OUT_DIR = output_dir(args.episodes)
OUT_DIR.mkdir(parents=True, exist_ok=True)
tag = f"{args.minutes}min_seed{args.seed}" + ("_prio" if args.priority else "")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {DATA_PATH}\nCopy it into data/processed/ first.")


def print_full_table(results):
    """Per-slice PB and Wq, latency W = Wq + 1/mu_eff, throughput, utilisation, moves, reward."""
    print(f"{'Policy':<16}{'PB U':>8}{'PB e':>8}{'PB m':>8}{'Wq U':>8}{'Wq e':>8}{'Wq m':>8}"
          f"{'W U':>8}{'Thru':>8}{'Util':>8}{'Moves/d':>9}{'Reward/d':>10}")
    for name, r in results.items():
        print(f"{name:<16}{r['PB'][0]:>8.4f}{r['PB'][1]:>8.4f}{r['PB'][2]:>8.4f}"
              f"{r['Wq'][0]:>8.3f}{r['Wq'][1]:>8.3f}{r['Wq'][2]:>8.3f}"
              f"{r['W'][0]:>8.3f}{r['throughput']:>8.2f}{r['util']:>8.3f}"
              f"{r['moves_per_day']:>9.1f}{r['return_per_day']:>10.2f}")


# =====================================================================
# 0. DATA AND BASELINES
# =====================================================================
days = lambda r: list(range(r[0] - 1, r[1]))       # 1-based inclusive -> 0-based indices
train, val, test = days(cfg["train_days"]), days(cfg["val_days"]), days(cfg["test_days"])

data = load_data(DATA_PATH, cfg["decision_minutes"])
lam = data["mean"]
print("CSV :", DATA_PATH)
print(f"decision interval: {cfg['decision_minutes']} min | lambda array (days, steps, slices): {lam.shape}")

# Static allocations are designed from TRAIN days only
static_mean = make_alloc(lam[train].mean(axis=(0, 1)), cfg)      # rows: [PRBs, servers]
static_peak = make_alloc(lam[train].max(axis=(0, 1)), cfg)
print(f"Static (mean) alloc {SLICES}: PRBs {static_mean[0]} | servers {static_mean[1]}")
print(f"Static (peak) alloc {SLICES}: PRBs {static_peak[0]} | servers {static_peak[1]}")

env_train = SlicingEnv(data, train, cfg, init_alloc=static_mean, seed=args.seed)
env_eval = SlicingEnv(data, train, cfg, init_alloc=static_mean)   # days are given per evaluate() call

val_prop = evaluate(env_eval, policy_proportional, val, static_mean)["return_per_day"]
val_static = evaluate(env_eval, policy_static, val, static_mean)["return_per_day"]
print(f"validation return/day  Static(mean) {val_static:.1f} | Proportional {val_prop:.1f}")

# =====================================================================
# 1. TRAIN DQN (random TRAIN day per episode; best model chosen on VALIDATION days)
# =====================================================================
print("\n" + "=" * 78)
print(f" DQN TRAINING | {args.episodes} episodes | seed {args.seed} | tag {tag}")
print("=" * 78)

agent = DQNAgent(env_train.obs_dim, env_train.n_actions, HP, device="cpu", seed=args.seed)
dqn_policy = lambda env, obs: agent.greedy(obs)

best_val, best_ep, step, best_state = -np.inf, 0, 0, None
ep_returns, val_points = [], []
best_path = OUT_DIR / f"dqn_best_{tag}.pt"
t0 = time.time()

for ep in range(1, args.episodes + 1):
    eps = max(HP["eps_end"],
              HP["eps_start"] - (HP["eps_start"] - HP["eps_end"]) * ep / (args.episodes * HP["eps_decay_frac"]))
    obs, _ = env_train.reset()
    done, ret = False, 0.0
    while not done:
        a = agent.act(obs, eps)
        nobs, r, done, _, _ = env_train.step(a)
        agent.remember(obs, a, r, nobs, done)
        step += 1
        if step % HP["learn_every"] == 0:
            agent.learn()
        obs, ret = nobs, ret + r
    ep_returns.append(ret)

    if ep % HP["eval_every"] == 0 or ep == args.episodes:
        v = evaluate(env_eval, dqn_policy, val, static_mean)["return_per_day"]
        val_points.append((ep, v))
        if v > best_val:
            best_val, best_ep = v, ep
            best_state = agent.get_state()          # in memory, so a failed disk write cannot change results
        print(f"ep {ep:5d} | eps {eps:.2f} | train return/day {np.mean(ep_returns[-HP['eval_every']:]):8.1f} | "
              f"val {v:8.1f} (best {best_val:8.1f} @ {best_ep}) | {(time.time() - t0) / 60:.1f} min")

print(f">> DQN training finished in {(time.time() - t0) / 60:.1f} min | best validation model: episode {best_ep}")

# =====================================================================
# 2. EVALUATE ALL POLICIES (validation + test) with the BEST validation model
# =====================================================================
agent.set_state(best_state)
agent.save(best_path)
for name, d in [("VALIDATION", val), ("TEST", test)]:
    res = {
        "Static (mean)": evaluate(env_eval, policy_static, d, static_mean),
        "Static (peak)": evaluate(env_eval, policy_static, d, static_peak),
        "Proportional": evaluate(env_eval, policy_proportional, d, static_mean),
        f"DQN (ep{best_ep})": evaluate(env_eval, dqn_policy, d, static_mean),
    }
    print(f"\n=== {name} days {d[0] + 1}-{d[-1] + 1} ===")
    print_full_table(res)
    if name == "TEST":
        rows = []
        for pname, r in res.items():
            rows.append({"policy": pname,
                         **{f"PB_{s}": r["PB"][i] for i, s in enumerate(SLICES)},
                         **{f"Wq_{s}": r["Wq"][i] for i, s in enumerate(SLICES)},
                         **{f"W_{s}": r["W"][i] for i, s in enumerate(SLICES)},
                         **{f"Lq_{s}": r["Lq"][i] for i, s in enumerate(SLICES)},
                         "throughput": r["throughput"],
                         "util": r["util"], "moves_per_day": r["moves_per_day"],
                         "return_per_day": r["return_per_day"]})
        csv_out = OUT_DIR / f"test_results_{tag}.csv"
        pd.DataFrame(rows).to_csv(csv_out, index=False)
        print(f">> test table saved: {csv_out}")

# =====================================================================
# 3. SENSITIVITY: decision interval (baselines, TEST days, minute-level KPIs)
# =====================================================================
if not args.no_sensitivity:
    print("\n=== SENSITIVITY: decision interval (TEST days, minute-level KPIs) ===")
    print(f"{'interval':>9}{'steps/day':>10}{'Static PB URLLC':>17}{'Prop PB URLLC':>15}"
          f"{'Prop PB eMBB':>14}{'Moves/day':>11}")
    for dm in (5, 15, 60):
        c2 = dict(cfg); c2["decision_minutes"] = dm
        d2 = load_data(DATA_PATH, dm)
        sm = make_alloc(d2["mean"][train].mean(axis=(0, 1)), c2)
        e2 = SlicingEnv(d2, train, c2)
        st, pr = evaluate(e2, policy_static, test, sm), evaluate(e2, policy_proportional, test, sm)
        print(f"{dm:>7}m{d2['mean'].shape[1]:>10}{st['PB'][0]:>17.4f}{pr['PB'][0]:>15.4f}"
              f"{pr['PB'][1]:>14.4f}{pr['moves_per_day']:>11.1f}")

# =====================================================================
# 4. PLOTS
# =====================================================================
# 4a. training convergence
fig, ax = plt.subplots(figsize=(10, 5))
w = min(50, max(1, len(ep_returns) // 10))
smooth = np.convolve(np.array(ep_returns), np.ones(w) / w, mode="valid")
ax.plot(np.arange(w, len(ep_returns) + 1), smooth, alpha=0.8, label=f"Training return/day (moving avg {w})")
ax.plot(*zip(*val_points), "o-", label="Validation return/day (greedy)")
ax.axhline(val_prop, color="green", linestyle="--", label="Proportional (validation)")
ax.axhline(val_static, color="red", linestyle="--", label="Static mean (validation)")
ax.set_xlabel("Episode (1 episode = 1 day)"); ax.set_ylabel("Return per day (higher is better)")
ax.set_title("DQN training convergence"); ax.grid(True, linestyle="--", alpha=0.6); ax.legend()
plt.tight_layout(); plt.savefig(OUT_DIR / f"training_curve_{tag}.png", dpi=200)


# 4b. one test day: traffic, proportional allocation, DQN allocation
def run_day(policy, d):
    obs, _ = env_eval.reset(day=d, init_alloc=static_mean)
    allocs, done = [], False
    while not done:
        obs, r, done, _, info = env_eval.step(policy(env_eval, obs))
        allocs.append(info["alloc"])
    return np.array(allocs)                         # shape (steps, 2, 3): [PRBs, servers] x slices


d0 = test[0]
hrs = np.arange(env_eval.steps) * cfg["decision_minutes"] / 60
fig, ax = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
for s in range(3):
    ax[0].plot(hrs, lam[d0, :, s], label=SLICES[s])
k = 1
for pname, pol in [("Proportional", policy_proportional), ("DQN", dqn_policy)]:
    al = run_day(pol, d0)
    for r, res in enumerate(["PRBs (bandwidth)", "servers (capacity)"]):
        for s in range(3):
            ax[k].plot(hrs, al[:, r, s], label=SLICES[s])
        ax[k].set_ylabel(f"Allocated {res.split()[0]}"); ax[k].set_title(f"{pname}: {res}")
        k += 1
ax[0].set_ylabel("lambda"); ax[0].set_title(f"Day {d0 + 1}: traffic from CSV"); ax[-1].set_xlabel("Hour of day")
for a in ax:
    a.grid(True, linestyle="--", alpha=0.6); a.legend(loc="upper right")
plt.tight_layout(); plt.savefig(OUT_DIR / f"allocation_dqn_vs_prop_{tag}.png", dpi=200)

print(f"\n>> plots saved in: {OUT_DIR}")
plt.show()
