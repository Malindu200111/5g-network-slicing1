"""
Train a DQN on the slicing environment and compare with the baselines.
Run from the project root:
    python train_dqn.py                    # default: 2000 episodes, interval from config.py
    python train_dqn.py --episodes 300     # quick test (written to results/quick_ep300/)
    python train_dqn.py --minutes 5        # decision interval override (5, 15, 60 ...)
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
                             policy_static, print_table)
from train_hparams import FULL_EPISODES, HP, output_dir

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=FULL_EPISODES)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--minutes", type=int, default=ASSUMED["decision_minutes"])
args = parser.parse_args()

cfg = dict(ASSUMED)
cfg["decision_minutes"] = args.minutes
np.random.seed(args.seed)
torch.manual_seed(args.seed)
OUT_DIR = output_dir(args.episodes)
OUT_DIR.mkdir(parents=True, exist_ok=True)
tag = f"{args.minutes}min_seed{args.seed}"

# ---------------- data and baselines ----------------
days = lambda r: list(range(r[0] - 1, r[1]))
train, val, test = days(cfg["train_days"]), days(cfg["val_days"]), days(cfg["test_days"])
data = load_data(DATA_PATH, cfg["decision_minutes"])
lam = data["mean"]
static_mean = make_alloc(lam[train].mean(axis=(0, 1)), cfg)      # rows: [PRBs, servers]
static_peak = make_alloc(lam[train].max(axis=(0, 1)), cfg)
print(f"interval {args.minutes} min | steps/day {lam.shape[1]} | PRBs {cfg['total_prbs']} | servers {cfg['total_servers']} | "
      f"static(mean) PRBs {static_mean[0]} servers {static_mean[1]} | episodes {args.episodes}")

env_train = SlicingEnv(data, train, cfg, init_alloc=static_mean, seed=args.seed)
env_eval = SlicingEnv(data, train, cfg, init_alloc=static_mean)
agent = DQNAgent(env_train.obs_dim, env_train.n_actions, HP, device="cpu", seed=args.seed)
dqn_policy = lambda env, obs: agent.greedy(obs)

val_prop = evaluate(env_eval, policy_proportional, val, static_mean)["return_per_day"]
val_static = evaluate(env_eval, policy_static, val, static_mean)["return_per_day"]
print(f"validation return/day  Static(mean) {val_static:.1f} | Proportional {val_prop:.1f}")

# ---------------- training ----------------
best_val, best_ep, step, best_state = -np.inf, 0, 0, None
ep_returns, val_points = [], []
best_path = OUT_DIR / f"dqn_best_{tag}.pt"
t0 = time.time()
for ep in range(1, args.episodes + 1):
    eps = max(HP["eps_end"], HP["eps_start"] - (HP["eps_start"] - HP["eps_end"]) * ep / (args.episodes * HP["eps_decay_frac"]))
    obs, _ = env_train.reset()                       # random TRAIN day
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
        el = time.time() - t0
        print(f"ep {ep:5d} | eps {eps:.2f} | train return/day {np.mean(ep_returns[-HP['eval_every']:]):8.1f} | "
              f"val {v:8.1f} (best {best_val:8.1f} @ {best_ep}) | {el/60:.1f} min")
print(f"training time: {(time.time() - t0)/60:.1f} min")

# ---------------- final test with the BEST validation model ----------------
agent.set_state(best_state)
agent.save(best_path)
results = {
    "Static (mean)": evaluate(env_eval, policy_static, test, static_mean),
    "Static (peak)": evaluate(env_eval, policy_static, test, static_peak),
    "Proportional": evaluate(env_eval, policy_proportional, test, static_mean),
    f"DQN (ep {best_ep})": evaluate(env_eval, dqn_policy, test, static_mean),
}
print(f"\n=== TEST days {test[0]+1}-{test[-1]+1} (model chosen on validation days) ===")
print_table(results)

rows = []
for name, r in results.items():
    rows.append({"policy": name, **{f"PB_{s}": r["PB"][i] for i, s in enumerate(SLICES)},
                 **{f"Wq_{s}": r["Wq"][i] for i, s in enumerate(SLICES)},
                 **{f"W_{s}": r["W"][i] for i, s in enumerate(SLICES)},
                 **{f"Lq_{s}": r["Lq"][i] for i, s in enumerate(SLICES)},
                 "throughput": r["throughput"],
                 "util": r["util"], "moves_per_day": r["moves_per_day"], "return_per_day": r["return_per_day"]})
pd.DataFrame(rows).to_csv(OUT_DIR / f"test_results_{tag}.csv", index=False)

# ---------------- plots ----------------
fig, ax = plt.subplots(figsize=(10, 5))
w = min(50, max(1, len(ep_returns) // 10))
smooth = np.convolve(np.array(ep_returns), np.ones(w) / w, mode="valid")
ax.plot(np.arange(w, len(ep_returns) + 1), smooth, label=f"Training return/day (moving avg {w})", alpha=0.8)
ax.plot(*zip(*val_points), "o-", label="Validation return/day (greedy)")
ax.axhline(val_prop, color="green", linestyle="--", label="Proportional (validation)")
ax.axhline(val_static, color="red", linestyle="--", label="Static mean (validation)")
ax.set_xlabel("Episode (1 episode = 1 day)"); ax.set_ylabel("Return per day (higher is better)")
ax.set_title("DQN training convergence"); ax.grid(True, linestyle="--", alpha=0.6); ax.legend()
plt.tight_layout(); plt.savefig(OUT_DIR / f"training_curve_{tag}.png", dpi=200)

def run_day(policy, d):
    obs, _ = env_eval.reset(day=d, init_alloc=static_mean)
    allocs, done = [], False
    while not done:
        obs, r, done, _, info = env_eval.step(policy(env_eval, obs))
        allocs.append(info["alloc"])
    return np.array(allocs)

d0 = test[0]
hrs = np.arange(env_eval.steps) * cfg["decision_minutes"] / 60
fig, ax = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
for s in range(3):
    ax[0].plot(hrs, lam[d0, :, s], label=SLICES[s])
k = 1
for name, pol in [("Proportional", policy_proportional), ("DQN", dqn_policy)]:
    al = run_day(pol, d0)                               # (steps, 2, 3)
    for r, res in enumerate(["PRBs (bandwidth)", "servers (capacity)"]):
        for s in range(3):
            ax[k].plot(hrs, al[:, r, s], label=SLICES[s])
        ax[k].set_ylabel(f"Allocated {res.split()[0]}"); ax[k].set_title(f"{name}: {res}")
        k += 1
ax[0].set_ylabel("lambda"); ax[0].set_title(f"Day {d0+1}: traffic from CSV"); ax[-1].set_xlabel("Hour of day")
for a in ax:
    a.grid(True, linestyle="--", alpha=0.6); a.legend(loc="upper right")
plt.tight_layout(); plt.savefig(OUT_DIR / f"allocation_dqn_vs_prop_{tag}.png", dpi=200)
plt.show()
