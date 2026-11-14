"""
Burst test (deliverable 4, elastic provisioning): the TEST days with the strongest URLLC spikes versus the
typical day. Compares the trained DQN with the baselines on exactly those days.
    python burst_test.py --seed 0            # needs results/dqn_best_15min_seed0.pt
"""
import argparse

import numpy as np
import pandas as pd

from src.agent import DQNAgent
from src.config import ASSUMED, DATA_PATH, RESULTS_DIR, SLICES
from src.environment import (SlicingEnv, evaluate, load_data, make_alloc, policy_proportional, policy_static)
from train_hparams import HP

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--priority", action="store_true", help="model/environment of main.py --priority")
ap.add_argument("--n", type=int, default=5, help="number of burst days")
args = ap.parse_args()
cfg = dict(ASSUMED)
cfg["priority_sharing"] = bool(args.priority)
suffix = "_prio" if args.priority else ""
days = lambda r: list(range(r[0] - 1, r[1]))
train, test = days(cfg["train_days"]), days(cfg["test_days"])
data = load_data(DATA_PATH, cfg["decision_minutes"])
static = make_alloc(data["mean"][train].mean(axis=(0, 1)), cfg)
env = SlicingEnv(data, train, cfg, init_alloc=static)
agent = DQNAgent(env.obs_dim, env.n_actions, HP, device="cpu", seed=args.seed)
agent.load(RESULTS_DIR / f"dqn_best_15min_seed{args.seed}{suffix}.pt")
policies = {"Static (mean)": policy_static, "Proportional": policy_proportional, "DQN": lambda e, o: agent.greedy(o)}

peak_urllc = np.array([data["peak"][d, :, 0].max() for d in test])       # strongest URLLC minute-spike mean per day
burst = [test[i] for i in np.argsort(-peak_urllc)[: args.n]]
rest = [d for d in test if d not in burst]
rows = []
for label, ds in [(f"burst days (top {args.n} URLLC peaks)", burst), ("all other test days", rest)]:
    for name, pol in policies.items():
        r = evaluate(env, pol, ds, static)
        rows.append({"days": label, "policy": name, "PB_URLLC": r["PB"][0], "Wq_URLLC": r["Wq"][0],
                     "PB_eMBB": r["PB"][1], "PB_mMTC": r["PB"][2], "return_per_day": r["return_per_day"]})
out = pd.DataFrame(rows)
print(f"burst days (1-based): {[d + 1 for d in burst]}")
print(out.round(4).to_string(index=False))
out.to_csv(RESULTS_DIR / f"burst_test_seed{args.seed}{suffix}.csv", index=False)
print(f">> saved results/burst_test_seed{args.seed}{suffix}.csv")
