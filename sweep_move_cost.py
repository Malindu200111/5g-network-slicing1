"""
Sweep the reconfiguration penalty (move_cost) and compare on COMMON metrics.
Rewards from different move_cost values are NOT comparable, so every trained agent is
re-evaluated on the TEST days with move_cost = 0 (pure QoS reward) and we also report
loss (PB) per slice and reconfigurations per day.

    python sweep_move_cost.py                      # default sweep
    python sweep_move_cost.py --episodes 300 --seeds 0   # quick test
"""
import argparse
import time

import numpy as np
import pandas as pd

from src.config import ASSUMED, DATA_PATH, RESULTS_DIR, SLICES
from src.environment import (SlicingEnv, evaluate, load_data, policy_proportional,
                             proportional_alloc)
from src.trainer import train_dqn
import train_hparams as H            # DQN hyperparameters (shared with train_dqn.py)

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--costs", type=float, nargs="+", default=[0.05, 0.3, 1.0, 2.0])
parser.add_argument("--minutes", type=int, default=ASSUMED["decision_minutes"])
parser.add_argument("--fresh", action="store_true", help="ignore previous results and start over")
args = parser.parse_args()

RESULTS_DIR.mkdir(exist_ok=True)
cfg0 = dict(ASSUMED); cfg0["decision_minutes"] = args.minutes
days = lambda r: list(range(r[0] - 1, r[1]))
train, val, test = days(cfg0["train_days"]), days(cfg0["val_days"]), days(cfg0["test_days"])
data = load_data(DATA_PATH, args.minutes)
static_mean = proportional_alloc(data["mean"][train].mean(axis=(0, 1)), cfg0["total_prbs"], cfg0["min_alloc"])

# common evaluation environment: pure QoS reward (move_cost = 0)
cfg_eval = dict(cfg0); cfg_eval["move_cost"] = 0.0
env_common = SlicingEnv(data, train, cfg_eval, init_alloc=static_mean)

RAW = RESULTS_DIR / "sweep_move_cost_raw.csv"
rows, done = [], set()
if RAW.exists() and not args.fresh:                  # resume after a crash
    old = pd.read_csv(RAW)
    if "episodes" in old.columns:                    # only reuse runs trained with the same episode count
        old = old[(old["episodes"].isna()) | (old["episodes"] == args.episodes) | (old["policy"] == "Proportional")]
    rows = old.to_dict("records")
    done = {(round(r["move_cost"], 6), int(r["seed"])) for r in rows if r["policy"] == "DQN"}
    print(f"Resuming: {len(done)} finished DQN runs found in {RAW.name}: {sorted(done)}")

if not any(r["policy"] == "Proportional" for r in rows):
    prop = evaluate(env_common, policy_proportional, test, static_mean)
    rows.append({"policy": "Proportional", "move_cost": np.nan, "seed": -1, "episodes": np.nan,
                 **{f"PB_{s}": prop["PB"][i] for i, s in enumerate(SLICES)},
                 "moves_per_day": prop["moves_per_day"], "qos_return": prop["return_per_day"]})

t0 = time.time()
for mc in args.costs:
    for seed in args.seeds:
        if (round(mc, 6), seed) in done:
            print(f"skip move_cost {mc} seed {seed} (already done)")
            continue
        cfg = dict(cfg0); cfg["move_cost"] = mc
        print(f"\n=== move_cost {mc} | seed {seed} | {(time.time()-t0)/60:.1f} min elapsed ===", flush=True)
        agent, info = train_dqn(data, cfg, H.HP, train, val, static_mean, args.episodes, seed,
                                RESULTS_DIR / f"sweep_mc{mc}_s{seed}.pt")
        r = evaluate(env_common, lambda env, obs: agent.greedy(obs), test, static_mean)
        rows.append({"policy": "DQN", "move_cost": mc, "seed": seed, "episodes": args.episodes,
                     **{f"PB_{s}": r["PB"][i] for i, s in enumerate(SLICES)},
                     "moves_per_day": r["moves_per_day"], "qos_return": r["return_per_day"]})
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "sweep_move_cost_raw.csv", index=False)

df = pd.DataFrame(rows)
cols = [f"PB_{s}" for s in SLICES] + ["moves_per_day", "qos_return"]
summary = df[df.policy == "DQN"].groupby("move_cost")[cols].agg(["mean", "std"]).round(4)
summary.to_csv(RESULTS_DIR / "sweep_move_cost_summary.csv")

print("\n\n=== TEST days, common metrics (QoS return uses move_cost = 0) ===")
print(f"{'':<14}{'PB URLLC':>10}{'PB eMBB':>10}{'PB mMTC':>10}{'Moves/day':>11}{'QoS ret/day':>13}")
p = df[df.policy == "Proportional"].iloc[0]
print(f"{'Proportional':<14}{p.PB_URLLC:>10.4f}{p.PB_eMBB:>10.4f}{p.PB_mMTC:>10.4f}{p.moves_per_day:>11.1f}{p.qos_return:>13.1f}")
for mc, g in df[df.policy == "DQN"].groupby("move_cost"):
    print(f"{'DQN mc=' + str(mc):<14}{g.PB_URLLC.mean():>10.4f}{g.PB_eMBB.mean():>10.4f}{g.PB_mMTC.mean():>10.4f}"
          f"{g.moves_per_day.mean():>11.1f}{g.qos_return.mean():>13.1f}   (n={len(g)} seeds, moves std {g.moves_per_day.std():.1f})")
print(f"\ntotal time {(time.time()-t0)/60:.1f} min | saved results/sweep_move_cost_summary.csv")
