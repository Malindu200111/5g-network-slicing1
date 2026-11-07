"""
Export the allocation decisions of DQN / Proportional / Static to a CSV, then REPLAY them in a
packet-level SimPy simulation (time-varying PRBs, minute-level lambda from the dataset CSV) and
compare simulated packet drops with the analytical M/M/c/K values used by the environment/reward.

Run from the project root (after main.py has produced results/dqn_best_15min_seed0.pt):
    python simpy_replay.py --days 151-151 --workers 1      # 1 test day: timing check (do this first)
    python simpy_replay.py                                 # test days 151-180 (30 days)
    python simpy_replay.py --days 1-180 --every 6          # whole 6 months, every 6th day
    python simpy_replay.py --days all --workers 6          # all 180 days (long!)
    python simpy_replay.py --stage export                  # only write the allocation CSV (fast)

Rate scaling: lambda and mu are both multiplied by --scale (default 0.1). Blocking probability of an
M/M/c/K queue depends only on lambda/mu, so drop PROBABILITIES are unchanged, while the number of
simulated packets (run time) is 10x smaller. Simulated waiting times are converted back (x scale).
Estimated real packet counts = simulated count / scale.
"""
import argparse
import multiprocessing as mp
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ASSUMED, DATA_PATH, RESULTS_DIR, SLICES
from src.environment import (SlicingEnv, load_data, policy_proportional, policy_static,
                             proportional_alloc)

HP = {"hidden": 128, "lr": 1e-3, "gamma": 0.95, "batch_size": 128, "buffer_size": 100_000,
      "learn_start": 1_000, "learn_every": 4, "tau": 0.01, "double_dqn": False}


def parse_days(txt, n_days):
    if txt == "all":
        return list(range(n_days))
    a, b = (int(x) for x in txt.split("-"))
    return list(range(a - 1, b))


# =====================================================================
# STAGE 1: export allocation decisions to CSV (analytical KPIs per step included)
# =====================================================================
def export_allocations(args, cfg, data, static_mean, out_csv):
    train = list(range(cfg["train_days"][0] - 1, cfg["train_days"][1]))
    env = SlicingEnv(data, train, cfg, init_alloc=static_mean)
    policies = {}
    names = [p.strip() for p in args.policies.split(",")]
    if "Static" in names:
        policies["Static"] = policy_static
    if "Proportional" in names:
        policies["Proportional"] = policy_proportional
    if "DQN" in names:
        from src.agent import DQNAgent
        model = args.model or str(RESULTS_DIR / f"dqn_best_{args.minutes}min_seed{args.seed}.pt")
        agent = DQNAgent(env.obs_dim, env.n_actions, HP, device="cpu", seed=args.seed)
        agent.load(model)
        print(f"loaded DQN model: {model}")
        policies["DQN"] = lambda e, o: agent.greedy(o)

    n_days = data["mean"].shape[0]
    split = lambda d: ("train" if cfg["train_days"][0] - 1 <= d < cfg["train_days"][1] else
                       "val" if cfg["val_days"][0] - 1 <= d < cfg["val_days"][1] else "test")
    rows, t0 = [], time.time()
    for pname, pol in policies.items():
        for d in range(n_days):
            obs, _ = env.reset(day=d, init_alloc=static_mean)
            done, t = False, 0
            while not done:
                obs, r, done, _, info = env.step(pol(env, obs))
                row = {"policy": pname, "split": split(d), "day": d + 1, "step": t,
                       "hour": t * cfg["decision_minutes"] / 60, "reward": r}
                for s, name in enumerate(SLICES):
                    m = info["metrics"][s]
                    row.update({f"alloc_{name}": int(info["alloc"][s]), f"lam_{name}": float(info["lam"][s]),
                                f"lam_eff_{name}": m["lam_eff"], f"PB_an_{name}": m["PB"],
                                f"Wq_an_{name}": m["Wq"]})
                rows.append(row)
                t += 1
        print(f"  exported {pname} ({n_days} days)  [{time.time() - t0:.0f}s]")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f">> allocation CSV saved: {out_csv}  ({len(df)} rows = {len(policies)} policies x {n_days} days x "
          f"{env.steps} steps)")
    return df


# =====================================================================
# STAGE 2: packet-level SimPy replay with time-varying capacity
# =====================================================================
class SliceSim:
    """M/M/c(t)/K(t) queue: c follows the allocation, K = c + buffer. A shrinking c lets in-service
    packets finish (no eviction); new arrivals are blocked while the system holds >= K packets."""

    def __init__(self, env, rng, mu, buf, c0, step_sec, steps):
        self.env, self.rng, self.mu, self.buf = env, rng, mu, buf
        self.c, self.busy, self.q = int(c0), 0, deque()
        self.step_sec, self.steps = step_sec, steps
        self.arr = np.zeros(steps); self.drop = np.zeros(steps)
        self.served = np.zeros(steps); self.wait = np.zeros(steps)

    def _k(self):
        return min(int(self.env.now / self.step_sec), self.steps - 1)

    def arrive(self):
        k = self._k()
        self.arr[k] += 1
        if self.busy + len(self.q) >= self.c + self.buf:
            self.drop[k] += 1
            return
        if self.busy < self.c:
            self._start(self.env.now)
        else:
            self.q.append(self.env.now)

    def _start(self, t_arrival):
        k = self._k()
        self.busy += 1
        self.served[k] += 1
        self.wait[k] += self.env.now - t_arrival
        ev = self.env.timeout(self.rng.expovariate(self.mu))
        ev.callbacks.append(self._depart)

    def _depart(self, _):
        self.busy -= 1
        if self.q and self.busy < self.c:
            self._start(self.q.popleft())

    def set_c(self, c):
        self.c = int(c)
        while self.q and self.busy < self.c:
            self._start(self.q.popleft())


def _source(env, sl, lam_min, nprng, scale):
    """Poisson arrivals with the minute-level lambda from the CSV (x scale)."""
    for m in range(1440):
        n = nprng.poisson(lam_min[m] * scale * 60.0)
        if n:
            prev = 0.0
            for x in (np.sort(nprng.random(n)) * 60.0).tolist():
                yield env.timeout(x - prev)
                prev = x
                sl.arrive()
            yield env.timeout(60.0 - prev)
        else:
            yield env.timeout(60.0)


def _controller(env, slices, alloc, step_sec):
    for t in range(1, alloc.shape[0]):
        yield env.timeout(step_sec)
        for s in range(3):
            slices[s].set_c(alloc[t, s])


def replay_job(job):
    import simpy
    policy, pidx, day, alloc, lam_min, mu, buf, scale, seed, step_sec = job
    steps = alloc.shape[0]
    t0 = time.time()
    env = simpy.Environment()
    py_rng = random.Random(seed * 1_000_003 + pidx * 10_007 + day)
    nprng = np.random.default_rng([seed, day])                  # same arrivals for every policy
    mu_s = mu * scale
    slices = [SliceSim(env, py_rng, mu_s, buf, alloc[0, s], step_sec, steps) for s in range(3)]
    for s in range(3):
        env.process(_source(env, slices[s], lam_min[:, s], nprng, scale))
    env.process(_controller(env, slices, alloc, step_sec))
    env.run(until=steps * step_sec)
    return {"policy": policy, "day": day, "secs": time.time() - t0,
            "arr": np.array([s.arr for s in slices]), "drop": np.array([s.drop for s in slices]),
            "served": np.array([s.served for s in slices]), "wait": np.array([s.wait for s in slices])}


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="both", choices=["export", "replay", "both"])
    ap.add_argument("--days", default="151-180", help="1-based range 'a-b' or 'all'")
    ap.add_argument("--every", type=int, default=1, help="use every k-th day of the range")
    ap.add_argument("--policies", default="Static,Proportional,DQN")
    ap.add_argument("--model", default=None, help="path of the DQN .pt (default results/dqn_best_<tag>.pt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=int, default=ASSUMED["decision_minutes"])
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    args = ap.parse_args()

    cfg = dict(ASSUMED); cfg["decision_minutes"] = args.minutes
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"{args.minutes}min_seed{args.seed}"
    alloc_csv = RESULTS_DIR / f"allocations_{tag}.csv"

    data = load_data(DATA_PATH, cfg["decision_minutes"])
    lam = data["mean"]
    train = list(range(cfg["train_days"][0] - 1, cfg["train_days"][1]))
    static_mean = proportional_alloc(lam[train].mean(axis=(0, 1)), cfg["total_prbs"], cfg["min_alloc"])

    # ---------------- stage 1
    if args.stage in ("export", "both"):
        adf = export_allocations(args, cfg, data, static_mean, alloc_csv)
    else:
        adf = pd.read_csv(alloc_csv)
        print(f"read allocations: {alloc_csv}")
    if args.stage == "export":
        return

    # ---------------- stage 2
    days = parse_days(args.days, lam.shape[0])[:: args.every]
    pols = [p for p in args.policies.split(",") if p in set(adf.policy)]
    steps, step_sec = lam.shape[1], cfg["decision_minutes"] * 60.0
    jobs = []
    for pidx, p in enumerate(pols):
        sub = adf[adf.policy == p]
        for d in days:
            a = sub[sub.day == d + 1].sort_values("step")[[f"alloc_{s}" for s in SLICES]].to_numpy()
            jobs.append((p, pidx, d + 1, a, data["fine"][d].reshape(1440, 3), cfg["mu"],
                         cfg["buffer_per_slice"], args.scale, args.seed, step_sec))
    print(f"\nSimPy replay: {len(pols)} policies x {len(days)} days = {len(jobs)} runs | scale {args.scale} | "
          f"workers {args.workers}")

    results, t0 = [], time.time()
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            it = pool.imap_unordered(replay_job, jobs)
            for i, r in enumerate(it, 1):
                results.append(r)
                if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                    el = time.time() - t0
                    print(f"  {i}/{len(jobs)} runs | elapsed {el / 60:.1f} min | ETA {el / i * (len(jobs) - i) / 60:.1f} min")
    else:
        for i, j in enumerate(jobs, 1):
            r = replay_job(j)
            results.append(r)
            el = time.time() - t0
            print(f"  {i}/{len(jobs)} {r['policy']:<12} day {r['day']:>3}: {r['secs']:.1f}s | "
                  f"ETA {el / i * (len(jobs) - i) / 60:.1f} min")

    # ---------------- OUTPUT DATASET: one row per (policy, day, step, slice)
    step_rows = []
    for r in results:
        sub = adf[(adf.policy == r["policy"]) & (adf.day == r["day"])].sort_values("step").reset_index(drop=True)
        for s, name in enumerate(SLICES):
            for k in range(steps):
                a, dr, sv = r["arr"][s][k], r["drop"][s][k], r["served"][s][k]
                step_rows.append({
                    "policy": r["policy"], "day": r["day"], "step": k, "hour": sub.loc[k, "hour"], "slice": name,
                    "alloc_prbs": int(sub.loc[k, f"alloc_{name}"]), "lambda_csv": sub.loc[k, f"lam_{name}"],
                    "sim_arrivals": a, "sim_drops": dr, "sim_service_starts": sv,
                    "PB_sim": dr / a if a else 0.0, "PB_analytical": sub.loc[k, f"PB_an_{name}"],
                    "Wq_sim_s": r["wait"][s][k] / sv * args.scale if sv else 0.0,
                    "Wq_analytical_s": sub.loc[k, f"Wq_an_{name}"],
                    "est_real_arrivals": a / args.scale, "est_real_drops": dr / args.scale})
    out_steps = RESULTS_DIR / f"simpy_output_dataset_{tag}.csv"
    steps_df = pd.DataFrame(step_rows).sort_values(["policy", "day", "step", "slice"])
    steps_df.to_csv(out_steps, index=False)
    print(f">> output dataset (all policies) saved: {out_steps} ({len(steps_df)} rows)")
    for p in pols:                                   # one file per policy: Static = without DQN, DQN = with DQN
        f = RESULTS_DIR / f"simpy_output_{p}_{tag}.csv"
        steps_df[steps_df.policy == p].to_csv(f, index=False)
        print(f"   {f.name}")

    # ---------------- aggregate per (policy, day, slice)
    rows = []
    for r in results:
        sub = adf[(adf.policy == r["policy"]) & (adf.day == r["day"])].sort_values("step")
        for s, name in enumerate(SLICES):
            arr, drop = r["arr"][s].sum(), r["drop"][s].sum()
            srv, wt = r["served"][s].sum(), r["wait"][s].sum()
            lam_w, le = sub[f"lam_{name}"].to_numpy(), sub[f"lam_eff_{name}"].to_numpy()
            rows.append({"policy": r["policy"], "day": r["day"], "slice": name,
                         "sim_arrivals": arr, "sim_drops": drop,
                         "PB_sim": drop / max(arr, 1),
                         "PB_an": float((lam_w * sub[f"PB_an_{name}"]).sum() / lam_w.sum()),
                         "Wq_sim_s": wt / max(srv, 1) * args.scale,
                         "Wq_an_s": float((le * sub[f"Wq_an_{name}"]).sum() / max(le.sum(), 1e-12)),
                         "est_real_arrivals": arr / args.scale, "est_real_drops": drop / args.scale})
    res = pd.DataFrame(rows).sort_values(["policy", "day", "slice"])
    out_daily = RESULTS_DIR / f"simpy_replay_daily_{tag}.csv"
    res.to_csv(out_daily, index=False)

    # ---------------- summary table
    print(f"\n=== SimPy replay vs analytical | {len(days)} days | scale {args.scale} ===")
    print(f"{'Policy':<13}{'Slice':<7}{'PB sim':>9}{'PB analyt':>11}{'diff':>8}{'Wq sim(s)':>11}{'Wq an(s)':>10}"
          f"{'est drops/day':>15}")
    summ = []
    for p in pols:
        for name in SLICES + ["ALL"]:
            d = res[res.policy == p] if name == "ALL" else res[(res.policy == p) & (res.slice == name)]
            arr, drop = d.sim_arrivals.sum(), d.sim_drops.sum()
            pb_sim = drop / max(arr, 1)
            pb_an = (d.PB_an * d.sim_arrivals).sum() / max(arr, 1)     # weighted by simulated arrivals
            wq_sim = np.average(d.Wq_sim_s, weights=d.sim_arrivals) if arr else 0.0
            wq_an = np.average(d.Wq_an_s, weights=d.sim_arrivals) if arr else 0.0
            drops_day = d.est_real_drops.sum() / len(days)
            print(f"{p:<13}{name:<7}{pb_sim:>9.4f}{pb_an:>11.4f}{pb_sim - pb_an:>+8.4f}{wq_sim:>11.3f}{wq_an:>10.3f}"
                  f"{drops_day:>15,.0f}")
            summ.append({"policy": p, "slice": name, "PB_sim": pb_sim, "PB_an": pb_an,
                         "Wq_sim_s": wq_sim, "Wq_an_s": wq_an, "est_drops_per_day": drops_day})
    pd.DataFrame(summ).to_csv(RESULTS_DIR / f"simpy_replay_summary_{tag}.csv", index=False)

    # ---------------- plots: analytical vs simulated PB, and daily drops per policy
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    mk = {"URLLC": "o", "eMBB": "s", "mMTC": "^"}
    for name in SLICES:
        d = res[res.slice == name]
        ax[0].scatter(d.PB_an, d.PB_sim, marker=mk[name], alpha=0.6, label=name)
    lim = max(res.PB_an.max(), res.PB_sim.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], "k--", lw=1)
    ax[0].set_xlabel("Analytical $P_B$ (M/M/c/K)"); ax[0].set_ylabel("SimPy $P_B$")
    ax[0].set_title("Per-day blocking: analytical vs packet-level SimPy"); ax[0].legend(); ax[0].grid(True, ls="--", alpha=0.6)
    tot = res.groupby(["policy", "day"]).est_real_drops.sum().reset_index()
    for p in pols:
        d = tot[tot.policy == p]
        ax[1].plot(d.day, d.est_real_drops / 1e3, "o-", ms=3, label=p)
    ax[1].set_xlabel("Day"); ax[1].set_ylabel("Estimated dropped packets per day (thousands)")
    ax[1].set_title("Packet drops per day (SimPy replay)"); ax[1].legend(); ax[1].grid(True, ls="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"simpy_replay_{tag}.png", dpi=200)
    print(f"\n>> saved: {alloc_csv.name}, {out_daily.name}, simpy_replay_summary_{tag}.csv, simpy_replay_{tag}.png "
          f"in {RESULTS_DIR}")
    plt.show()


if __name__ == "__main__":
    mp.freeze_support()
    main()
