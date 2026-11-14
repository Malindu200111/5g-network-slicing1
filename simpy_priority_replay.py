"""
Priority-based sharing of server capacity, evaluated at packet level (objective 1 / deliverable 4).

The agent (DQN/PPO/baseline) decides, every interval, how many servers and PRBs each slice gets
(a GUARANTEED share). This script replays those decisions in three modes with IDENTICAL arrivals:
  isolated : each slice only uses its own c_s servers (what the environment/reward assumes)
  pool_fifo: same shared pool but first-come-first-served (separates the effect of POOLING from PRIORITY)
  priority : the 30 servers form one pool. A slice always may use its guaranteed c_s servers; servers that
             are idle are lent to other slices, and when a server becomes free the next packet is chosen
             by NON-PREEMPTIVE PRIORITY: first slices still below their guarantee, each in the order
             URLLC > eMBB > mMTC, then (borrowing) any waiting slice in the same priority order.
             Per-slice system capacity stays K_s = c_s + buffer, so blocking is comparable.
Needs results/allocations_<tag>.csv (python simpy_replay.py --stage export ...).
    python simpy_priority_replay.py --days 151-155 --policies DQN,Proportional --scale 0.1   # quick check
    python simpy_priority_replay.py                                                           # test days, real rates
"""
import argparse
import multiprocessing as mp
import random
import time
from collections import deque

import numpy as np
import pandas as pd

from simpy_replay import SliceSim, _source, parse_days
from src.config import ASSUMED, RESULTS_DIR, SLICES
from src.environment import buffers, load_data, service_rate
from src.config import DATA_PATH


class PoolSim:
    """Three slices on one shared server pool with guaranteed shares and non-preemptive priority."""

    def __init__(self, env, rng, mu, buf, c, step_sec, steps, fifo=False):
        self.env, self.rng, self.buf, self.fifo = env, rng, buf, fifo
        self.mu, self.c = list(mu), [int(x) for x in c]
        self.busy, self.q = [0, 0, 0], [deque(), deque(), deque()]
        self.step_sec, self.steps = step_sec, steps
        self.arr, self.drop = np.zeros((3, steps)), np.zeros((3, steps))
        self.served, self.wait = np.zeros((3, steps)), np.zeros((3, steps))

    def _k(self):
        return min(int(self.env.now / self.step_sec), self.steps - 1)

    def arrive(self, s):
        k = self._k()
        self.arr[s, k] += 1
        if self.busy[s] + len(self.q[s]) >= self.c[s] + self.buf[s]:
            self.drop[s, k] += 1
            return
        self.q[s].append(self.env.now)
        self._dispatch()

    def _start(self, s, t_arr):
        k = self._k()
        self.busy[s] += 1
        self.served[s, k] += 1
        self.wait[s, k] += self.env.now - t_arr
        ev = self.env.timeout(self.rng.expovariate(self.mu[s]))
        ev.callbacks.append(lambda _, s=s: self._depart(s))

    def _depart(self, s):
        self.busy[s] -= 1
        self._dispatch()

    def _dispatch(self):
        while sum(self.busy) < sum(self.c):
            if self.fifo:                                  # pooled but NO priority: oldest waiting packet first
                order = sorted((self.q[s][0], s) for s in range(3) if self.q[s])
                if not order:
                    return
                s = order[0][1]
            else:
                order = [s for s in range(3) if self.q[s] and self.busy[s] < self.c[s]]   # guaranteed share first
                order = order or [s for s in range(3) if self.q[s]]                      # then borrow idle servers
                if not order:
                    return
                s = order[0]                                                              # URLLC > eMBB > mMTC
            self._start(s, self.q[s].popleft())

    def set_params(self, c, mu):
        self.c, self.mu = [int(x) for x in c], list(mu)
        self._dispatch()


class _Handle:
    def __init__(self, pool, s):
        self.pool, self.s = pool, s

    def arrive(self):
        self.pool.arrive(self.s)


def _ctrl_pool(env, pool, srv, mu, step_sec):
    for t in range(1, srv.shape[0]):
        yield env.timeout(step_sec)
        pool.set_params(srv[t], mu[t])


def _ctrl_iso(env, slices, srv, mu, step_sec):
    for t in range(1, srv.shape[0]):
        yield env.timeout(step_sec)
        for s in range(3):
            slices[s].set_params(srv[t, s], mu[t, s])


def job(j):
    import simpy
    policy, day, mode, srv, mu, lam_min, buf, scale, seed, step_sec = j
    steps, mu = srv.shape[0], mu * scale
    env = simpy.Environment()
    rng = random.Random(seed * 7919 + day)
    nprng = np.random.default_rng([seed, day])                # same arrivals in both modes and for all policies
    if mode in ("priority", "pool_fifo"):
        pool = PoolSim(env, rng, mu[0], buf, srv[0], step_sec, steps, fifo=(mode == "pool_fifo"))
        hs = [_Handle(pool, s) for s in range(3)]
        env.process(_ctrl_pool(env, pool, srv, mu, step_sec))
        out = lambda: (pool.arr, pool.drop, pool.served, pool.wait)
    else:
        sl = [SliceSim(env, rng, mu[0, s], buf[s], srv[0, s], step_sec, steps) for s in range(3)]
        hs = sl
        env.process(_ctrl_iso(env, sl, srv, mu, step_sec))
        out = lambda: tuple(np.array([getattr(x, n) for x in sl]) for n in ("arr", "drop", "served", "wait"))
    for s in range(3):
        env.process(_source(env, hs[s], lam_min[:, s], nprng, scale))
    env.run(until=steps * step_sec)
    arr, drop, served, wait = out()
    return {"policy": policy, "day": day, "mode": mode, "arr": arr.sum(1), "drop": drop.sum(1),
            "served": served.sum(1), "wait": wait.sum(1)}


MODES = ("isolated", "pool_fifo", "priority")      # dedicated | pooled, no priority | pooled + priority


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="151-180")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--policies", default="Static,Proportional,DQN")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="e.g. _prio")
    ap.add_argument("--minutes", type=int, default=ASSUMED["decision_minutes"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg = dict(ASSUMED); cfg["decision_minutes"] = args.minutes
    tag = f"{args.minutes}min_seed{args.seed}{args.suffix}"
    adf = pd.read_csv(RESULTS_DIR / f"allocations_{tag}.csv")
    data = load_data(DATA_PATH, args.minutes)
    days = parse_days(args.days, data["mean"].shape[0])[:: args.every]
    pols = [p.strip() for p in args.policies.split(",") if p.strip() in set(adf.policy)]
    step_sec = args.minutes * 60.0
    jobs = []
    for p in pols:
        for d in days:
            sd = adf[(adf.policy == p) & (adf.day == d + 1)].sort_values("step")
            prb, srv = sd[[f"alloc_{s}" for s in SLICES]].to_numpy(), sd[[f"srv_{s}" for s in SLICES]].to_numpy()
            mu = service_rate(prb, srv, cfg)
            for mode in MODES:
                jobs.append((p, d + 1, mode, srv, mu, data["fine"][d].reshape(1440, 3),
                             buffers(cfg), args.scale, args.seed, step_sec))
    print(f"priority replay: {len(pols)} policies x {len(days)} days x 3 modes = {len(jobs)} runs | scale {args.scale}")
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        res = list(pool.imap_unordered(job, jobs))
    print(f"done in {(time.time() - t0) / 60:.1f} min")

    rows = []
    for r in res:
        for s, name in enumerate(SLICES):
            rows.append({"policy": r["policy"], "mode": r["mode"], "day": r["day"], "slice": name,
                         "arr": r["arr"][s], "drop": r["drop"][s], "served": r["served"][s], "wait": r["wait"][s]})
    df = pd.DataFrame(rows)
    g = df.groupby(["policy", "mode", "slice"])[["arr", "drop", "served", "wait"]].sum().reset_index()
    g["PB"] = g["drop"] / g["arr"].clip(lower=1)
    g["Wq_s"] = g["wait"] / g["served"].clip(lower=1) * args.scale
    g.drop(columns=["arr", "drop", "served", "wait"]).to_csv(RESULTS_DIR / f"simpy_priority_replay_{tag}.csv", index=False)
    print(f"\n{'Policy':<13}{'Mode':<10}{'PB URLLC':>10}{'PB eMBB':>10}{'PB mMTC':>10}{'Wq URLLC':>10}{'Wq eMBB':>10}{'Wq mMTC':>10}")
    for p in pols:
        for mode in MODES:
            x = g[(g.policy == p) & (g["mode"] == mode)].set_index("slice")
            print(f"{p:<13}{mode:<10}" + "".join(f"{x.loc[s, 'PB']:>10.4f}" for s in SLICES)
                  + "".join(f"{x.loc[s, 'Wq_s']:>10.3f}" for s in SLICES))
    print(f">> saved results/simpy_priority_replay_{tag}.csv")


if __name__ == "__main__":
    main()
