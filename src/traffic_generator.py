"""
Traffic generator for the 5G slicing simulator (Deliverable 1).

A slice generator is parameterised by
    lam   arrival rate  - constant, or a time profile lam(t) (callable, or minute-level array e.g. from the dataset CSV)
    mu    per-server service rate
    c     number of servers
    K     system capacity (servers + buffer); arrivals are blocked when K packets are in the system
and it reports the QUEUE LENGTH over time (telemetry), together with blocking and waiting time.
Several slices can share one pool of servers with non-preemptive priority (URLLC > eMBB > mMTC).

    from src.traffic_generator import SliceTraffic, run_generator
    out = run_generator({"URLLC": SliceTraffic(lam=6, mu=2, c=10, K=20)}, horizon=3600, sample_every=10)
"""
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence, Union

import numpy as np

LamSpec = Union[float, Sequence[float], Callable[[float], float]]


@dataclass
class SliceTraffic:
    lam: LamSpec          # packets/s: number, callable lam(t_seconds), or per-minute array
    mu: float             # per-server service rate (packets/s)
    c: int                # servers
    K: int                # capacity incl. servers (buffer = K - c)
    name: str = ""

    def lam_at(self, t):
        if callable(self.lam):
            return float(self.lam(t))
        if np.isscalar(self.lam):
            return float(self.lam)
        arr = np.asarray(self.lam, float)
        return float(arr[min(int(t // 60), len(arr) - 1)])

    def lam_max(self, horizon):
        if callable(self.lam):
            return max(float(self.lam(t)) for t in np.linspace(0, horizon, 400))
        return float(np.max(self.lam))


def run_generator(slices, horizon, sample_every=1.0, seed=0, priority=False, pool_servers=None, warmup=0.0):
    """
    Generate Poisson traffic (non-homogeneous via thinning when lam varies) and run the queues.
    slices   : dict name -> SliceTraffic (order = priority order when priority=True)
    priority : False = each slice has its own c servers and K (isolated);
               True  = ONE shared pool of `pool_servers` servers (default sum of c) and capacity sum of K,
                       non-preemptive priority in dict order.
    Returns dict with per-slice summary ('PB','Wq','W','Lq','L','arrivals','blocked') and the queue-length
    time series ('t', 'queue_len' per slice, 'in_system' per slice) sampled every `sample_every` seconds.
    """
    import simpy
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    env = simpy.Environment()
    names = list(slices)
    nS = len(names)
    st = {n: dict(arr=0, blk=0, served=0, wsum=0.0) for n in names}
    queues = {n: deque() for n in names}
    busy = {n: 0 for n in names}
    pool_c = pool_servers if pool_servers is not None else sum(slices[n].c for n in names)
    pool_K = sum(slices[n].K for n in names)
    pool_busy = [0]
    ts = np.arange(0.0, horizon + 1e-9, sample_every)
    qlen = {n: np.zeros(len(ts)) for n in names}
    insys = {n: np.zeros(len(ts)) for n in names}
    last = [0.0]
    # time-weighted averages (exact): update before every state change
    tw = {n: dict(q=0.0, s=0.0) for n in names}

    def tick():
        now = env.now
        dt = now - max(last[0], warmup) if now > warmup else 0.0
        if dt > 0:
            for n in names:
                tw[n]["q"] += dt * len(queues[n])
                tw[n]["s"] += dt * (len(queues[n]) + busy[n])
        last[0] = now

    def can_start(n):
        return (pool_busy[0] < pool_c) if priority else (busy[n] < slices[n].c)

    def in_system(n):
        return len(queues[n]) + busy[n]

    def full(n):
        if priority:
            return sum(in_system(x) for x in names) >= pool_K
        return in_system(n) >= slices[n].K

    def start(n, t_arr):
        busy[n] += 1; pool_busy[0] += 1
        if env.now > warmup:
            st[n]["served"] += 1; st[n]["wsum"] += env.now - t_arr
        ev = env.timeout(rng.expovariate(slices[n].mu))
        ev.callbacks.append(lambda _e, n=n: depart(n))

    def depart(n):
        tick()
        busy[n] -= 1; pool_busy[0] -= 1
        order = names if priority else [n]
        for x in order:                                   # highest priority first (own slice only if isolated)
            if queues[x] and can_start(x):
                start(x, queues[x].popleft())
                break

    def arrive(n):
        tick()
        if env.now > warmup:
            st[n]["arr"] += 1
        if full(n):
            if env.now > warmup:
                st[n]["blk"] += 1
            return
        if can_start(n):
            start(n, env.now)
        else:
            queues[n].append(env.now)

    def source(n):
        sl = slices[n]
        lmax = max(sl.lam_max(horizon), 1e-9)
        while True:                                       # thinning of a rate-lmax Poisson process
            yield env.timeout(nprng.exponential(1.0 / lmax))
            if nprng.random() < sl.lam_at(env.now) / lmax:
                arrive(n)

    def sampler():
        for i, t in enumerate(ts):
            yield env.timeout(max(t - env.now, 0.0))
            for n in names:
                qlen[n][i] = len(queues[n]); insys[n][i] = in_system(n)

    for n in names:
        env.process(source(n))
    env.process(sampler())
    env.run(until=horizon)
    tick()
    T = max(horizon - warmup, 1e-9)
    summ = {}
    for n in names:
        a = max(st[n]["arr"], 1)
        wq = st[n]["wsum"] / max(st[n]["served"], 1)
        summ[n] = dict(arrivals=st[n]["arr"], blocked=st[n]["blk"], PB=st[n]["blk"] / a, Wq=wq,
                       W=wq + 1.0 / slices[n].mu, Lq=tw[n]["q"] / T, mean_in_queue_sampled=float(qlen[n].mean()))
    return {"summary": summ, "t": ts, "queue_len": qlen, "in_system": insys}
