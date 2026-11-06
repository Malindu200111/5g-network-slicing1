"""
Queueing models for the slices.
  A) mmck_metrics      - analytical M/M/c/K (fast, used inside the environment / reward)
  B) simulate_mmck     - SimPy single-slice simulation (validation)
  C) simulate_priority - SimPy 3-slice non-preemptive priority simulation (validation)
"""
import numpy as np


# ---------------------------------------------------------------------
# A) Analytical M/M/c/K (log-space, stable for c, K ~ 100)
# ---------------------------------------------------------------------
def mmck_metrics(lam, mu, c, K):
    if lam <= 1e-12:
        return {"PB": 0.0, "lam_eff": 0.0, "L": 0.0, "Lq": 0.0, "Wq": 0.0, "W": 1.0 / mu, "util": 0.0}
    a = lam / mu
    n = np.arange(K + 1)
    logfact = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, K + 1)))))
    logt = np.where(
        n <= c,
        n * np.log(a) - logfact[n],
        n * np.log(a) - logfact[c] - (n - c) * np.log(c),
    )
    p = np.exp(logt - logt.max())
    p /= p.sum()
    PB = float(p[K])                       # blocking probability
    lam_eff = lam * (1.0 - PB)             # eq. (3.1)
    L = float((n * p).sum())
    Lq = float((np.maximum(n - c, 0) * p).sum())
    Wq = Lq / lam_eff if lam_eff > 0 else 0.0   # Little's law, eq. (3.2)
    return {"PB": PB, "lam_eff": lam_eff, "L": L, "Lq": Lq, "Wq": Wq,
            "W": Wq + 1.0 / mu, "util": lam_eff / (c * mu)}


def mmck_metrics_batch(lams, mu, c, K):
    """Vectorised M/M/c/K over an array of arrival rates (one per minute). Returns arrays."""
    lams = np.maximum(np.asarray(lams, dtype=float), 1e-9)
    a = lams / mu
    n = np.arange(K + 1)
    logfact = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, K + 1)))))
    la = np.log(a)[:, None]
    logt = np.where(n <= c, n * la - logfact[n], n * la - logfact[c] - (n - c) * np.log(c))
    p = np.exp(logt - logt.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    PB = p[:, K]
    lam_eff = lams * (1.0 - PB)
    Lq = (np.maximum(n - c, 0) * p).sum(axis=1)
    Wq = Lq / np.maximum(lam_eff, 1e-12)
    return {"PB": PB, "lam_eff": lam_eff, "Wq": Wq, "util": lam_eff / (c * mu)}


# ---------------------------------------------------------------------
# B) SimPy single slice
# ---------------------------------------------------------------------
def simulate_mmck(lam, mu, c, K, sim_time=20000, warmup=1000, seed=1):
    import simpy
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    servers = simpy.Resource(env, capacity=c)
    st = {"n": 0, "arr": 0, "blk": 0, "wq": [], "w": [], "area": 0.0, "last": 0.0}

    def upd():
        if env.now > warmup:
            st["area"] += st["n"] * (env.now - max(st["last"], warmup))
        st["last"] = env.now

    def packet():
        t0 = env.now
        counted = t0 > warmup
        st["arr"] += counted
        upd()
        if st["n"] >= K:
            st["blk"] += counted
            return
        st["n"] += 1
        with servers.request() as req:
            yield req
            if counted:
                st["wq"].append(env.now - t0)
            yield env.timeout(rng.exponential(1.0 / mu))
        upd()
        st["n"] -= 1
        if counted:
            st["w"].append(env.now - t0)

    def source():
        while True:
            yield env.timeout(rng.exponential(1.0 / lam))
            env.process(packet())

    env.process(source())
    env.run(until=sim_time)
    upd()
    return {"PB": st["blk"] / max(1, st["arr"]), "Wq": float(np.mean(st["wq"])),
            "W": float(np.mean(st["w"])), "L": st["area"] / (sim_time - warmup)}


# ---------------------------------------------------------------------
# C) SimPy priority multi-slice (non-preemptive, shared servers and K)
# ---------------------------------------------------------------------
def simulate_priority(lams, mu, c, K, slices=("URLLC", "eMBB", "mMTC"),
                      sim_time=20000, warmup=1000, seed=1):
    import simpy
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    servers = simpy.PriorityResource(env, capacity=c)
    st = {"n": 0, "area": 0.0, "last": 0.0}
    rec = {s: {"arr": 0, "blk": 0, "wq": [], "w": []} for s in slices}

    def upd():
        if env.now > warmup:
            st["area"] += st["n"] * (env.now - max(st["last"], warmup))
        st["last"] = env.now

    def packet(name, prio):
        t0 = env.now
        counted = t0 > warmup
        rec[name]["arr"] += counted
        upd()
        if st["n"] >= K:
            rec[name]["blk"] += counted
            return
        st["n"] += 1
        with servers.request(priority=prio) as req:
            yield req
            if counted:
                rec[name]["wq"].append(env.now - t0)
            yield env.timeout(rng.exponential(1.0 / mu))
        upd()
        st["n"] -= 1
        if counted:
            rec[name]["w"].append(env.now - t0)

    def source(name, prio, lam):
        while True:
            yield env.timeout(rng.exponential(1.0 / lam))
            env.process(packet(name, prio))

    for prio, name in enumerate(slices):
        env.process(source(name, prio, lams[name]))
    env.run(until=sim_time)
    upd()
    out = {s: {"PB": r["blk"] / max(1, r["arr"]),
               "Wq": float(np.mean(r["wq"])) if r["wq"] else 0.0,
               "W": float(np.mean(r["w"])) if r["w"] else 0.0} for s, r in rec.items()}
    tot_a = sum(r["arr"] for r in rec.values())
    tot_b = sum(r["blk"] for r in rec.values())
    out["aggregate"] = {"PB": tot_b / max(1, tot_a), "L": st["area"] / (sim_time - warmup)}
    return out
