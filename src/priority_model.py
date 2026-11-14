"""
Analytical waiting times for the priority-based queue (Objective 1 / Deliverable 2).

Model: non-preemptive priority M/M/c, K priority classes, SAME service rate mu for all classes,
class 1 = highest priority (URLLC), then eMBB, then mMTC, infinite buffer (needs rho < 1).

    a      = lambda_total / mu                      offered load (Erlangs)
    rho    = a / c
    C(c,a) = Erlang-C probability that an arrival has to wait
    sigma_k = sum_{i<=k} lambda_i / (c * mu)         cumulative load of classes 1..k

    Wq_k = C(c,a) / (c * mu) / ((1 - sigma_{k-1}) * (1 - sigma_k))          (Cobham-type result)

Check: the lambda-weighted average of Wq_k equals the FCFS value C / (c*mu*(1-rho)) (conservation law).
For a FINITE buffer (M/M/c/K) there is no simple closed form; the formula is then an approximation
that is accurate when blocking is small. The SimPy priority simulation is used to check it.
"""
import math

import numpy as np


def erlang_c(c, a):
    """Probability of waiting in M/M/c (a = lambda/mu, requires a < c)."""
    if a >= c:
        raise ValueError("Erlang-C needs a < c (rho < 1)")
    if a <= 0:
        return 0.0
    rho = a / c
    log_term = c * math.log(a) - math.lgamma(c + 1) - math.log(1.0 - rho)
    s = sum(math.exp(n * math.log(a) - math.lgamma(n + 1)) for n in range(c))
    top = math.exp(log_term)
    return top / (s + top)


def priority_wq_nonpreemptive(lams, mu, c):
    """lams: arrival rates ordered from HIGHEST to LOWEST priority. Returns (Wq per class, Wq FCFS)."""
    lam_tot = sum(lams)
    a = lam_tot / mu
    rho = a / c
    if rho >= 1.0:
        raise ValueError(f"rho = {rho:.3f} >= 1: the infinite-buffer formula does not apply")
    base = erlang_c(c, a) / (c * mu)
    out, sigma_prev = [], 0.0
    for lam_k in lams:
        sigma_k = sigma_prev + lam_k / (c * mu)
        out.append(base / ((1.0 - sigma_prev) * (1.0 - sigma_k)))
        sigma_prev = sigma_k
    wq_fcfs = base / (1.0 - rho)
    return out, wq_fcfs


def priority_ctmc(lams, mu, c, K):
    """
    EXACT finite-buffer non-preemptive priority M/M/c/K with identical exponential service (classes ordered from the
    highest to the lowest priority, one shared buffer of K places in total, arrivals blocked when the system holds K).
    Because every server has the same rate, the departure rate is mu*(busy servers) whatever class is in service, so the
    class composition of the SERVERS does not matter. The Markov state is therefore
        n < c  : n packets, all in service                      ->  state ("s", n)
        n >= c : all c servers busy and queue (q_1, q_2, q_3)    ->  state (q_1, q_2, q_3), q_1+q_2+q_3 <= K-c
    A freed server always takes the highest-priority waiting packet. Solve pi Q = 0 (sparse), then
        P_B (same for every class, PASTA) = P(n = K),   Lq_k = E[q_k],   Wq_k = Lq_k / (lam_k (1 - P_B))   (Little).
    """
    import itertools

    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    lams = [float(x) for x in lams]
    m = len(lams)
    Q_ = K - c
    states = [("s", n) for n in range(c)]
    states += [q for q in itertools.product(range(Q_ + 1), repeat=m) if sum(q) <= Q_]
    idx = {st: i for i, st in enumerate(states)}
    rows, cols, vals = [], [], []

    def add(a, b, r):
        rows.append(idx[a]); cols.append(idx[b]); vals.append(r)

    zero = tuple([0] * m)
    for st in states:
        if st[0] == "s":                                   # n < c packets, no queue
            n = st[1]
            nxt = ("s", n + 1) if n + 1 < c else zero      # n + 1 == c -> all servers busy, empty queue
            for k in range(m):
                add(st, nxt, lams[k])
            if n > 0:
                add(st, ("s", n - 1), mu * n)
        else:                                              # all servers busy, queue q
            q = st
            tot = sum(q)
            if tot < Q_:
                for k in range(m):
                    add(st, tuple(q[j] + (j == k) for j in range(m)), lams[k])
            if tot > 0:                                    # departure: highest-priority waiting packet enters service
                j = next(j for j in range(m) if q[j] > 0)
                add(st, tuple(q[x] - (x == j) for x in range(m)), mu * c)
            else:
                add(st, ("s", c - 1), mu * c) if c > 0 else None
    N = len(states)
    Qm = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    Qm = Qm - sp.diags(np.asarray(Qm.sum(axis=1)).ravel())
    A = Qm.T.tolil()
    A[0, :] = 1.0                                          # replace one balance equation by sum(pi) = 1
    b = np.zeros(N); b[0] = 1.0
    pi = spl.spsolve(A.tocsc(), b)
    pb = float(sum(pi[idx[q]] for q in states if q[0] != "s" and sum(q) == Q_)) if Q_ > 0 else float(pi[idx[zero]])
    lq = np.array([sum(pi[idx[q]] * q[k] for q in states if q[0] != "s") for k in range(m)])
    lam_eff = np.array(lams) * (1.0 - pb)
    wq = lq / np.maximum(lam_eff, 1e-12)
    return {"PB": np.full(m, pb), "Wq": wq, "W": wq + 1.0 / mu, "Lq": lq}


def priority_metrics_mmck(lams, mu, c, K):
    """
    Finite-buffer priority model (approximation). Classes are ordered from highest to lowest priority.
      1. The shared M/M/c/K queue gives ONE blocking probability and ONE mean waiting time (FIFO) for all packets;
         per-class blocking is equal (checked against SimPy), carried rate of class k = lam_k (1 - P_B).
      2. The Cobham-type formula (infinite buffer, carried rates, rho clipped below 1) gives the RELATIVE waiting
         times of the classes.
      3. They are rescaled so that the lambda-weighted mean equals the M/M/c/K waiting time (conservation law for
         work-conserving disciplines): Wq_k = Wq_k(Cobham) * Wq(M/M/c/K) / Wq_FCFS(Cobham).
    Returns per-class P_B, Wq and W.
    """
    from .queuing_model import mmck_metrics
    lams = np.asarray(lams, dtype=float)
    agg = mmck_metrics(float(lams.sum()), mu, c, K)
    carried = lams * (1.0 - agg["PB"])
    cap = 0.999 * c * mu
    if carried.sum() > cap:
        carried = carried * cap / carried.sum()
    wq_inf, wq_fcfs_inf = priority_wq_nonpreemptive(carried, mu, c)
    wq = np.asarray(wq_inf) * agg["Wq"] / max(wq_fcfs_inf, 1e-12)
    return {"PB": np.full(len(lams), agg["PB"]), "Wq": wq, "W": wq + 1.0 / mu}
