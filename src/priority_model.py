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


def erlang_c(c, a):
    """Probability of waiting in M/M/c (a = lambda/mu, requires a < c)."""
    if a >= c:
        raise ValueError("Erlang-C needs a < c (rho < 1)")
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
