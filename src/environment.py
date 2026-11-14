"""
Slicing environment: CSV lambda(t) -> M/M/c/K queues -> state / reward (for the DQN agent).
1 episode = 1 day = 1440 / decision_minutes decisions (96 at the default 15 minutes).
Two resources are allocated separately (allocation array has shape (2, 3)):
  alloc[0] = bandwidth  (PRBs per slice)           alloc[1] = server capacity (servers per slice)
Actions (13): 0 = keep | 1..6 = move `move_size` PRBs from slice i to slice j |
              7..12 = move `server_move_size` servers from slice i to slice j.
A slice with b PRBs and c servers is an M/M/c/K queue whose per-server service rate is
  1/mu_eff = 1/mu_server + c/(b * rate_per_prb)   (compute time + radio transmission time),
so BOTH resources change the queue (see service_rate).
Decisions are causal: when deciding for an interval, policies only see traffic and KPIs from the
most recently completed interval (plus the current allocation and the clock/calendar).
"""
import numpy as np
import pandas as pd

from .config import ASSUMED, CSV_COLS, SLICES
from .queuing_model import mmck_metrics_batch, mmck_metrics_frac_c


def load_data(csv_path, decision_minutes=5):
    """
    Return dict with:
      mean[day, step, slice]  interval-mean lambda
      peak[day, step, slice]  interval-peak lambda (shows spikes hidden by the mean)
      fine[day, step, minute, slice]  minute-level lambda inside each interval
      weekend[day]
    """
    if 1440 % decision_minutes:
        raise ValueError("decision_minutes must divide 1440 (e.g. 5, 10, 15, 30, 60).")
    df = pd.read_csv(csv_path).sort_values("time_minute")
    n_days = int(df["day_number"].max())
    if len(df) != n_days * 1440:
        raise ValueError("CSV must contain exactly 1440 rows per day.")
    arr = df[CSV_COLS].to_numpy(dtype=float)
    steps = 1440 // decision_minutes
    fine = arr.reshape(n_days, steps, decision_minutes, 3)
    weekend = df.groupby("day_number")["is_weekend"].first().to_numpy().astype(int)
    return {"mean": fine.mean(axis=2), "peak": fine.max(axis=2), "fine": fine, "weekend": weekend}


def slice_param(cfg, key):
    """Per-slice parameter array in SLICES order (cfg[key] may be a dict by slice name or a single number)."""
    v = cfg[key]
    return np.array([v[s] for s in SLICES], dtype=float) if isinstance(v, dict) else np.full(3, float(v))


def buffers(cfg):
    """Buffer size (waiting places) of each slice as a list of ints."""
    return [int(x) for x in slice_param(cfg, "buffer_per_slice")]


def service_rate(prbs, servers, cfg, s=None):
    """Per-server service rate (pkts/s). `s` = slice index, or None when the last axis of the inputs is the slice."""
    mu_srv = slice_param(cfg, "mu_server")
    mu_srv = mu_srv if s is None else mu_srv[s]
    return 1.0 / (1.0 / mu_srv + servers / (prbs * cfg["rate_per_prb"]))


def nominal_capacity(cfg):
    """Total service capacity (pkts/s) of the equal split; used only to normalise traffic in the state."""
    b, c = cfg["total_prbs"] / 3, cfg["total_servers"] / 3
    return float(sum(c * service_rate(b, c, cfg, s) for s in range(3)))


def proportional_alloc(weights, total, min_alloc):
    """Integer allocation proportional to weights, each >= min_alloc, sums to total."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    n = len(w)
    raw = min_alloc + (total - n * min_alloc) * w
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: int(total - base.sum())]:
        base[i] += 1
    return base


def make_alloc(weights, cfg):
    """Both resources split proportionally to `weights`: returns array [[PRBs x3], [servers x3]]."""
    return np.vstack([proportional_alloc(weights, cfg["total_prbs"], cfg["min_alloc"]),
                      proportional_alloc(weights, cfg["total_servers"], cfg["min_servers"])])


class SlicingEnv:
    PAIRS = [(i, j) for i in range(3) for j in range(3) if i != j]

    def __init__(self, data, days, cfg=ASSUMED, init_alloc=None, seed=0):
        self.lam, self.peak, self.fine, self.weekend = data["mean"], data["peak"], data["fine"], data["weekend"]
        self.cfg = cfg
        self.days = list(days)                      # 0-based day indices
        self.steps = self.lam.shape[1]
        self.rng = np.random.default_rng(seed)
        self.n_actions = 1 + 2 * len(self.PAIRS)        # keep + PRB moves + server moves
        self.obs_dim = 3 * 8 + 3
        self.nominal_cap = nominal_capacity(cfg)
        self.w = np.array([cfg["slice_weights"][s] for s in SLICES])
        self.init_alloc = (np.array(init_alloc) if init_alloc is not None
                           else make_alloc([1, 1, 1], cfg))
        self.alloc = self.init_alloc.copy()

    def last_interval(self, t):
        """(day, step) of the last COMPLETED interval when deciding for step t of the current day."""
        if t > 0:
            return self.day, t - 1
        if self.day > 0:
            return self.day - 1, self.steps - 1
        return 0, 0                                 # very first interval of the dataset: nothing earlier exists

    def _metrics_priority(self, alloc, t, day=None):
        """
        Priority-sharing KPIs. Each slice keeps its guaranteed c_s servers; servers that other slices leave
        idle are lent in priority order URLLC > eMBB > mMTC (non-preemptive). Mean-field surrogate, per minute:
          busy_j = carried_j / mu_j (isolated M/M/c/K), idle_j = (c_j - busy_j)+
          want_s = (offered_s / target_util) - c_s            (extra servers slice s would like)
          borrowed_s = eta * min(want_s, idle of the OTHER slices left after higher priorities)
        and slice s is then an M/M/(c_s + borrowed_s)/K_s queue (K_s = c_s + buffer, as in the SimPy replay).
        eta and target_util are fitted to the SimPy priority replay (calibrate_priority.py).
        """
        c = self.cfg
        day = self.day if day is None else day
        mus, cs, lam_m, iso = [], [], [], []
        for s in range(3):
            b, k = int(alloc[0, s]), int(alloc[1, s])
            mu = service_rate(b, k, c, s)
            lm = self.fine[day, t, :, s]
            mus.append(mu); cs.append(k); lam_m.append(lm)
            iso.append(mmck_metrics_batch(lm, mu, k, k + buffers(c)[s]))
        idle = np.array([np.maximum(cs[s] - iso[s]["lam_eff"] / mus[s], 0.0) for s in range(3)])   # (3, minutes)
        out = []
        for s in range(3):                                                       # priority order
            offered = np.maximum(lam_m[s], 1e-9) / mus[s]
            want = np.maximum(offered / c["priority_target_util"] - cs[s], 0.0)
            others = idle.sum(axis=0) - idle[s]
            take = np.minimum(want, others)
            if take.max() > 0:                                                   # remove what was lent
                share = np.where(others > 0, take / np.maximum(others, 1e-12), 0.0)
                for j in range(3):
                    if j != s:
                        idle[j] = idle[j] * (1 - share)
            k = cs[s]
            m = mmck_metrics_frac_c(lam_m[s], mus[s], k + c["priority_eta"] * take, k + buffers(c)[s])
            PB = float((lam_m[s] * m["PB"]).sum() / max(lam_m[s].sum(), 1e-12))
            Wq = float((m["lam_eff"] * m["Wq"]).sum() / max(m["lam_eff"].sum(), 1e-12))
            cap = k * mus[s]
            Lq = float((m["lam_eff"] * m["Wq"]).mean())
            out.append({"PB": PB, "Wq": Wq, "lam_eff": float(m["lam_eff"].mean()),
                        "Lq": Lq, "L": Lq + float(m["lam_eff"].mean()) / mus[s],
                        "util": float(m["lam_eff"].mean() / cap), "mu": mus[s], "cap": cap})
        return out

    def _metrics(self, alloc, t, day=None):
        if self.cfg.get("priority_sharing", False):
            return self._metrics_priority(alloc, t, day)
        return self._metrics_isolated(alloc, t, day)

    def _metrics_isolated(self, alloc, t, day=None):
        """
        Per-slice KPIs for step t of `day` (default: current day). The queue is evaluated MINUTE BY
        MINUTE inside the interval (steady state per minute) and then aggregated, so spikes are not
        averaged away.
        """
        c = self.cfg
        day = self.day if day is None else day
        out = []
        for s in range(3):
            b, cs = int(alloc[0, s]), int(alloc[1, s])
            mu = service_rate(b, cs, c, s)
            cap = cs * mu
            lam_m = self.fine[day, t, :, s]
            m = mmck_metrics_batch(lam_m, mu, cs, cs + buffers(c)[s])
            PB = float((lam_m * m["PB"]).sum() / max(lam_m.sum(), 1e-12))        # packet-weighted
            Wq = float((m["lam_eff"] * m["Wq"]).sum() / max(m["lam_eff"].sum(), 1e-12))
            Lq = float((m["lam_eff"] * m["Wq"]).mean())                     # Little's law: Lq = lam_eff * Wq
            out.append({"PB": PB, "Wq": Wq, "lam_eff": float(m["lam_eff"].mean()),
                        "Lq": Lq, "L": Lq + float(m["lam_eff"].mean()) / mu,   # queue / system length (packets)
                        "util": float(m["lam_eff"].mean() / cap), "mu": mu, "cap": cap})
        return out

    def _obs(self, t):
        day, k = self.last_interval(t)
        m = self._metrics(self.alloc, k, day)
        tot, nb, ns = self.nominal_cap, self.cfg["total_prbs"], self.cfg["total_servers"]
        f = []
        buf = buffers(self.cfg)
        for s in range(3):
            f += [self.lam[day, k, s] / tot, self.peak[day, k, s] / tot,
                  self.alloc[0, s] / nb, self.alloc[1, s] / ns, m[s]["util"], m[s]["PB"],
                  min(m[s]["Wq"], self.cfg["wq_cap"]) / self.cfg["wq_cap"],
                  min(m[s]["Lq"] / buf[s], 1.0)]                     # queue length / buffer size
        ang = 2 * np.pi * t / self.steps
        f += [np.sin(ang), np.cos(ang), float(self.weekend[self.day])]
        return np.array(f, dtype=np.float32)

    def apply_action(self, action):
        new = self.alloc.copy()
        if action == 0:
            return new, 0
        r, p = divmod(action - 1, len(self.PAIRS))      # r = 0: bandwidth (PRBs), r = 1: server capacity
        i, j = self.PAIRS[p]
        mv, mn = ((self.cfg["move_size"], self.cfg["min_alloc"]) if r == 0
                  else (self.cfg["server_move_size"], self.cfg["min_servers"]))
        if new[r, i] - mv < mn:
            return new, 0                           # invalid move -> no-op
        new[r, i] -= mv
        new[r, j] += mv
        return new, mv

    def reset(self, day=None, init_alloc=None):
        self.day = int(day) if day is not None else int(self.rng.choice(self.days))
        self.alloc = (np.array(init_alloc) if init_alloc is not None else self.init_alloc).copy()
        self.t = 0
        return self._obs(0), {"day": self.day}

    def step(self, action):
        c = self.cfg
        self.alloc, moved = self.apply_action(action)
        m = self._metrics(self.alloc, self.t)
        pb = np.array([x["PB"] for x in m])
        # latency term uses the full delay W = Wq + 1/mu_eff (waiting + service), so slowing a slice's service
        # (few PRBs per server) is punished as well, not only its waiting time
        wq = np.array([min(x["Wq"] + 1.0 / x["mu"], c["wq_cap"]) / c["wq_cap"] for x in m])
        waste = 1.0 - sum(x["lam_eff"] for x in m) / max(sum(x["cap"] for x in m), 1e-12)   # idle share of capacity
        reward = (-float((self.w * (c["w_pb"] * pb + c["w_wq"] * wq)).sum())
                  - c.get("w_waste", 0.0) * waste - c["move_cost"] * (moved > 0))
        info = {"alloc": self.alloc.copy(), "metrics": m, "lam": self.lam[self.day, self.t].copy(), "waste": waste}
        self.t += 1
        done = self.t >= self.steps
        obs = self._obs(self.t) if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, done, False, info


# ----- baseline policies (same action space the DQN will use) -----
def policy_static(env, obs):
    return 0


def policy_proportional(env, obs):
    """Heuristic: move the resource (PRBs or servers) that is furthest from its proportional target."""
    day, k = env.last_interval(env.t)
    c, best = env.cfg, None
    for r, (tot, mn, mv) in enumerate([(c["total_prbs"], c["min_alloc"], c["move_size"]),
                                       (c["total_servers"], c["min_servers"], c["server_move_size"])]):
        diff = proportional_alloc(env.lam[day, k], tot, mn) - env.alloc[r]
        i, j = int(np.argmin(diff)), int(np.argmax(diff))
        if i != j and diff[j] >= mv // 2 + 1 and (best is None or diff[j] / tot > best[0]):
            best = (diff[j] / tot, r, i, j)
    if best is None:
        return 0
    _, r, i, j = best
    return 1 + r * len(SlicingEnv.PAIRS) + SlicingEnv.PAIRS.index((i, j))


def evaluate(env, policy, days, init_alloc=None):
    tot_lam = np.zeros(3); blocked = np.zeros(3); wq_num = np.zeros(3); w_num = np.zeros(3)
    carried = np.zeros(3); cap_sum = 0.0; lq_sum = np.zeros(3); l_sum = np.zeros(3)
    ret = 0.0; n_moves = 0
    for d in days:
        obs, _ = env.reset(day=d, init_alloc=init_alloc)
        done = False
        while not done:
            a = policy(env, obs)
            before = env.alloc.copy()
            obs, r, done, _, info = env.step(a)
            n_moves += int((before != info["alloc"]).any())
            ret += r
            for s in range(3):
                m = info["metrics"][s]
                tot_lam[s] += info["lam"][s]
                blocked[s] += info["lam"][s] * m["PB"]
                carried[s] += m["lam_eff"]
                wq_num[s] += m["lam_eff"] * m["Wq"]
                w_num[s] += m["lam_eff"] * (m["Wq"] + 1.0 / m["mu"])
                cap_sum += m["cap"]; lq_sum[s] += m["Lq"]; l_sum[s] += m["L"]
    n_steps = len(days) * env.steps
    return {"PB": blocked / np.maximum(tot_lam, 1e-12),
            "Wq": wq_num / np.maximum(carried, 1e-12),
            "W": w_num / np.maximum(carried, 1e-12),            # latency W = Wq + 1/mu_eff
            "Lq": lq_sum / n_steps, "L": l_sum / n_steps,       # mean queue length / packets in system
            "util": carried.sum() / max(cap_sum, 1e-12),        # carried / deployed capacity
            "throughput": carried.sum() / n_steps,              # mean carried packets/s
            "return_per_day": ret / len(days),
            "moves_per_day": n_moves / len(days)}


def print_table(results):
    print(f"{'Policy':<18}{'PB URLLC':>10}{'PB eMBB':>10}{'PB mMTC':>10}"
          f"{'Wq URLLC':>10}{'Wq eMBB':>10}{'Util':>8}{'Moves/d':>9}{'Reward/d':>10}")
    for name, r in results.items():
        print(f"{name:<18}{r['PB'][0]:>10.4f}{r['PB'][1]:>10.4f}{r['PB'][2]:>10.4f}"
              f"{r['Wq'][0]:>10.3f}{r['Wq'][1]:>10.3f}{r['util']:>8.3f}"
              f"{r['moves_per_day']:>9.1f}{r['return_per_day']:>10.2f}")
