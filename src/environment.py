"""
Slicing environment: CSV lambda(t) -> M/M/c/K queues -> state / reward (for the DQN agent).
1 episode = 1 day = 288 decisions (every 5 minutes).
Actions: 0 = keep, 1..6 = move `move_size` PRBs from slice i to slice j.
"""
import numpy as np
import pandas as pd

from .config import ASSUMED, CSV_COLS, SLICES
from .queuing_model import mmck_metrics_batch


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


class SlicingEnv:
    PAIRS = [(i, j) for i in range(3) for j in range(3) if i != j]

    def __init__(self, data, days, cfg=ASSUMED, init_alloc=None, seed=0):
        self.lam, self.peak, self.fine, self.weekend = data["mean"], data["peak"], data["fine"], data["weekend"]
        self.cfg = cfg
        self.days = list(days)                      # 0-based day indices
        self.steps = self.lam.shape[1]
        self.rng = np.random.default_rng(seed)
        self.n_actions = 1 + len(self.PAIRS)
        self.obs_dim = 3 * 6 + 3
        self.w = np.array([cfg["slice_weights"][s] for s in SLICES])
        self.init_alloc = (np.array(init_alloc) if init_alloc is not None
                           else proportional_alloc([1, 1, 1], cfg["total_prbs"], cfg["min_alloc"]))
        self.alloc = self.init_alloc.copy()

    def _metrics(self, alloc, t):
        """
        Per-slice KPIs for decision step t. The queue is evaluated MINUTE BY MINUTE inside the
        interval (steady state per minute) and then aggregated, so spikes are not averaged away.
        """
        c = self.cfg
        out = []
        for s in range(3):
            cs = int(alloc[s])
            lam_m = self.fine[self.day, t, :, s]
            b = mmck_metrics_batch(lam_m, c["mu"], cs, cs + c["buffer_per_slice"])
            PB = float((lam_m * b["PB"]).sum() / max(lam_m.sum(), 1e-12))        # packet-weighted
            Wq = float((b["lam_eff"] * b["Wq"]).sum() / max(b["lam_eff"].sum(), 1e-12))
            out.append({"PB": PB, "Wq": Wq, "lam_eff": float(b["lam_eff"].mean()),
                        "util": float(b["lam_eff"].mean() / (cs * c["mu"]))})
        return out

    def _obs(self, t):
        m = self._metrics(self.alloc, t)
        tot = self.cfg["total_prbs"]
        f = []
        for s in range(3):
            f += [self.lam[self.day, t, s] / tot, self.peak[self.day, t, s] / tot,
                  self.alloc[s] / tot, m[s]["util"], m[s]["PB"],
                  min(m[s]["Wq"], self.cfg["wq_cap"]) / self.cfg["wq_cap"]]
        ang = 2 * np.pi * t / self.steps
        f += [np.sin(ang), np.cos(ang), float(self.weekend[self.day])]
        return np.array(f, dtype=np.float32)

    def apply_action(self, action):
        new = self.alloc.copy()
        if action == 0:
            return new, 0
        i, j = self.PAIRS[action - 1]
        mv = self.cfg["move_size"]
        if new[i] - mv < self.cfg["min_alloc"]:
            return new, 0                           # invalid move -> no-op
        new[i] -= mv
        new[j] += mv
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
        wq = np.array([min(x["Wq"], c["wq_cap"]) / c["wq_cap"] for x in m])
        reward = -float((self.w * (c["w_pb"] * pb + c["w_wq"] * wq)).sum()) - c["move_cost"] * (moved > 0)
        info = {"alloc": self.alloc.copy(), "metrics": m, "lam": self.lam[self.day, self.t].copy()}
        self.t += 1
        done = self.t >= self.steps
        obs = self._obs(self.t) if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, done, False, info


# ----- baseline policies (same action space the DQN will use) -----
def policy_static(env, obs):
    return 0


def policy_proportional(env, obs):
    lam_now = env.lam[env.day, env.t]
    target = proportional_alloc(lam_now, env.cfg["total_prbs"], env.cfg["min_alloc"])
    diff = target - env.alloc
    i, j = int(np.argmin(diff)), int(np.argmax(diff))
    if i == j or diff[j] < env.cfg["move_size"] // 2 + 1:
        return 0
    return 1 + SlicingEnv.PAIRS.index((i, j))


def evaluate(env, policy, days, init_alloc=None):
    tot_lam = np.zeros(3); blocked = np.zeros(3); wq_num = np.zeros(3); carried = np.zeros(3)
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
    n_steps = len(days) * env.steps
    return {"PB": blocked / np.maximum(tot_lam, 1e-12),
            "Wq": wq_num / np.maximum(carried, 1e-12),
            "util": carried.sum() / (env.cfg["total_prbs"] * n_steps * env.cfg["mu"]),
            "return_per_day": ret / len(days),
            "moves_per_day": n_moves / len(days)}


def print_table(results):
    print(f"{'Policy':<18}{'PB URLLC':>10}{'PB eMBB':>10}{'PB mMTC':>10}"
          f"{'Wq URLLC':>10}{'Wq eMBB':>10}{'Util':>8}{'Moves/d':>9}{'Reward/d':>10}")
    for name, r in results.items():
        print(f"{name:<18}{r['PB'][0]:>10.4f}{r['PB'][1]:>10.4f}{r['PB'][2]:>10.4f}"
              f"{r['Wq'][0]:>10.3f}{r['Wq'][1]:>10.3f}{r['util']:>8.3f}"
              f"{r['moves_per_day']:>9.1f}{r['return_per_day']:>10.2f}")
