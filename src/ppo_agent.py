"""PPO agent (clipped objective, GAE) for the same discrete action space the DQN uses."""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(inp, out, hidden):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.pi, self.v = _mlp(obs_dim, n_actions, hidden), _mlp(obs_dim, 1, hidden)

    def dist(self, x):
        return Categorical(logits=self.pi(x))

    def value(self, x):
        return self.v(x).squeeze(-1)


class PPOAgent:
    def __init__(self, obs_dim, n_actions, hp, device="cpu", seed=0):
        self.hp, self.device = hp, device
        torch.manual_seed(seed)
        self.net = ActorCritic(obs_dim, n_actions, hp["hidden"]).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=hp["lr"])
        self.rng = np.random.default_rng(seed)
        self.buf = {"o": [], "a": [], "logp": [], "r": [], "v": [], "d": []}

    # ---- acting ----
    def act(self, obs):
        """Sample an action; remember what the update needs."""
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            d = self.net.dist(o)
            a = d.sample()
            return int(a), float(d.log_prob(a)), float(self.net.value(o))

    def greedy(self, obs):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return int(torch.argmax(self.net.pi(o)))

    def remember(self, o, a, logp, r, v, done):
        for k, x in zip(("o", "a", "logp", "r", "v", "d"), (o, a, logp, r * self.hp["reward_scale"], v, done)):
            self.buf[k].append(x)

    # ---- learning ----
    def update(self):
        hp, b = self.hp, self.buf
        r, v, d = (np.array(b[k], dtype=np.float32) for k in ("r", "v", "d"))
        adv, last = np.zeros_like(r), 0.0
        for t in reversed(range(len(r))):                      # GAE; episodes end at done=True
            nxt_v = 0.0 if (d[t] or t == len(r) - 1) else v[t + 1]
            delta = r[t] + hp["gamma"] * nxt_v - v[t]
            last = delta + hp["gamma"] * hp["gae_lambda"] * (0.0 if d[t] else last)
            adv[t] = last
        ret = adv + v
        o = torch.as_tensor(np.array(b["o"]), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.array(b["a"]), dtype=torch.int64, device=self.device)
        logp_old = torch.as_tensor(np.array(b["logp"]), dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)
        n = len(r)
        for _ in range(hp["epochs"]):
            idx = self.rng.permutation(n)
            for s in range(0, n, hp["minibatch"]):
                m = torch.as_tensor(idx[s:s + hp["minibatch"]], device=self.device)
                dist = self.net.dist(o[m])
                ratio = torch.exp(dist.log_prob(a[m]) - logp_old[m])
                surr = torch.min(ratio * adv_t[m], torch.clamp(ratio, 1 - hp["clip"], 1 + hp["clip"]) * adv_t[m])
                loss = (-surr.mean() + hp["vf_coef"] * (self.net.value(o[m]) - ret_t[m]).pow(2).mean()
                        - hp["ent_coef"] * dist.entropy().mean())
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), hp["max_grad"])
                self.opt.step()
        for k in self.buf:
            self.buf[k].clear()

    # ---- checkpoints (same interface as DQNAgent) ----
    def get_state(self):
        return copy.deepcopy(self.net.state_dict())

    def set_state(self, state):
        self.net.load_state_dict(state)

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
