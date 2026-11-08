"""DQN agent: Q-network, replay buffer, epsilon-greedy policy, target network."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.cap, self.n, self.i = capacity, 0, 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, o, a, r, no, d):
        self.obs[self.i], self.act[self.i], self.rew[self.i] = o, a, r
        self.nobs[self.i], self.done[self.i] = no, float(d)
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, size=batch)
        return (self.obs[idx], self.act[idx], self.rew[idx], self.nobs[idx], self.done[idx])

    def __len__(self):
        return self.n


class DQNAgent:
    def __init__(self, obs_dim, n_actions, hp, device="cpu", seed=0):
        self.hp, self.device, self.n_actions = hp, device, n_actions
        self.rng = np.random.default_rng([seed, 1])     # separate stream from the environment's default_rng(seed)
        self.q = QNetwork(obs_dim, n_actions, hp["hidden"]).to(device)
        self.q_target = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=hp["lr"])
        self.buffer = ReplayBuffer(hp["buffer_size"], obs_dim)
        self.learn_steps = 0

    def act(self, obs, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(obs)

    @torch.no_grad()
    def greedy(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q(x).argmax(dim=1).item())

    def remember(self, o, a, r, no, d):
        self.buffer.add(o, a, r, no, d)

    def learn(self):
        hp = self.hp
        if len(self.buffer) < max(hp["batch_size"], hp["learn_start"]):
            return None
        o, a, r, no, d = [torch.as_tensor(x, device=self.device)
                          for x in self.buffer.sample(hp["batch_size"], self.rng)]
        q_sa = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if hp["double_dqn"]:                      # optional Double-DQN target
                a_star = self.q(no).argmax(dim=1, keepdim=True)
                q_next = self.q_target(no).gather(1, a_star).squeeze(1)
            else:                                     # standard DQN target
                q_next = self.q_target(no).max(dim=1).values
            target = r + hp["gamma"] * (1.0 - d) * q_next
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        # soft update of the target network
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.mul_(1.0 - hp["tau"]).add_(hp["tau"] * p)
        self.learn_steps += 1
        return float(loss.item())

    # --- keep the best weights in memory (no disk writes during training) ---
    def get_state(self):
        return {k: v.detach().clone() for k, v in self.q.state_dict().items()}

    def set_state(self, state):
        self.q.load_state_dict(state)
        self.q_target = copy.deepcopy(self.q)

    # --- disk checkpoint: never crashes the run; returns True/False ---
    def save(self, path):
        try:
            torch.save(self.q.state_dict(), path)
            return True
        except Exception as e:                       # disk full, OneDrive/antivirus lock, ...
            print(f"WARNING: could not save checkpoint to {path}: {e} (any file already there is NOT this model)")
            return False

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.q_target = copy.deepcopy(self.q)
