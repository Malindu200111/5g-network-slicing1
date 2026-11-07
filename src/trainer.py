"""Reusable DQN training loop (used by the sweep script)."""
import numpy as np
import torch

from .agent import DQNAgent
from .environment import SlicingEnv, evaluate


def train_dqn(data, cfg, hp, train_days, val_days, init_alloc, episodes, seed,
              save_path, verbose=True):
    """Train one DQN; keep the weights with the best validation return. Returns (agent, info)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env_train = SlicingEnv(data, train_days, cfg, init_alloc=init_alloc, seed=seed)
    env_eval = SlicingEnv(data, train_days, cfg, init_alloc=init_alloc)
    agent = DQNAgent(env_train.obs_dim, env_train.n_actions, hp, device="cpu", seed=seed)
    policy = lambda env, obs: agent.greedy(obs)

    best_val, best_ep, step, best_state = -np.inf, 0, 0, None
    ep_returns, val_points = [], []
    for ep in range(1, episodes + 1):
        eps = max(hp["eps_end"], hp["eps_start"] - (hp["eps_start"] - hp["eps_end"]) * ep / (episodes * hp["eps_decay_frac"]))
        obs, _ = env_train.reset()
        done, ret = False, 0.0
        while not done:
            a = agent.act(obs, eps)
            nobs, r, done, _, _ = env_train.step(a)
            agent.remember(obs, a, r, nobs, done)
            step += 1
            if step % hp["learn_every"] == 0:
                agent.learn()
            obs, ret = nobs, ret + r
        ep_returns.append(ret)
        if ep % hp["eval_every"] == 0 or ep == episodes:
            v = evaluate(env_eval, policy, val_days, init_alloc)["return_per_day"]
            val_points.append((ep, v))
            if v > best_val:
                best_val, best_ep = v, ep
                best_state = agent.get_state()      # in memory, no disk write
            if verbose:
                print(f"  ep {ep:5d} | val {v:8.1f} (best {best_val:8.1f} @ {best_ep})", flush=True)
    agent.set_state(best_state)
    agent.save(save_path)                           # single write at the end (warns on failure)
    return agent, {"best_ep": best_ep, "best_val": best_val,
                   "ep_returns": ep_returns, "val_points": val_points}
