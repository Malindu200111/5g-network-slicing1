"""DQN hyperparameters and output location shared by the training, sweep and replay scripts."""
from src.config import RESULTS_DIR

FULL_EPISODES = 2000            # length of a full run; only these write to results/ directly

HP = {
    "hidden": 128,
    "lr": 1e-3,
    "gamma": 0.95,
    "batch_size": 128,
    "buffer_size": 100_000,
    "learn_start": 1_000,       # steps of experience before learning starts
    "learn_every": 4,           # one gradient step per 4 env steps
    "tau": 0.01,                # soft target update rate
    "double_dqn": False,        # True = Double-DQN target
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_frac": 0.6,      # epsilon reaches eps_end after 60% of the episodes
    "eval_every": 50,           # validation every N episodes
}


def output_dir(episodes):
    """Full runs write to results/; shorter runs get their own folder so they cannot replace them."""
    return RESULTS_DIR if episodes == FULL_EPISODES else RESULTS_DIR / f"quick_ep{episodes}"


# PPO (same environment, same number of training episodes as the DQN)
PPO_HP = {
    "hidden": 128,
    "lr": 3e-4,
    "gamma": 0.95,              # same discount as the DQN
    "gae_lambda": 0.95,
    "clip": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "reward_scale": 0.1,        # rewards are about -2 per step; scaled for stable value learning
    "episodes_per_update": 8,   # 8 days (768 steps) per policy update
    "epochs": 4,
    "minibatch": 128,
    "eval_every": 50,
}


# PPO hyperparameters (same network size, discount, episodes and validation protocol as the DQN)
PPO_HP = {
    "hidden": 128,
    "lr": 3e-4,
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "epochs": 6,
    "minibatch": 128,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad": 0.5,
    "reward_scale": 0.1,        # rewards are about -2 per step; scaled for stable value learning
    "episodes_per_update": 5,   # one update per 5 days of experience (480 steps)
    "eval_every": 50,
}
