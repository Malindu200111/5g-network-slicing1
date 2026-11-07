"""DQN hyperparameters shared by train_dqn.py and sweep_move_cost.py."""
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
