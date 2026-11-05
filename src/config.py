import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "srilanka_5g_synthetic_traffic_6months.csv")

# RL Environment Settings
STATE_DIM = 15      # Slices 3 * Metrics 5 (Arrival λ, Lq, Latency W, Loss PB, PRB)
ACTION_DIM = 5     # Discrete PRB allocation choices
TOTAL_PRBS = 100   # කුළුණේ සම්පූර්ණ ධාරිතාව
MAX_STEPS = 1440   # දිනක මිනිත්තු ගණන (Episode length)

# Hyperparameters
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_CAPACITY = 20000