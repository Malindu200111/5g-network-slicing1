"""All paths and ASSUMED parameters live here (list them in the thesis)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "srilanka_5g_synthetic_traffic_6months.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

SLICES = ["URLLC", "eMBB", "mMTC"]
CSV_COLS = ["urllc_arrival_rate", "embb_arrival_rate", "mmtc_arrival_rate"]

ASSUMED = {
    "total_prbs": 60,           # capacity-constrained scenario (100 PRBs left the network ~50% idle)
    "mu": 1.0,                  # pkts/s per PRB
    "buffer_per_slice": 20,     # K_s = c_s + buffer
    "decision_minutes": 15,    # 5 = finer (final results), 15 = faster (development)
    "move_size": 5,             # PRBs moved by one action
    "min_alloc": 5,             # minimum PRBs per slice (starvation protection)
    "slice_weights": {"URLLC": 3.0, "eMBB": 1.0, "mMTC": 0.5},
    "w_pb": 10.0,               # reward weight on blocking probability
    "w_wq": 1.0,                # reward weight on normalised waiting time
    "wq_cap": 5.0,              # seconds; Wq clipped here before normalising
    "move_cost": 0.05,          # penalty for reconfiguring
    "train_days": (1, 120),
    "val_days": (121, 150),
    "test_days": (151, 180),
}
