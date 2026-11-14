"""All paths and ASSUMED parameters live here (list them in the thesis)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "srilanka_5g_synthetic_traffic_6months.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

SLICES = ["URLLC", "eMBB", "mMTC"]
CSV_COLS = ["urllc_arrival_rate", "embb_arrival_rate", "mmtc_arrival_rate"]

ASSUMED = {
    # ---- two separately allocated resources (proposal objective 3) ----
    # (1) BANDWIDTH: radio PRBs.   (2) SERVER CAPACITY: processing server instances (UPF/MEC-like).
    "total_prbs": 60,           # bandwidth pool (capacity-constrained scenario)
    "min_alloc": 5,             # minimum PRBs per slice (starvation protection)
    "move_size": 5,             # PRBs moved by one bandwidth action
    "total_servers": 30,        # server-capacity pool
    "min_servers": 3,           # minimum servers per slice
    "server_move_size": 2,      # servers moved by one server action
    # service time of one packet = compute time + radio transmission time:
    #   1/mu_eff = 1/mu_server + c_s/(b_s * rate_per_prb)      (c_s servers share b_s PRBs)
    # per-slice service profile (ASSUMED values): URLLC servers are the fastest (dedicated low-latency processing),
    # mMTC the slowest; URLLC gets a short buffer (late packets are dropped, not delayed), eMBB a long one.
    "mu_server": {"URLLC": 6.0, "eMBB": 4.0, "mMTC": 3.0},   # pkts/s one server can process (compute speed)
    "rate_per_prb": 2.0,        # pkts/s one PRB can carry (radio speed)
    "buffer_per_slice": {"URLLC": 10, "eMBB": 30, "mMTC": 20},   # K_s = c_s + buffer_s
    # ---- priority-based sharing of idle servers (URLLC > eMBB > mMTC), see environment._metrics_priority ----
    "priority_sharing": False,  # True = KPIs from the priority-sharing surrogate (calibrated to SimPy)
    "priority_eta": 1.0,        # fraction of idle servers that a needy slice can really use (fitted to SimPy on VALIDATION days)
    "priority_target_util": 0.8,# a slice "wants" extra servers until its load a/c falls to this value (fitted, validation days)
    "decision_minutes": 15,     # interval used for the reported results (5 = finer but slower)
    "slice_weights": {"URLLC": 3.0, "eMBB": 1.0, "mMTC": 0.5},
    "w_pb": 10.0,               # reward weight on blocking probability
    "w_wq": 1.0,
    "w_waste": 1.0,             # reward weight on idle capacity: waste = 1 - carried load / deployed capacity (0 = off)                # reward weight on normalised waiting time
    "wq_cap": 5.0,              # seconds; Wq clipped here before normalising
    "move_cost": 0.3,          # penalty for reconfiguring
    "train_days": (1, 120),
    "val_days": (121, 150),
    "test_days": (151, 180),
}
