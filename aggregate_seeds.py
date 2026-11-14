"""Aggregate multi-seed test results, separately for each mode.

isolated : results/test_results_15min_seed{N}.csv
priority : results/test_results_15min_seed{N}_prio.csv
"""
import glob
import pandas as pd

cols = ["PB_URLLC", "PB_eMBB", "PB_mMTC", "Wq_URLLC", "Wq_eMBB", "Wq_mMTC",
        "W_URLLC", "W_eMBB", "W_mMTC", "Lq_URLLC", "Lq_eMBB", "Lq_mMTC",
        "throughput", "util", "moves_per_day", "return_per_day"]

modes = {
    "isolated": sorted(f for f in glob.glob("results/test_results_15min_seed*.csv")
                       if not f.endswith("_prio.csv")),
    "priority": sorted(glob.glob("results/test_results_15min_seed*_prio.csv")),
}

for mode, files in modes.items():
    print(f"\n===== {mode.upper()}: {len(files)} seed files =====")
    if not files:
        continue
    df = pd.concat([pd.read_csv(f) for f in files])
    df["policy"] = df["policy"].str.replace(r"DQN.*", "DQN", regex=True)
    summary = df.groupby("policy")[cols].agg(["mean", "std"]).round(4)
    print(summary.T)
    out = f"results/summary_mean_std_15min_{mode}.csv"
    summary.to_csv(out)
    print("Saved:", out)