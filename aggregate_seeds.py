import glob
import pandas as pd

files = sorted(glob.glob("results/test_results_15min_seed*.csv"))
print(f"{len(files)} seed files found:", files)

df = pd.concat([pd.read_csv(f) for f in files])
df["policy"] = df["policy"].str.replace(r"DQN.*", "DQN", regex=True)

cols = ["PB_URLLC", "PB_eMBB", "PB_mMTC", "Wq_URLLC", "Wq_eMBB", "Wq_mMTC",
        "throughput", "util", "moves_per_day", "return_per_day"]
summary = df.groupby("policy")[cols].agg(["mean", "std"]).round(4)

print(summary.T)
summary.to_csv("results/summary_mean_std_15min.csv")
print("\nSaved: results/summary_mean_std_15min.csv")