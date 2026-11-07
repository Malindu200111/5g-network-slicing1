"""
SimPy verification of the analytical queueing model.  Run from the project root:
    python verify_simpy.py            # ~3-6 min
    python verify_simpy.py --quick    # ~1 min, noisier
Outputs: results/simpy_validation.csv, results/simpy_validation.png, results/simpy_priority.csv

Part A  analytical M/M/c/K  vs  SimPy M/M/c/K   (PB, Wq, W, L) over a load sweep
Part B  analytical aggregate vs SimPy 3-slice NON-PREEMPTIVE PRIORITY queue
        (aggregate PB / L must match M/M/c/K; Wq must be lower for URLLC, higher for mMTC)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import queuing_model as qm
from src.config import ASSUMED, RESULTS_DIR, SLICES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    sim_time, warmup = (2000, 200) if args.quick else (6000, 500)
    n_seeds = 1 if args.quick else args.seeds
    RESULTS_DIR.mkdir(exist_ok=True)

    mu, buf = ASSUMED["mu"], ASSUMED["buffer_per_slice"]
    # (label, c) - typical PRB counts of the three slices in the environment
    configs = [("URLLC-like c=12", 12), ("eMBB-like c=37", 37), ("mMTC-like c=11", 11)]
    rhos = [0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.3]        # offered load rho = lam / (c * mu)

    # ------------------------------------------------------------------ Part A
    rows = []
    for label, c in configs:
        K = c + buf
        for rho in rhos:
            lam = rho * c * mu
            an = qm.mmck_metrics(lam, mu, c, K)
            sims = [qm.simulate_mmck(lam, mu, c, K, sim_time=sim_time, warmup=warmup, seed=s + 1)
                    for s in range(n_seeds)]
            sim = {k: float(np.mean([x[k] for x in sims])) for k in ("PB", "Wq", "W", "L")}
            rows.append({"config": label, "c": c, "K": K, "rho": rho, "lam": lam,
                         "PB_an": an["PB"], "PB_sim": sim["PB"],
                         "Wq_an": an["Wq"], "Wq_sim": sim["Wq"],
                         "W_an": an["W"], "W_sim": sim["W"],
                         "L_an": an["L"], "L_sim": sim["L"]})
            print(f"{label:<16} rho={rho:.1f}  PB an/sim {an['PB']:.4f}/{sim['PB']:.4f} | "
                  f"Wq an/sim {an['Wq']:.3f}/{sim['Wq']:.3f} | L an/sim {an['L']:.2f}/{sim['L']:.2f}")
    df = pd.DataFrame(rows)
    df["PB_abs_err"] = (df.PB_an - df.PB_sim).abs()
    df["Wq_abs_err"] = (df.Wq_an - df.Wq_sim).abs()
    df["W_rel_err_%"] = 100 * (df.W_an - df.W_sim).abs() / df.W_an
    df["L_rel_err_%"] = 100 * (df.L_an - df.L_sim).abs() / df.L_an
    df.to_csv(RESULTS_DIR / "simpy_validation.csv", index=False)
    print("\nPart A summary (all configs / loads):")
    print(f"  max |PB error|  = {df.PB_abs_err.max():.4f}   mean = {df.PB_abs_err.mean():.4f}")
    print(f"  max |Wq error|  = {df.Wq_abs_err.max():.3f} s  mean = {df.Wq_abs_err.mean():.3f} s")
    print(f"  mean W error    = {df['W_rel_err_%'].mean():.2f} %   mean L error = {df['L_rel_err_%'].mean():.2f} %")

    # analytical curves + SimPy markers (deliverable 2: queue behaviour curves vs load)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    fine = np.linspace(0.3, 1.4, 80)
    for label, c in configs:
        K = c + buf
        an = [qm.mmck_metrics(r * c * mu, mu, c, K) for r in fine]
        d = df[df.config == label]
        for a, key, ylab in zip(ax, ("PB", "Wq", "L"), ("Blocking probability $P_B$",
                                                        "Mean waiting time $W_q$ (s)",
                                                        "Mean number in system $L$")):
            line, = a.plot(fine, [x[key] for x in an], label=f"{label} (analytical)")
            a.plot(d.rho, d[f"{key}_sim"], "o", color=line.get_color(), mfc="none", label=f"{label} (SimPy)")
            a.set_xlabel(r"Offered load $\rho=\lambda/(c\mu)$"); a.set_ylabel(ylab)
            a.grid(True, linestyle="--", alpha=0.6)
    ax[0].legend(fontsize=7)
    fig.suptitle("M/M/c/K: analytical model vs SimPy simulation")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "simpy_validation.png", dpi=200)

    # ------------------------------------------------------------------ Part B
    c_tot, mu = 60, mu
    K_tot = c_tot + 3 * buf
    scenarios = {"normal load": {"URLLC": 6, "eMBB": 30, "mMTC": 6},
                 "peak load": {"URLLC": 16, "eMBB": 38, "mMTC": 6},
                 "burst (overload)": {"URLLC": 20, "eMBB": 50, "mMTC": 14}}
    prow = []
    print("\nPart B: shared pool c=%d, K=%d, non-preemptive priority URLLC > eMBB > mMTC" % (c_tot, K_tot))
    for sc, lams in scenarios.items():
        tot = sum(lams.values())
        an = qm.mmck_metrics(tot, mu, c_tot, K_tot)
        sims = [qm.simulate_priority(lams, mu, c_tot, K_tot, sim_time=sim_time, warmup=warmup, seed=s + 1)
                for s in range(n_seeds)]
        agg_pb = np.mean([x["aggregate"]["PB"] for x in sims])
        agg_L = np.mean([x["aggregate"]["L"] for x in sims])
        print(f"\n[{sc}] total lambda={tot} | aggregate PB an/sim {an['PB']:.4f}/{agg_pb:.4f} | "
              f"L an/sim {an['L']:.2f}/{agg_L:.2f} | FIFO Wq (analytical) {an['Wq']:.3f}")
        for s in SLICES:
            wq = np.mean([x[s]["Wq"] for x in sims]); pb = np.mean([x[s]["PB"] for x in sims])
            print(f"   {s:<6} lambda={lams[s]:>3}  PB_sim {pb:.4f}  Wq_sim {wq:.3f}")
            prow.append({"scenario": sc, "slice": s, "lam": lams[s], "PB_sim": pb, "Wq_sim": wq,
                         "agg_PB_an": an["PB"], "agg_PB_sim": agg_pb, "FIFO_Wq_an": an["Wq"]})
    pd.DataFrame(prow).to_csv(RESULTS_DIR / "simpy_priority.csv", index=False)
    print(f"\n>> saved: {RESULTS_DIR / 'simpy_validation.csv'}, simpy_validation.png, simpy_priority.csv")
    plt.show()


if __name__ == "__main__":
    main()
