"""
D2: analytical load curves of the queuing model (P_B, Wq, W, utilisation against offered load).
Run from the project root:   python load_curves.py          (about 2-3 min: the exact priority CTMC is solved per load point)
Writes results/load_curves.png and results/load_curves.csv

Panel row 1: single M/M/c/K queue (per-slice model used by the environment), c=6 and c=18.
Panel row 2: shared pool c=30, K=90 with non-preemptive priority URLLC > eMBB > mMTC (exact finite-buffer CTMC),
             traffic mix of the peak scenario (16 : 38 : 6), compared with the FIFO pool (aggregate M/M/c/K).
SimPy points from results/simpy_priority.csv (verify_simpy.py) are overlaid when that file exists.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import queuing_model as qm
from src.priority_model import priority_ctmc
from src.config import RESULTS_DIR, SLICES

MU = 2.0                       # per-server rate of the equal split (as in verify_simpy.py)
rows = []

fig, ax = plt.subplots(2, 4, figsize=(20, 9))
rho = np.linspace(0.2, 1.5, 66)

# ---------- Row 1: single queues ----------
for c, K, col in [(6, 26, "tab:blue"), (18, 38, "tab:orange")]:
    m = [qm.mmck_metrics(r * c * MU, MU, c, K) for r in rho]
    PB = np.array([x["PB"] for x in m]); Wq = np.array([x["Wq"] for x in m])
    W = Wq + 1 / MU
    util = np.array([x["lam_eff"] for x in m]) / (c * MU)
    for a, y in zip(ax[0], [PB, Wq, W, util]):
        a.plot(rho, y, color=col, label=f"c={c}, K={K}")
    for r, p, w, wt, u in zip(rho, PB, Wq, W, util):
        rows.append({"model": f"M/M/{c}/{K}", "rho": r, "PB": p, "Wq": w, "W": wt, "util": u})
for a, (t, yl) in zip(ax[0], [("Blocking probability $P_B$", "$P_B$"), ("Waiting time $W_q$", "seconds"),
                              ("Latency $W=W_q+1/\\mu$", "seconds"), ("Utilisation (carried / capacity)", "")]):
    a.set_title(t); a.set_ylabel(yl); a.set_xlabel(r"Offered load $\rho=\lambda/(c\mu)$")
    a.axvline(1.0, color="grey", linestyle=":"); a.grid(True, linestyle="--", alpha=0.5); a.legend()
ax[0, 1].set_yscale("log"); ax[0, 1].set_ylim(1e-4, 5)
ax[0, 0].set_yscale("log"); ax[0, 0].set_ylim(1e-6, 1)

# ---------- Row 2: priority pool ----------
c_tot, K_tot = 30, 90
mix = np.array([16.0, 38.0, 6.0]); mix = mix / mix.sum()
rho_p = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
res = {s: {"PB": [], "Wq": []} for s in SLICES}; fifo_wq, fifo_pb = [], []
for r in rho_p:
    tot = r * c_tot * MU
    lams = list(tot * mix)
    pm = priority_ctmc(lams, MU, c_tot, K_tot)
    an = qm.mmck_metrics(tot, MU, c_tot, K_tot)
    fifo_wq.append(an["Wq"]); fifo_pb.append(an["PB"])
    for i, s in enumerate(SLICES):
        res[s]["PB"].append(pm["PB"] if np.isscalar(pm["PB"]) else pm["PB"][i])
        res[s]["Wq"].append(pm["Wq"][i])
        rows.append({"model": "priority pool c=30 K=90", "rho": r, "slice": s, "lam": lams[i],
                     "PB": res[s]["PB"][-1], "Wq": res[s]["Wq"][-1], "FIFO_Wq": an["Wq"]})
    print(f"rho={r:.1f} done", flush=True)
colors = {"URLLC": "tab:blue", "eMBB": "tab:orange", "mMTC": "tab:green"}
for s in SLICES:
    ax[1, 0].plot(rho_p, res[s]["Wq"], "o-", color=colors[s], label=s)
ax[1, 0].plot(rho_p, fifo_wq, "k--", label="FIFO pool (any class)")
ax[1, 0].set_yscale("log"); ax[1, 0].set_title("Priority pool: waiting time per class")
ax[1, 0].set_ylabel("$W_q$ (seconds)")
for s in SLICES:
    ax[1, 1].plot(rho_p, [p + 1e-12 for p in res[s]["PB"]], "o-", color=colors[s], label=s)
ax[1, 1].plot(rho_p, fifo_pb, "k--", label="FIFO pool")
ax[1, 1].set_yscale("log"); ax[1, 1].set_ylim(1e-6, 1); ax[1, 1].set_title("Priority pool: blocking (shared buffer)")
ax[1, 1].set_ylabel("$P_B$")
for s in SLICES:
    ax[1, 2].plot(rho_p, np.array(res[s]["Wq"]) + 1 / MU, "o-", color=colors[s], label=s)
ax[1, 2].set_title("Priority pool: latency $W$ per class"); ax[1, 2].set_ylabel("seconds"); ax[1, 2].set_yscale("log")
ratio = np.array(res["mMTC"]["Wq"]) / np.maximum(np.array(res["URLLC"]["Wq"]), 1e-12)
ax[1, 3].plot(rho_p, ratio, "o-", color="tab:red")
ax[1, 3].set_title("$W_q$(mMTC) / $W_q$(URLLC)"); ax[1, 3].set_ylabel("ratio"); ax[1, 3].set_yscale("log")
for a in ax[1]:
    a.set_xlabel(r"Offered load $\rho=\lambda_{tot}/(c\mu)$"); a.axvline(1.0, color="grey", linestyle=":")
    a.grid(True, linestyle="--", alpha=0.5)
for a in ax[1, :3]:
    a.legend(fontsize=8)

# SimPy overlay
f = RESULTS_DIR / "simpy_priority.csv"
if f.exists():
    sp = pd.read_csv(f)
    rho_of = {"peak load": 1.0}      # only the scenario whose class mix (16:38:6) is the one plotted
    for sc, rr in rho_of.items():
        for s in SLICES:
            r = sp[(sp.scenario == sc) & (sp.slice == s)]
            if len(r):
                ax[1, 0].plot(rr, max(float(r.Wq_sim.iloc[0]), 1e-4), "x", color=colors[s], markersize=9, markeredgewidth=2)
    ax[1, 0].plot([], [], "kx", label="SimPy (peak load)")
    ax[1, 0].legend(fontsize=8)
fig.suptitle("Queuing model load curves (D2): single M/M/c/K queues (top) and the priority pool (bottom)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "load_curves.png", dpi=200)
pd.DataFrame(rows).to_csv(RESULTS_DIR / "load_curves.csv", index=False)
print(">> saved results/load_curves.png and load_curves.csv")
