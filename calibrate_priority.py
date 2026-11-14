"""
Check / fit the priority-sharing surrogate (environment._metrics_priority) against the packet-level SimPy
priority replay (simpy_priority_replay.PoolSim). Uses VALIDATION days only (121-150) so that the test days stay clean.
    python calibrate_priority.py --days 121-150 --every 6 --scale 1.0     # simulate (slow) and save results/priority_sim_ref.csv
    python calibrate_priority.py --fit                                    # grid-search eta / target_util on the saved reference
"""
import argparse, itertools, multiprocessing as mp, time
import numpy as np, pandas as pd
from simpy_priority_replay import job
from simpy_replay import parse_days
from src.config import ASSUMED, DATA_PATH, RESULTS_DIR, SLICES
from src.environment import SlicingEnv, buffers, load_data, service_rate

ap = argparse.ArgumentParser()
ap.add_argument("--days", default="121-150"); ap.add_argument("--every", type=int, default=6)
ap.add_argument("--policies", default="Static,Proportional,DQN"); ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--fit", action="store_true"); ap.add_argument("--check", action="store_true",
                help="no fitting: error of the CURRENT config values on the saved reference (use on TEST days)")
ap.add_argument("--ref", default="priority_sim_ref.csv", help="reference file name inside results/"); ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--suffix", default="")
args = ap.parse_args()
cfg = dict(ASSUMED); tag = "15min_seed0" + args.suffix
adf = pd.read_csv(RESULTS_DIR / f"allocations_{tag}.csv")
data = load_data(DATA_PATH, 15)
REF = RESULTS_DIR / args.ref

if not (args.fit or args.check):
    days = parse_days(args.days, data["mean"].shape[0])[:: args.every]
    jobs = []
    for p in [x for x in args.policies.split(",") if x in set(adf.policy)]:
        for d in days:
            sd = adf[(adf.policy == p) & (adf.day == d + 1)].sort_values("step")
            prb, srv = sd[[f"alloc_{s}" for s in SLICES]].to_numpy(), sd[[f"srv_{s}" for s in SLICES]].to_numpy()
            jobs.append((p, d + 1, "priority", srv, service_rate(prb, srv, cfg), data["fine"][d].reshape(1440, 3),
                         buffers(cfg), args.scale, args.seed, 900.0))
    t0 = time.time(); print(len(jobs), "SimPy runs")
    with mp.Pool(args.workers) as pool:
        res = list(pool.imap_unordered(job, jobs))
    rows = [{"policy": r["policy"], "day": r["day"], "slice": n, "arr": r["arr"][s], "drop": r["drop"][s],
             "served": r["served"][s], "wait": r["wait"][s]} for r in res for s, n in enumerate(SLICES)]
    pd.DataFrame(rows).to_csv(REF, index=False); print(f"saved {REF} in {(time.time()-t0)/60:.1f} min")
else:
    ref = pd.read_csv(REF)
    g = ref.groupby(["policy", "slice"])[["arr", "drop", "served", "wait"]].sum()
    sim_pb = (g["drop"] / g["arr"]).to_dict(); sim_wq = (g["wait"] / g["served"] * args.scale).to_dict()
    pairs = ref[["policy", "day"]].drop_duplicates().values.tolist()

    def surrogate(eta, tu):
        c = dict(cfg, priority_sharing=True, priority_eta=eta, priority_target_util=tu)
        env = SlicingEnv(data, [0], c)
        num = {}
        for p, d in pairs:
            sd = adf[(adf.policy == p) & (adf.day == d)].sort_values("step"); env.day = d - 1
            for t, (_, row) in enumerate(sd.iterrows()):
                al = np.array([[row[f"alloc_{s}"] for s in SLICES], [row[f"srv_{s}"] for s in SLICES]], dtype=int)
                m = env._metrics(al, t, d - 1)
                for i, s in enumerate(SLICES):
                    lam = env.lam[d - 1, t, i]; k = (p, s)
                    a = num.setdefault(k, [0.0, 0.0, 0.0, 0.0])
                    a[0] += lam; a[1] += lam * m[i]["PB"]; a[2] += m[i]["lam_eff"]; a[3] += m[i]["lam_eff"] * m[i]["Wq"]
        return {k: (v[1] / v[0], v[3] / max(v[2], 1e-12)) for k, v in num.items()}

    best = None
    grid = ([(cfg["priority_eta"], cfg["priority_target_util"])] if args.check
            else itertools.product([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.6, 0.7, 0.8, 0.9]))
    for eta, tu in grid:
        sur = surrogate(eta, tu)
        err_pb = np.mean([abs(sur[k][0] - sim_pb[k]) for k in sim_pb])
        err_wq = np.mean([abs(sur[k][1] - sim_wq[k]) for k in sim_pb])
        score = err_pb / 0.02 + err_wq / 0.1
        print(f"eta {eta:.1f} tu {tu:.1f} | mean|dPB| {err_pb:.4f} mean|dWq| {err_wq:.3f} | score {score:.2f}")
        if best is None or score < best[0]:
            best = (score, eta, tu, sur)
    _, eta, tu, sur = best
    print(f"\nBEST eta={eta} target_util={tu}\n{'policy':<13}{'slice':<7}{'PB sim':>9}{'PB surr':>9}{'Wq sim':>9}{'Wq surr':>9}")
    for k in sorted(sim_pb):
        print(f"{k[0]:<13}{k[1]:<7}{sim_pb[k]:>9.4f}{sur[k][0]:>9.4f}{sim_wq[k]:>9.3f}{sur[k][1]:>9.3f}")
