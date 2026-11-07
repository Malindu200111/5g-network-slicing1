# DRL-Based Dynamic Resource Allocation and Queuing Optimization in 5G Network Slicing

A simulation framework in which a **Deep Q-Network (DQN)** agent dynamically re-allocates Physical Resource Blocks (PRBs) among three 5G slices (**URLLC, eMBB, mMTC**). Each slice is modelled as a finite-capacity **M/M/c/K** queue; blocking probability and waiting time computed from the queueing model drive the reward. The learned policy is compared with static and proportional allocation, and the results are validated with a packet-level **SimPy** discrete-event simulation.

**Institution:** Sri Lanka Technology Campus (SLTC), Faculty of Engineering
**Programme:** BScHons (Eng) Electronics and Telecommunication, Group 09, 2026
**Authors:** K.K.M.K.B. Kossinna (23UG1-0015), A.K.H.T.N. Amarasekara (23UG1-0032)
**Supervisor:** Mr. Nuwan Atthanayake | **Co-supervisors:** Ms. Dilanka De Silva, Ms. Naduni Herath

---

## Project status

| Component | Status |
|---|---|
| Multi-slice environment driven by a 180-day traffic CSV | Done |
| Analytical M/M/c/K model (P_B, Wq, W, L, utilisation; Little's law) | Done |
| SimPy validation of the analytical model (load sweep, 3-slice priority queue) | Done (final full run pending) |
| SimPy packet-level replay of Static / Proportional / DQN allocations | Done (seed 0; other seeds pending) |
| DQN agent (5 seeds, best-validation checkpoint, reconfiguration-cost sweep) | Done |
| PPO agent and DQN-vs-PPO comparison | **Not implemented yet** (planned) |
| Priority-based queueing inside the RL environment | Not included (priority queue is validated in SimPy only) |
| Hardware-in-the-loop (ESP32) | Not included (optional in the proposal) |

---

## System model

- **Resources:** 60 PRBs shared by three slices; at least 5 PRBs per slice (starvation protection). One PRB = one server of rate mu = 1 packet/s, so a slice with `c` PRBs is an M/M/c/K queue with `K = c + 20`.
- **Decision interval:** 15 minutes (96 decisions per day, 1 episode = 1 day). Queue metrics are evaluated minute by minute inside each interval using the minute-level arrival rate from the CSV, then aggregated.
- **State (21 values):** per slice, interval-mean and interval-peak arrival rate, current allocation, utilisation, blocking probability, normalised waiting time; plus time-of-day (sin, cos) and a weekend flag.
- **Actions (7):** keep the allocation, or move 5 PRBs from slice *i* to slice *j*.
- **Reward:** `-sum_s w_s * (10 * P_B,s + Wq,s / 5 s) - move_cost * 1[reconfigured]` with slice weights URLLC 3.0, eMBB 1.0, mMTC 0.5 and `move_cost = 0.3`.
- **Data split:** training days 1-120, validation days 121-150, test days 151-180. The best checkpoint is chosen on validation days; all reported results use the test days.
- **Baselines:** *Static (mean)* and *Static (peak)* allocations designed from training days only, and a *Proportional* heuristic that follows the current arrival rates.

All assumed parameters are listed in `src/config.py`.

---

## Repository structure

```text
.
├── data/processed/srilanka_5g_synthetic_traffic_6months.csv   # 180 days x 1440 min, 3 slices
├── src/
│   ├── config.py          # paths and all assumed parameters
│   ├── environment.py     # slicing environment, baselines, evaluation
│   ├── queuing_model.py   # analytical M/M/c/K + SimPy simulators (single slice, 3-slice priority)
│   ├── agent.py           # DQN (replay buffer, target network, optional Double-DQN)
│   └── trainer.py         # reusable DQN training loop (used by the sweep)
├── main.py                # full pipeline for one seed: baselines, DQN training, evaluation, plots
├── train_dqn.py           # standalone DQN training and comparison script
├── train_hparams.py       # DQN hyperparameters shared by train_dqn.py and sweep_move_cost.py
├── sweep_move_cost.py     # reconfiguration-penalty sweep
├── aggregate_seeds.py     # mean/std over seeds from results/test_results_*.csv
├── verify_simpy.py        # analytical M/M/c/K vs SimPy, priority-queue check
├── simpy_replay.py        # export allocations and replay them packet by packet in SimPy
├── requirements.txt
└── results/               # CSVs, figures and DQN checkpoints produced by the scripts
```

The traffic dataset is **synthetic**. Columns: `time_minute, day_number, day_of_week, is_weekend, month_index, hour_of_day, embb_arrival_rate, urllc_arrival_rate, mmtc_arrival_rate` (259,200 rows).

---

## Quick start

```powershell
git clone https://github.com/Malindu200111/5g-network-slicing1.git
cd 5g-network-slicing1

python -m venv venv
.\venv\Scripts\Activate.ps1          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

```powershell
python main.py --episodes 300 --no-sensitivity   # quick test (a few minutes)
python main.py --seed 0                          # full run, 2000 episodes (repeat with --seed 1..4)
python aggregate_seeds.py                        # mean/std over seeds -> results/summary_mean_std_15min.csv
python verify_simpy.py --quick                   # SimPy validation of the analytical model (drop --quick for the final run)
python simpy_replay.py                           # SimPy replay of Static/Proportional/DQN on test days 151-180
```

`simpy_replay.py` needs `results/dqn_best_15min_seed0.pt` (produced by `main.py --seed 0`); use `--seed N` for another model. It writes one output dataset per policy, for example `results/simpy_output_Static_15min_seed0.csv` (without DQN) and `results/simpy_output_DQN_15min_seed0.csv` (with DQN), with allocated PRBs, simulated arrivals and drops, and analytical vs simulated P_B and Wq for every day, 15-minute step and slice.

---

## Results (test days 151-180)

### DQN vs baselines, analytical evaluation (5 training seeds, mean +/- std)

| Policy | P_B URLLC | P_B eMBB | P_B mMTC | Wq URLLC (s) | Reconfig./day | Return/day |
|---|---|---|---|---|---|---|
| Static (mean) | 0.1435 | 0.1731 | 0.1573 | 0.656 | 0 | -409.2 |
| Static (peak) | 0.1435 | 0.2613 | 0.0684 | 0.656 | 0 | -458.6 |
| Proportional | 0.0415 | 0.1574 | 0.0562 | 0.312 | 35.5 | -237.2 |
| **DQN** | **0.0134 +/- 0.0012** | **0.1308 +/- 0.0022** | 0.0965 +/- 0.0037 | **0.164 +/- 0.016** | 15.0 | **-193.8 +/- 0.9** |

Baselines are deterministic, so they have no seed variance.

### Packet-level SimPy replay (DQN seed 0, rates scaled by 0.1, estimated real packets per day)

| Policy | Dropped packets/day | URLLC drops/day | P_B (all slices), SimPy | P_B (all slices), analytical |
|---|---|---|---|---|
| Static (mean) | 783,285 | 108,319 | 0.164 | 0.166 |
| Proportional | 578,111 | 27,608 | 0.121 | 0.125 |
| **DQN** | **484,859** | **6,225** | **0.102** | **0.106** |

The ranking Static > Proportional > DQN is the same in the analytical model and in the SimPy replay. SimPy gives slightly lower blocking than the analytical model in every case (difference at most about 0.02, largest for mMTC); likely causes are the rate scaling, empty queues at the start of each day, and the minute-wise steady-state assumption of the analytical model.

### Analytical model vs SimPy (`verify_simpy.py --quick`)

Over 21 operating points (3 slice sizes x offered load 0.4-1.3): maximum |P_B error| 0.013, mean error of W and L about 1.4%. In the 3-slice non-preemptive priority simulation, the aggregate blocking probability and mean number in system match the analytical M/M/c/K values, and URLLC waits much less than eMBB and mMTC.

---

## Limitations

- **Trade-off for mMTC:** the DQN protects URLLC (reward weight 3.0) at the expense of mMTC, whose blocking and waiting time are higher than under the Proportional heuristic.
- **Capacity-constrained scenario:** with 60 PRBs, eMBB blocking stays high for every policy at peak hours.
- **PRBs as servers:** bandwidth and server count are not modelled separately (c = number of PRBs, mu fixed).
- **Observation:** the agent sees the arrival rate of the interval it is deciding for, as the Proportional heuristic does. A deployable system would need a forecast.
- **Synthetic traffic and simulation only:** no operator data or hardware was used.
- **Priority queueing** is validated in SimPy but is not part of the RL environment.
- **PPO** is not implemented yet.

---

## Reproducibility notes

- Seeds: `--seed N` seeds NumPy and PyTorch; five DQN seeds (0-4) were trained and evaluated.
- Training uses a random training day per episode, epsilon decays linearly over 60% of the episodes, and the checkpoint with the best validation return is kept.
- The reconfiguration penalty `move_cost = 0.3` was chosen with `sweep_move_cost.py` (0.05, 0.3 and 1.0 compared on the test days with a common pure-QoS metric); see `results/sweep_move_cost_summary.csv`.

## License

Academic project. No open-source licence has been added; all rights reserved unless a `LICENSE` file is included.