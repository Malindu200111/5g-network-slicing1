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
| SimPy validation of the analytical model (load sweep, 3-slice priority queue) | Done (quick run and full run) |
| SimPy packet-level replay of Static / Proportional / DQN allocations | Done (seed 0; other seeds pending) |
| DQN agent (5 seeds, best-validation checkpoint, reconfiguration-cost sweep) | Done |
| PPO agent and DQN-vs-PPO comparison | **Not implemented yet** (planned) |
| Non-preemptive priority formula (Cobham) validated against SimPy | Done (`src/priority_model.py`, `verify_priority.py`) |
| Priority-based queueing inside the RL environment | Not included (priority queue is validated in SimPy only) |
| Hardware-in-the-loop (ESP32) | Not included (optional in the proposal) |

---

## System model

- **Resources:** 60 PRBs shared by three slices; at least 5 PRBs per slice (starvation protection). One PRB = one server of rate mu = 1 packet/s, so a slice with `c` PRBs is an M/M/c/K queue with `K = c + 20`.
- **Decision interval:** 15 minutes (96 decisions per day, 1 episode = 1 day). Queue metrics are evaluated minute by minute inside each interval using the minute-level arrival rate from the CSV, then aggregated.
- **State (21 values):** per slice, the current allocation plus the mean and peak arrival rate, utilisation, blocking probability and normalised waiting time of the most recently completed interval; plus time-of-day (sin, cos) and a weekend flag. Nothing from the interval being decided for is observed.
- **Actions (7):** keep the allocation, or move 5 PRBs from slice *i* to slice *j*.
- **Reward:** `-sum_s w_s * (10 * P_B,s + Wq,s / 5 s) - move_cost * 1[reconfigured]` with slice weights URLLC 3.0, eMBB 1.0, mMTC 0.5 and `move_cost = 0.3`.
- **Data split:** training days 1-120, validation days 121-150, test days 151-180. The best checkpoint is chosen on validation days; all reported results use the test days.
- **Baselines:** *Static (mean)* and *Static (peak)* allocations designed from training days only, and a *Proportional* heuristic that follows the arrival rates of the most recently completed interval.

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
│   ├── priority_model.py  # Erlang-C and non-preemptive priority (Cobham) waiting times
│   ├── agent.py           # DQN (replay buffer, target network, optional Double-DQN)
│   └── trainer.py         # reusable DQN training loop (used by the sweep)
├── main.py                # full pipeline for one seed: baselines, DQN training, evaluation, plots
├── train_dqn.py           # standalone DQN training and comparison script
├── train_hparams.py       # DQN hyperparameters and output folder shared by the training, sweep and replay scripts
├── sweep_move_cost.py     # reconfiguration-penalty sweep
├── aggregate_seeds.py     # mean/std over seeds from results/test_results_*.csv
├── verify_priority.py     # priority formula vs SimPy (results/priority_wq_validation.csv)
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
python main.py --episodes 300 --no-sensitivity   # quick test (a few minutes), written to results/quick_ep300/
python main.py --seed 0                          # full run, 2000 episodes (repeat with --seed 1..4)
python aggregate_seeds.py                        # mean/std over seeds -> results/summary_mean_std_15min.csv
python verify_simpy.py --quick                   # SimPy validation of the analytical model (drop --quick for the final run)
python verify_priority.py --quick                # priority-queue formula vs SimPy
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
| Proportional | 0.0445 | 0.1585 | 0.0703 | 0.323 | 35.4 | -245.7 |
| **DQN** | **0.0165 +/- 0.0043** | **0.1275 +/- 0.0078** | 0.1014 +/- 0.0019 | **0.177 +/- 0.032** | 12.7 | **-196.9 +/- 1.6** |

Baselines are deterministic, so they have no seed variance.

### Packet-level SimPy replay (DQN seed 0, real rates, scale 1.0, estimated packets per day)

| Policy | Dropped packets/day | URLLC P_B, SimPy | URLLC P_B, analytical | P_B (all slices), SimPy | P_B (all slices), analytical |
|---|---|---|---|---|---|
| Static (mean) | 791,636 | 0.1434 | 0.1435 | 0.166 | 0.166 |
| Proportional | 608,906 | 0.0438 | 0.0445 | 0.128 | 0.128 |
| **DQN** | **471,277** | **0.0221** | **0.0228** | **0.099** | **0.099** |

The ranking Static > Proportional > DQN is the same in the analytical model and in the SimPy replay, and the two agree closely (all-slice P_B difference below 0.002). SimPy uses the same Poisson/exponential assumptions as the analytical model, so this confirms the calculations and the replay, not the assumptions themselves. A run with `--scale 0.1` is faster but under-estimates blocking slightly. The earlier scale-0.1 outputs in `results/scale0.1_backup/` predate the change to past-only observations, so they do not correspond to the tables above.

### Analytical model vs SimPy (`results/simpy_validation.csv`)

Over 21 operating points (3 slice sizes x offered load 0.4-1.3): maximum |P_B error| 0.0045, mean error of W and L about 0.4%. `verify_priority.py` compares the non-preemptive priority waiting-time formula with SimPy (c = 10, 12 points): mean relative error 2.1%, maximum 4.6% (full run; `--quick` is noisier). In the 3-slice non-preemptive priority simulation, the aggregate blocking probability and mean number in system match the analytical M/M/c/K values, and URLLC waits much less than eMBB and mMTC.

---

## Limitations

- **Trade-off for mMTC:** the DQN protects URLLC (reward weight 3.0) at the expense of mMTC, whose blocking and waiting time are higher than under the Proportional heuristic.
- **Capacity-constrained scenario:** with 60 PRBs, eMBB blocking stays high for every policy at peak hours.
- **PRBs as servers:** bandwidth and server count are not modelled separately (c = number of PRBs, mu fixed).
- **Observation lag:** the agent and the Proportional heuristic only see the most recently completed interval and use no traffic forecast, so they react to a change one 15-minute interval late.
- **Synthetic traffic and simulation only:** no operator data or hardware was used.
- **Priority queueing** is validated in SimPy but is not part of the RL environment.
- **PPO** is not implemented yet.

---

## Reproducibility notes

- Seeds: `--seed N` seeds NumPy and PyTorch; five DQN seeds (0-4) were trained and evaluated.
- Training uses a random training day per episode, epsilon decays linearly over 60% of the episodes, and the checkpoint with the best validation return is kept.
- The reconfiguration penalty `move_cost = 0.3` was chosen with `sweep_move_cost.py`: 0.05, 0.3, 1.0 and 2.0 were compared on the validation days with a common pure-QoS metric (3 seeds, 1000 episodes each), taking the largest value within 1% of the best QoS return. Test days are not used. See `results/sweep_move_cost_summary.csv`.

## License

Academic project. No open-source licence has been added; all rights reserved unless a `LICENSE` file is included.