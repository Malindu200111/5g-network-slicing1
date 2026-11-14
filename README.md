# DRL-Based Dynamic Resource Allocation and Queuing Optimization in 5G Network Slicing

A simulation framework in which a **Deep Q-Network (DQN)** agent dynamically re-allocates two resources, **Physical Resource Blocks (PRBs) and server capacity**, among three 5G slices (**URLLC, eMBB, mMTC**). Each slice is modelled as a finite-capacity **M/M/c/K** queue, optionally with a shared server pool and non-preemptive priority URLLC > eMBB > mMTC; blocking probability and waiting time computed from the queueing model drive the reward. The learned policy is compared with static and proportional allocation, and the results are validated with a packet-level **SimPy** discrete-event simulation.

**Institution:** Sri Lanka Technology Campus (SLTC), Faculty of Engineering
**Programme:** BScHons (Eng) Electronics and Telecommunication, Group 09, 2026
**Authors:** K.K.M.K.B. Kossinna (23UG1-0015), A.K.H.T.N. Amarasekara (23UG1-0032)
**Supervisor:** Mr. Nuwan Atthanayake | **Co-supervisors:** Ms. Dilanka De Silva, Ms. Naduni Herath

---

## Project status

| Component | Status |
|---|---|
| Multi-slice environment driven by a 180-day traffic CSV (PRBs + servers) | Done |
| Analytical M/M/c/K model (P_B, Wq, W, L, utilisation; Little's law) and load curves (`load_curves.py`) | Done |
| Exact finite-buffer non-preemptive priority model (`priority_ctmc`) and Cobham formula | Done (`src/priority_model.py`) |
| Parameterised traffic generator (lambda, mu, c, K) with queue-length series | Done (`src/traffic_generator.py`) |
| SimPy validation of the queuing models (load sweep, 3-slice priority pool) | Done (`verify_simpy.py --seeds 10`, `verify_priority.py`) |
| SimPy packet-level replay of Static / Proportional / DQN allocations | Done for seed 0, both modes; other seeds pending |
| DQN agent (5 seeds, isolated and priority-sharing modes, best-validation checkpoint) | Done |
| Priority-based queueing inside the RL environment | Done as a fitted mean-field surrogate (`--priority`), checked against SimPy |
| PPO agent | Code present (`train_ppo.py`), **not run and not reported**; future work |
| Hardware-in-the-loop (ESP32) | Not included (optional in the proposal) |

---

## System model

- **Resources (allocated separately):** (1) **bandwidth** = 60 PRBs (min 5 per slice, moved in steps of 5); (2) **server capacity** = 30 processing servers (min 3 per slice, moved in steps of 2). A slice with `b` PRBs and `c` servers is an M/M/c/K queue with `K = c + 20` whose per-server service rate follows `1/mu_eff = 1/mu_server + c/(b * rate_per_prb)` (compute time + radio transmission time; `mu_server` per slice URLLC 6 / eMBB 4 / mMTC 3 pkts/s, `rate_per_prb` = 2 pkts/s, all assumed). Buffers are also per slice (URLLC 10, eMBB 30, mMTC 20 waiting places; assumed). The reward has an extra idle-capacity term `w_waste * (1 - carried load / deployed capacity)`. Both resources therefore change the queue: more servers add parallelism, more PRBs make each packet faster. 
- **Decision interval:** 15 minutes (96 decisions per day, 1 episode = 1 day). Queue metrics are evaluated minute by minute inside each interval using the minute-level arrival rate from the CSV, then aggregated.
- **State (27 values):** per slice, the current PRB and server allocation, the queue length (Lq / buffer size) plus the mean and peak arrival rate, utilisation, blocking probability and normalised waiting time of the most recently completed interval; plus time-of-day (sin, cos) and a weekend flag. Nothing from the interval being decided for is observed.
- **Actions (13):** keep; move 5 PRBs from slice *i* to slice *j* (6 actions); or move 2 servers from slice *i* to slice *j* (6 actions).
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
│   ├── priority_model.py  # Erlang-C, Cobham priority waiting times, exact finite-buffer priority CTMC
│   ├── traffic_generator.py # parameterised (lambda, mu, c, K) traffic generator
│   ├── agent.py           # DQN (replay buffer, target network, optional Double-DQN)
│   └── trainer.py         # reusable DQN training loop (used by the sweep)
├── main.py                # full pipeline for one seed: baselines, DQN training, evaluation, plots
├── train_dqn.py           # standalone DQN training and comparison script
├── train_hparams.py       # DQN hyperparameters and output folder shared by the training, sweep and replay scripts
├── sweep_move_cost.py     # reconfiguration-penalty sweep
├── aggregate_seeds.py     # mean/std over seeds from results/test_results_*.csv
├── verify_priority.py     # Cobham priority formula vs SimPy (results/priority_wq_validation.csv)
├── load_curves.py         # analytical load curves (P_B, Wq, W, utilisation; priority pool) -> results/load_curves.png
├── traffic_demo.py        # traffic generator demo and check against the priority CTMC
├── plot_comparison.py     # bar charts DQN vs baselines [--priority]
├── burst_test.py          # performance on the 5 test days with the highest URLLC peak [--priority]
├── calibrate_priority.py  # fit/check of the priority surrogate against SimPy
├── simpy_priority_replay.py # packet-level replay in isolated / pool_fifo / priority modes
├── train_ppo.py           # PPO (code only; not part of the reported results)
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
python main.py --priority --seed 0               # same with priority sharing in the environment (files get _prio)
python aggregate_seeds.py                        # mean/std over seeds, isolated and priority separately
python plot_comparison.py [--priority]           # comparison bar charts
python burst_test.py [--priority]                # peak-day test
python verify_simpy.py --seeds 10                # SimPy validation of the queuing models (--quick for a fast check)
python verify_priority.py                        # Cobham priority formula vs SimPy
python load_curves.py                            # analytical load curves
python simpy_replay.py                           # SimPy replay of Static/Proportional/DQN on test days 151-180
python simpy_priority_replay.py [--suffix _prio] # replay in isolated / pool_fifo / priority modes
```

`simpy_replay.py` needs `results/dqn_best_15min_seed0.pt` (produced by `main.py --seed 0`); use `--seed N` for another model. It writes one output dataset per policy, for example `results/simpy_output_Static_15min_seed0.csv` (without DQN) and `results/simpy_output_DQN_15min_seed0.csv` (with DQN), with allocated PRBs, simulated arrivals and drops, and analytical vs simulated P_B and Wq for every day, 15-minute step and slice.

---

## PPO and priority sharing (added in v5)
Note: PPO code is included but no PPO results are reported; the reported agent is the DQN.
- `python train_ppo.py --seed 0` trains a PPO agent (clipped objective, GAE, categorical policy) in the SAME environment, same 13 actions, same 2000 episodes and validation protocol as the DQN; it writes `results/test_results_ppo_<tag>.csv`. Compare it with the DQN row of `test_results_<tag>.csv`.
- `python simpy_replay.py --stage export --policies Static,Proportional,DQN,PPO` exports allocations (PPO needs `ppo_best_*.pt`).
- `python simpy_priority_replay.py` replays the agents' allocations in three packet-level modes with identical arrivals: `isolated` (each slice only its own servers, as in the reward), `pool_fifo` (one shared pool, first-come-first-served) and `priority` (guaranteed shares + non-preemptive priority URLLC > eMBB > mMTC for idle servers). The default `main.py` trains with isolated slices; `main.py --priority` trains with priority sharing inside the environment.

## Results (test days 151-180)

All numbers below come from the dual-resource model (60 PRBs + 30 servers, all rates assumed). DQN rows are the mean +/- std over five training seeds (0-4). Baselines are deterministic, so they have no seed variance. Wq and W are in seconds of simulated time. Source files: `results/summary_mean_std_15min_isolated.csv` and `..._priority.csv` (`python aggregate_seeds.py`).

### Isolated slices (each slice only uses its own servers)

| Policy | P_B URLLC | P_B eMBB | P_B mMTC | Wq URLLC | Wq eMBB | Wq mMTC | Throughput | Moves/day | Return/day |
|---|---|---|---|---|---|---|---|---|---|
| Static (mean) | 0.0840 | 0.1787 | 0.1790 | 0.199 | 0.450 | 0.516 | 46.17 | 0 | -377.1 |
| Static (peak) | 0.0840 | 0.2688 | 0.0920 | 0.199 | 0.629 | 0.266 | 43.36 | 0 | -431.1 |
| Proportional | 0.0244 | 0.1679 | **0.1039** | 0.088 | 0.488 | **0.399** | 47.69 | 43.1 | -279.8 |
| **DQN** | **0.0172 +/- 0.0007** | **0.1220 +/- 0.0009** | 0.1262 +/- 0.0074 | **0.076 +/- 0.003** | **0.426 +/- 0.009** | 0.813 +/- 0.032 | **49.36 +/- 0.08** | 22.9 +/- 2.1 | **-247.3 +/- 1.1** |

DQN is 11.6% better than the best baseline (Proportional) in return. The price is mMTC, the lowest-weight slice: its loss and waiting time are worse than under the Proportional rule.

### Shared server pool with priority (`main.py --priority`)

| Policy | P_B URLLC | P_B eMBB | P_B mMTC | Wq URLLC | Wq eMBB | Wq mMTC | Throughput | Moves/day | Return/day |
|---|---|---|---|---|---|---|---|---|---|
| Static (mean) | 0.0241 | 0.0863 | 0.0496 | 0.061 | 0.245 | 0.186 | 51.27 | 0 | -174.7 |
| Static (peak) | 0.0150 | 0.0928 | 0.0427 | 0.047 | 0.234 | 0.110 | 51.16 | 0 | -158.1 |
| Proportional | 0.0132 | 0.0822 | 0.0586 | 0.063 | 0.245 | 0.255 | 51.46 | 43.1 | -174.0 |
| **DQN** | **0.0026 +/- 0.0004** | **0.0173 +/- 0.0012** | **0.0281 +/- 0.0023** | **0.029 +/- 0.004** | **0.046 +/- 0.003** | **0.081 +/- 0.006** | **54.29 +/- 0.06** | 16.2 +/- 0.8 | **-61.3 +/- 0.4** |

DQN is 61% better than the best baseline (Static peak). `util` is above 1 for DQN (1.019) because it is carried load divided by the guaranteed capacity; borrowed servers are not counted. The only metric where DQN is not best is mMTC latency W (0.84 s against 0.71-0.79 s for the static rules).

### Peak days (`burst_test.py`, seed 0)

The five test days with the highest URLLC arrival peak (days 156, 162, 164, 173, 179) cost the DQN very little: -246.7 per day on those days against -247.3 on the other days (isolated), and -63.6 against -61.5 (priority). These are the highest days of the data set, not an injected overload. In isolated mode the DQN's mMTC loss on those days (0.136) is above the Proportional rule's (0.104).

### Packet-level SimPy replay (seed 0, real rates, scale 1.0)

SimPy replay of the seed 0 allocations on the 30 test days (`simpy_replay.py`, `simpy_priority_replay.py`), with the same Poisson/exponential assumptions as the analytical model.

| Mode, policy | P_B URLLC | P_B eMBB | P_B mMTC | Wq URLLC | Wq eMBB | Wq mMTC |
|---|---|---|---|---|---|---|
| Isolated, Static (mean), SimPy / analytical | 0.0841 / 0.0840 | 0.1786 / 0.1787 | 0.1774 / 0.1790 | 0.199 / 0.199 | 0.449 / 0.450 | 0.510 / 0.516 |
| Isolated, Proportional | 0.0243 / 0.0244 | 0.1680 / 0.1679 | 0.1020 / 0.1039 | 0.087 / 0.088 | 0.489 / 0.488 | 0.395 / 0.399 |
| Isolated, DQN | 0.0161 / 0.0162 | 0.1229 / 0.1229 | 0.1299 / 0.1329 | 0.072 / 0.072 | 0.421 / 0.421 | 0.844 / 0.840 |
| Priority, Static (mean), SimPy / training model | 0.0239 / 0.0241 | 0.0788 / 0.0863 | 0.0461 / 0.0496 | 0.057 / 0.061 | 0.221 / 0.245 | 0.155 / 0.186 |
| Priority, Proportional | 0.0115 / 0.0132 | 0.0806 / 0.0822 | 0.0466 / 0.0586 | 0.041 / 0.063 | 0.236 / 0.245 | 0.201 / 0.255 |
| Priority, DQN (priority-trained) | 0.0006 / 0.0028 | 0.0216 / 0.0186 | 0.0299 / 0.0292 | 0.007 / 0.029 | 0.044 / 0.048 | 0.060 / 0.086 |

* Isolated mode: packet-level and analytical results agree (P_B difference at most 0.003, Wq at most 0.006 s).
* Priority mode: the training model is slightly pessimistic for waiting time; the ranking is the same, and DQN has the lowest loss and waiting time on every slice.
* The same priority-trained DQN gives P_B 0.006 / 0.018 / 0.025 in a shared pool without priority and 0.56 for eMBB with no sharing: most of the gain comes from pooling servers, while priority mainly protects URLLC (P_B 0.0006 and Wq 0.007 s against 0.006 and 0.051 s).
* The isolated-trained DQN loses its advantage once the servers are pooled (for example mMTC P_B 0.078 against 0.047 for Proportional under priority), which is why the priority agent is trained inside the sharing environment.
* One day (151, seed 0, isolated) replayed alone: estimated dropped packets per day Static 753,581, Proportional 619,202, DQN 491,093.
* Only seed 0 has been replayed at packet level. SimPy uses the same assumptions as the analytical model, so this confirms the calculations and the queue approximation, not the assumptions themselves.

### Queuing models vs SimPy (`results/simpy_validation.csv`, `results/simpy_priority.csv`, `--seeds 10`)

* Single M/M/c/K queues (offered load 0.4-1.3): maximum |P_B error| 0.0026, maximum |Wq error| 0.009 s, mean error of W and L about 0.24%. The three configurations in the sweep use only two distinct sizes (c=6, K=26 and c=18, K=38).
* Shared pool c=30, K=90, non-preemptive priority, exact CTMC against SimPy: normal and burst loads agree to within noise (burst P_B 0.2848 / 0.2857, mMTC Wq 5.477 / 5.481). At peak load aggregate P_B is 0.0139 (SimPy) against 0.0148 (formula) and mMTC Wq 3.444 against 3.545; the gap shrank from 9% to 6% and from 14% to 3% when going from 1 to 3 to 10 seeds, which points to simulation noise on a rare event.
* Cobham's closed-form priority waiting time against SimPy (`verify_priority.py`, c=10, mu=1, large buffer K=310, load 0.5-0.95, 12 points): mean relative error 2.1%, maximum 4.6% (mMTC at load 0.95, where its waiting time is 6.6 s and the simulation is noisiest).
* Load curves of the model: `python load_curves.py` writes `results/load_curves.png`.

---

## Limitations

- **Trade-off for mMTC (isolated mode):** the DQN protects URLLC (reward weight 3.0) and eMBB at the expense of mMTC, whose blocking and waiting time are higher than under the Proportional rule (Wq 0.81 s against 0.40 s).
- **The priority-trained agent needs sharing:** without server sharing its eMBB loss is 0.56. Use the isolated agent when slices cannot share servers.
- **Priority in the training environment is a surrogate:** `_metrics_priority` is a fitted mean-field model, checked against SimPy (`calibrate_priority.py`, packet-level replay). The exact priority CTMC is exact only for equal service rates, and it is too slow to use inside the training loop.
- **Assumed parameters:** `mu_server`, `rate_per_prb`, the buffer sizes, 60 PRBs and 30 servers are assumptions, not measurements. The traffic is synthetic, so nothing here uses operator data or hardware.
- **Packet-level replay covers seed 0 only.** SimPy uses the same Poisson/exponential assumptions as the analytical model.
- **No injected overload test:** `burst_test.py` uses the five highest-peak test days, which are close to ordinary days for the DQN.
- **No PPO results:** the code exists but was not run; the objective allows "DQN or PPO".
- **Observation lag:** the agent and the Proportional heuristic only see the most recently completed interval and use no traffic forecast, so they react to a change one 15-minute interval late.
- **No starvation guard:** nothing stops the agent from keeping a low-weight slice at its minimum allocation.

---

## Reproducibility notes

- Seeds: `--seed N` seeds NumPy and PyTorch; five DQN seeds (0-4) were trained and evaluated.
- Training uses a random training day per episode, epsilon decays linearly over 60% of the episodes, and the checkpoint with the best validation return is kept.
- The reconfiguration penalty `move_cost = 0.3` was chosen with `sweep_move_cost.py`: 0.05, 0.3, 1.0 and 2.0 were compared on the validation days with a common pure-QoS metric (3 seeds, 1000 episodes each), taking the largest value within 1% of the best QoS return. Test days are not used. See `results/sweep_move_cost_summary.csv`.

## License

Academic project. No open-source licence has been added; all rights reserved unless a `LICENSE` file is included.

## Priority sharing inside the training environment (validated surrogate)
`python main.py --priority` trains and evaluates the agent with priority sharing: every slice keeps its guaranteed servers, idle servers of other slices are lent in the order URLLC > eMBB > mMTC (non-preemptive). Exact formulas do not exist for this system, so `environment._metrics_priority` is a mean-field surrogate with two fitted parameters (`priority_eta`, `priority_target_util`).
- **Fit** (`calibrate_priority.py --fit`, validation days 121-150, packet-level SimPy as reference): mean |dP_B| 0.0026, mean |dWq| 0.020 s.
- **Held-out check** (`calibrate_priority.py --check`, test days, same parameters): mean |dP_B| 0.0038, mean |dWq| 0.020 s (5 sampled days, 3 policies; surrogate is slightly optimistic for URLLC/mMTC waiting time).
- **Independent confirmation of the trained agent** (`simpy_priority_replay.py --suffix _prio`, real rates, all 30 test days, seed 0): in the priority-sharing system the DQN has P_B 0.0006 / 0.022 / 0.030 (URLLC/eMBB/mMTC) against 0.012 / 0.081 / 0.047 for Proportional. The surrogate predicts 0.003 / 0.019 / 0.029.
- **Caveat:** an agent trained with priority sharing relies on sharing. Replayed in an ISOLATED system (no sharing) its eMBB blocking is very high (0.56). Use the isolated agent (`python main.py`) when slices cannot share servers.
- `util` can exceed 1 in this mode because it is carried load divided by the slice's GUARANTEED capacity (borrowed servers are not counted).
- Files: `python plot_comparison.py [--priority]`, `python burst_test.py [--priority]`, `python simpy_replay.py --stage export --suffix _prio`.

## Slice profiles and parameters (deliverable 1)
| | URLLC | eMBB | mMTC |
|---|---|---|---|
| Arrival rate lambda | CSV (workday 08-17 spikes) | CSV (diurnal peaks) | CSV (night bursts, heavy tail) |
| Server speed mu_server (pkts/s, assumed) | 6 | 4 | 3 |
| Buffer (waiting places, assumed) | 10 | 30 | 20 |
| K = c + buffer | c+10 | c+30 | c+20 |
| Reward weight | 3 | 1 | 0.5 |
Queue length: `Lq` and `L` are computed (Little's law), part of the state (Lq / buffer) and of every results table.
Finite-buffer priority formula: the exact `priority_ctmc` below; the older rescaled approximation `priority_metrics_mmck` is kept for reference only.

## Finite-buffer priority formula (Deliverable 2) - exact solution

`src/priority_model.py` now has two priority results:
* `priority_wq_nonpreemptive` - Cobham's closed form (infinite buffer), valid when blocking is negligible.
* `priority_ctmc(lams, mu, c, K)` - **exact** finite-buffer non-preemptive priority M/M/c/K (identical exponential `mu`,
  one shared buffer, classes URLLC > eMBB > mMTC). State = n packets if n < c, else the queue vector (q1,q2,q3); the
  balance equations are solved with a sparse solver. Output: `PB` (equal for every class, PASTA), `Wq_k = E[q_k]/(lam_k(1-PB))`, `Lq`, `W`.
  It agrees with SimPy (`python verify_simpy.py`, part B; also `python traffic_demo.py`) within simulation noise,
  and with Cobham's formula when blocking is negligible. Cost: about 10 s for c=30, K=90 (use it for validation, not inside the training loop).
  `priority_metrics_mmck` (older rescaled approximation) is kept only for reference.
* Limits: exact only for equal `mu` across classes. The environment's per-step priority model is still the fitted mean-field surrogate
  (`_metrics_priority`), which is checked against SimPy (`calibrate_priority.py`).

## Traffic generator (Deliverable 1)

`src/traffic_generator.py`: `SliceTraffic(lam, mu, c, K)` with `lam` = constant, callable `lam(t)` or per-minute array (e.g. from the dataset CSV);
`run_generator(slices, horizon, sample_every, priority=..., pool_servers=...)` produces Poisson (non-homogeneous, by thinning) traffic, runs the
queues (isolated, or one shared pool with priority) and returns PB, Wq, W, Lq and the **queue-length time series**. Demo/check: `python traffic_demo.py`.