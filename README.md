```markdown
# Dynamic 5G Network Slicing via DRL & Queuing Theory (HIL-Enabled)
> An advanced 5G Multi-Slice Dynamic Resource Allocation and Queuing Optimization framework bridging Analytical Queuing Models ($M/M/c/K$), Deep Reinforcement Learning (DQN / PPO), and Hardware-in-the-Loop (HIL) emulation.

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-008000.svg)](https://gymnasium.farama.org/)
[![SimPy](https://img.shields.io/badge/SimPy-Simulation-FF8C00.svg)](https://simpy.readthedocs.io/)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32%20HIL-red.svg)](https://www.espressif.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎓 Academic Profile

* **Institution:** Sri Lanka Technology Campus (SLTC) | Faculty of Engineering
* **Degree Program:** BScHons (Eng) in Electronics and Telecommunication Engineering
* **Group:** Group 09 (Academic Year 2026)
* **Researchers:** 
  * K.K.M.K.B. Kossinna (`23UG1-0015`)
  * A.K.H.T.N. Amarasekara (`23UG1-0032`)
* **Supervisor:** Mr. Nuwan Atthanayake
* **Co-Supervisors:** Ms. Dilanka De Silva & Ms. Naduni Herath

---

## 📌 Project Overview

Traditional 5G radio resource allocation algorithms struggle with bursty, heterogeneous traffic without risking Service Level Agreement (SLA) violations or severe resource under-utilization. Furthermore, standard model-free Deep Reinforcement Learning (DRL) approaches treat network buffers as "black boxes," resulting in slow convergence, high exploration penalties, and unpredictable queue instabilities.

This project delivers a **hybrid, mathematically grounded framework** that unifies analytical queuing theory, DRL optimization, and physical hardware validation:

* **Analytical 5G Multi-Slice Modeling:** Simulates distinct slice profiles (**eMBB**, **URLLC**, and **mMTC**) using priority-based, finite-capacity $M/M/c/K$ queuing systems within a discrete-event simulation engine (SimPy).
* **DRL-Driven Optimization:** Trains PyTorch-based **DQN** and **PPO** agents to dynamically allocate Physical Resource Blocks (PRBs) and scale packet processing servers ($c$).
* **Mathematically Guided Rewards:** Incorporates **Little’s Law** ($L_q = \lambda_a W_q$) and packet blocking probability ($P_B$) directly into the reward engine to penalize latency spikes and buffer overflows before drops occur.
* **Hardware-in-the-Loop (HIL) Emulation:** Connects physical **ESP32 microcontrollers** via UDP/IP socket streaming to inject realistic, jittered IoT/mMTC telemetry into the virtual slicing environment in real time.

---

## 🛠️ Tech Stack & Tools

* **Programming Language:** Python 3.10+
* **Simulation Framework:** SimPy (Discrete-Event Network Simulation), Gymnasium
* **Reinforcement Learning:** PyTorch, NumPy (DQN with Target Networks & Replay Buffers, PPO)
* **Queuing Theory:** Priority-Based $M/M/c/K$ Analytical Formulations, Little's Law ($L_q = \lambda_a W_q$)
* **Hardware & Networking:** ESP32 Microcontrollers, Wi-Fi 802.11 b/g/n, UDP/IP Socket Programming
* **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, Seaborn

---

## 📐 System Architecture

```text
  +-------------------------------------------------------------------------+
  |                        TRAFFIC INGESTION ENGINE                         |
  |  - Synthetic Diurnal Traffic (TRCSL-calibrated eMBB & URLLC)            |
  |  - Physical ESP32 Microcontroller (Real-time UDP/IP IoT/mMTC Telemetry) |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                   PRIORITY-BASED M/M/c/K QUEUING CORE                   |
  |  - URLLC (Highest Priority, strict delay constraint <= 5ms)             |
  |  - eMBB (Bandwidth & Throughput intensive, dynamic server scaling)      |
  |  - mMTC (High connection density, finite buffer capacity K)             |
  |  - Real-time Metrics: Effective Arrival (\lambda_a), Queue Delay (W_q)  |
  +------------------------------------+------------------------------------+
                                       | State Telemetry [λ, L_q, W_q, PRBs]
                                       v
  +-------------------------------------------------------------------------+
  |                      DRL DECISION ENGINE (DQN / PPO)                    |
  |  - Evaluates queue state & SLA bounds                                   |
  |  - Computes Mathematically Guided Rewards: R(t) = f(W_q, SLA, P_B)       |
  |  - Outputs Dynamic PRB Allocation & Server Scaling Actions (c)          |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                     CLOSED-LOOP RESOURCE ACTUATION                      |
  |  - Dynamically reallocates spectrum & server capacity across slices     |
  +-------------------------------------------------------------------------+

```

---

## 📁 Repository Structure

```text
5g-network-slicing/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── srilanka_5g_synthetic_traffic_6months.csv  # 180-day baseline dataset
│
├── hardware/
│   └── esp32_traffic_generator.ino                    # ESP32 Wi-Fi UDP packet generator
│
├── src/
│   ├── __init__.py
│   ├── config.py                                      # Simulation & SLA parameters
│   ├── queuing_model.py                               # Analytical M/M/c/K math & Little's Law
│   ├── environment.py                                 # Gymnasium simulation environment
│   ├── agent.py                                       # PyTorch DQN & PPO neural networks
│   └── hil_receiver.py                                # UDP socket bridge for ESP32 packets
│
├── outputs/
│   ├── figures/                                       # Convergence & latency comparative graphs
│   └── dqn_5g_model.pth                               # Serialized model checkpoint
│
├── requirements.txt
├── README.md
└── main.py                                            # Master pipeline entry point

```

---

## 🚀 Quickstart

### 1. Clone & Set Up Virtual Environment

```powershell
git clone [https://github.com/Malindu200111/5g-network-slicing1.git](https://github.com/Malindu200111/5g-network-slicing1.git)
cd 5g-network-slicing1

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt

```

### 3. Run Training Pipeline

```powershell
python main.py

```

---

## 📊 Evaluation & Benchmark Targets

The framework benchmarks adaptive DRL control against conventional static slicing across five key metrics:

1. **URLLC Latency Violation Rate:** Targeted reduction to $\le 0.2\%$ within the $\le 5\text{ ms}$ SLA envelope.
2. **Buffer Packet Loss ($P_B$):** Mitigated drop probability during bursts via proactive server capacity ($c$) scaling.
3. **Queue Waiting Time ($W_q$):** Mathematically bounded mean delay across all slices.
4. **Bandwidth Savings:** Up to $25\% - 30\%$ PRB utilization efficiency gain over peak-provisioned static baselines.

```

<FollowUp label="Want me to write hardware/esp32_traffic_generator.ino and src/hil_receiver.py for the ESP32 socket bridge?" query="Write the ESP32 Arduino code (hardware/esp32_traffic_generator.ino) and the Python UDP receiver script (src/hil_receiver.py) for the Hardware-in-the-Loop setup."/>

```
