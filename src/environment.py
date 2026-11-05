import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.config import DATASET_PATH, STATE_DIM, ACTION_DIM, MAX_STEPS
from src.queuing_model import SliceQueueModel

class FiveGNetworkSlicingEnv(gym.Env):
    def __init__(self):
        super(FiveGNetworkSlicingEnv, self).__init__()
        self.df = pd.read_csv(DATASET_PATH)
        self.current_step = 0

        self.action_space = spaces.Discrete(ACTION_DIM)
        self.action_mapping = {
            0: [70, 20, 10],  # High eMBB
            1: [50, 40, 10],  # High URLLC
            2: [50, 20, 30],  # High mMTC
            3: [34, 33, 33],  # Equal Share
            4: [60, 30, 10],  # Balanced
        }

        self.observation_space = spaces.Box(
            low=0.0, high=np.finfo(np.float32).max, shape=(STATE_DIM,), dtype=np.float32
        )

        self.embb_queue = SliceQueueModel(mu=1.8, K=100, priority=2)
        self.urllc_queue = SliceQueueModel(mu=2.5, K=50, priority=1)
        self.mmtc_queue = SliceQueueModel(mu=1.2, K=150, priority=3)

        self.current_prbs = [60, 30, 10]

    def _get_obs(self, embb_m, urllc_m, mmtc_m, row):
        return np.array([
            row['embb_arrival_rate'], embb_m["queue_length"], embb_m["delay_ms"], embb_m["packet_loss"], self.current_prbs[0],
            row['urllc_arrival_rate'], urllc_m["queue_length"], urllc_m["delay_ms"], urllc_m["packet_loss"], self.current_prbs[1],
            row['mmtc_arrival_rate'], mmtc_m["queue_length"], mmtc_m["delay_ms"], mmtc_m["packet_loss"], self.current_prbs[2]
        ], dtype=np.float32)

    def step(self, action):
        self.current_prbs = self.action_mapping[action]
        row = self.df.iloc[self.current_step]

        embb_m = self.embb_queue.calculate_metrics(row['embb_arrival_rate'], self.current_prbs[0])
        urllc_m = self.urllc_queue.calculate_metrics(row['urllc_arrival_rate'], self.current_prbs[1])
        mmtc_m = self.mmtc_queue.calculate_metrics(row['mmtc_arrival_rate'], self.current_prbs[2])

        state = self._get_obs(embb_m, urllc_m, mmtc_m, row)

        # Reward Function (URLLC Latency target <= 5ms)
        reward = 15.0 if urllc_m["delay_ms"] <= 5.0 else -50.0
        total_loss = embb_m["packet_loss"] + urllc_m["packet_loss"] + mmtc_m["packet_loss"]
        reward -= (total_loss * 30.0)

        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1) or (self.current_step >= MAX_STEPS)
        info = {"urllc_delay": urllc_m["delay_ms"], "total_loss": total_loss}

        return state, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_prbs = [60, 30, 10]
        row = self.df.iloc[self.current_step]
        embb_m = self.embb_queue.calculate_metrics(row['embb_arrival_rate'], self.current_prbs[0])
        urllc_m = self.urllc_queue.calculate_metrics(row['urllc_arrival_rate'], self.current_prbs[1])
        mmtc_m = self.mmtc_queue.calculate_metrics(row['mmtc_arrival_rate'], self.current_prbs[2])
        return self._get_obs(embb_m, urllc_m, mmtc_m, row), {}