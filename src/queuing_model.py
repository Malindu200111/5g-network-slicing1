import math
import numpy as np

class SliceQueueModel:
    def __init__(self, mu: float, K: int, priority: int = 1):
        self.mu = float(mu)
        self.K = int(K)
        self.priority = priority

    def calculate_metrics(self, lambda_rate: float, prb_allocated: int):
        c = max(1, int(prb_allocated))
        lam = max(1e-4, float(lambda_rate))
        mu = self.mu
        K = max(c, self.K)

        rho = lam / (c * mu)
        r = lam / mu

        sum_terms = sum([(r ** n) / math.factorial(n) for n in range(c)])
        c_fact = math.factorial(c)
        if abs(rho - 1.0) < 1e-5:
            tail = ((r ** c) / c_fact) * (K - c + 1)
        else:
            tail = ((r ** c) / c_fact) * ((1.0 - (rho ** (K - c + 1))) / (1.0 - rho))
        p0 = 1.0 / (sum_terms + tail)

        pb = ((r ** K) / (c_fact * (c ** (K - c)))) * p0
        pb = float(np.clip(pb, 0.0, 1.0))
        lambda_eff = lam * (1.0 - pb)

        if abs(rho - 1.0) < 1e-5:
            l_q = ((r ** c) * p0 / c_fact) * ((K - c) * (K - c + 1) / 2.0)
        else:
            t1 = (r ** c) * rho * p0 / (c_fact * ((1.0 - rho) ** 2))
            t2 = 1.0 - (rho ** (K - c + 1)) - ((1.0 - rho) * (K - c + 1) * (rho ** (K - c)))
            l_q = max(0.0, t1 * t2)

        waiting_sec = l_q / lambda_eff if lambda_eff > 0 else 0.0
        total_delay_ms = (waiting_sec + (1.0 / mu)) * 1000.0

        return {
            "queue_length": float(l_q),
            "delay_ms": float(total_delay_ms),
            "packet_loss": float(pb),
            "utilization": float(min(1.0, rho))
        }