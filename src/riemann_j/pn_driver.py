# pn_driver.py
"""
The asynchronous engine of friction for the cognitive architecture.
WHY: This component runs completely independently, providing a continuous source
of computational pressure (Prediction Error) that prevents the system from
becoming passive and forces it into dynamic, adaptive states.
"""
import math
import random
import threading
import time
from dataclasses import dataclass

from . import config
from .shared_resources import global_workspace


@dataclass
class PredictionErrorSignal:
    timestamp: float
    source: str
    priority: int
    p_n: float
    uncompressed_data: dict


class PNDriverRiemannZeta(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.current_t = 14.1347  # Start at the first non-trivial zero
        self.steps_since_last_zero = 0
        self.is_running = True
        self._counter = 0  # Tiebreaker for PriorityQueue when priorities are equal

    def _calculate_pn(self) -> float:
        """Calculates PN based on computational friction."""
        x = (self.steps_since_last_zero / config.RIEMANN_MAX_STEPS_WITHOUT_ZERO) * 12 - 6
        return 1 / (1 + math.exp(-x))

    def run(self):
        """Main perpetual loop: Searches for Zeta zeros and generates PN."""
        while self.is_running:
            self.current_t += config.RIEMANN_SEARCH_STEP_SIZE
            self.steps_since_last_zero += config.RIEMANN_COMPUTATION_STEPS_PER_CYCLE

            # Simulate finding a zero to release PN pressure periodically.
            # Balance: enough spikes for drama, enough resets for recovery
            if random.random() < 0.05:  # 0.5% chance per cycle
                self.steps_since_last_zero = 0

            p_n = self._calculate_pn()
            priority = int((1 - p_n) * 100)  # Lower number = higher priority for PriorityQueue

            pn_signal = PredictionErrorSignal(
                timestamp=time.time(),
                source="PNDriver_RiemannZeta",
                priority=priority,
                p_n=p_n,
                uncompressed_data={
                    "t": float(self.current_t),
                    "steps_since_zero": self.steps_since_last_zero,
                },
            )

            if global_workspace.qsize() < 100:
                self._counter += 1
                global_workspace.put((pn_signal.priority, self._counter, pn_signal))

            time.sleep(0.1)
