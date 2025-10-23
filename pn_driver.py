# pn_driver.py
"""
The asynchronous engine of friction for the cognitive architecture.
WHY: This component runs completely independently, providing a continuous source
of computational pressure (Prediction Error) that prevents the system from
becoming passive and forces it into dynamic, adaptive states.
"""
import threading
import time
import math
import random
from dataclasses import dataclass

from shared_resources import global_workspace
import config

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
        self.current_t = 14.1347 # Start at the first non-trivial zero
        self.steps_since_last_zero = 0
        self.is_running = True

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
            if random.random() < 0.05:
                self.steps_since_last_zero = 0

            p_n = self._calculate_pn()
            priority = int((1 - p_n) * 100) # Lower number = higher priority for PriorityQueue

            pn_signal = PredictionErrorSignal(
                timestamp=time.time(),
                source="PNDriver_RiemannZeta",
                priority=priority,
                p_n=p_n,
                uncompressed_data={'t': float(self.current_t), 'steps_since_zero': self.steps_since_last_zero}
            )

            if global_workspace.qsize() < 100:
                global_workspace.put((pn_signal.priority, pn_signal))
            
            time.sleep(0.1)
