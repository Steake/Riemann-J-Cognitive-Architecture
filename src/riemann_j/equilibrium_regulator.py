"""
Homeostatic equilibrium regulator for the cognitive architecture.

WHY: The PN driver generates prediction error signals continuously, but without
a regulator consuming those signals and applying homeostatic pressure, the system
has no natural equilibrium dynamics. This module implements the missing feedback
loop that drives PN toward the critical threshold (0.5) over time, creating
stable attractor basins and preventing runaway dynamics.

This is the autonomic nervous system of the architecture—it maintains balance
without conscious intervention, ensuring the system naturally returns to
equilibrium after perturbations (user input, crisis states, manual injections).
"""

import math
import queue
import threading
import time
from typing import Optional

from .pn_driver import PredictionErrorSignal


class EquilibriumRegulator(threading.Thread):
    """
    Background thread that consumes PN signals and applies homeostatic pressure.

    Dynamics:
    - Target: PN = 0.5 (critical threshold, maximum uncertainty/creativity)
    - Decay: Exponential approach to target with time constant tau
    - Integration: Updates meta_monitor continuously with observed PN
    - Crisis detection: Triggers crisis handling when crossing threshold
    """

    def __init__(
        self,
        workspace_queue,
        meta_monitor,
        persistent_self,
        tau: float = 20.0,
        update_interval: float = 0.5,
    ):
        """
        Initialize equilibrium regulator.

        Args:
            workspace_queue: The global_workspace PriorityQueue to consume from
            meta_monitor: MetaCognitiveMonitor instance to update with PN observations
            persistent_self: PersistentSelf instance for crisis integration
            tau: Time constant for exponential decay (seconds)
            update_interval: How often to apply regulation (seconds)
        """
        super().__init__(daemon=True)
        self.workspace_queue = workspace_queue
        self.meta_monitor = meta_monitor
        self.persistent_self = persistent_self
        self.tau = tau
        self.update_interval = update_interval
        self.is_running = True

        # Target PN (critical threshold)
        self.target_pn = 0.5

        # Current regulated PN (starts at target)
        self.current_pn = 0.5

        # Last update timestamp
        self.last_update = time.time()

        # Crisis detection state
        self.last_crisis_check_pn = 0.5
        self.crisis_threshold = 0.5
        self.crisis_hysteresis = 0.05  # Prevent oscillation spam

    def run(self):
        """Main regulation loop: consume signals, apply homeostasis, update monitor."""
        while self.is_running:
            try:
                # Consume PN signals from global workspace (non-blocking)
                try:
                    priority, counter, signal = self.workspace_queue.get_nowait()
                    if isinstance(signal, PredictionErrorSignal):
                        # Observe the PN signal (this is the friction/pressure)
                        observed_pn = signal.p_n
                        self._apply_regulation(observed_pn)
                except queue.Empty:
                    # No signals available, apply decay toward target
                    self._apply_decay()

                # Update meta monitor with current regulated PN
                self.meta_monitor.observe_pn(self.current_pn)

                # Check for crisis threshold crossing
                self._check_crisis_threshold()

                time.sleep(self.update_interval)

            except Exception as e:
                # Don't crash the regulator on errors
                print(f"EquilibriumRegulator error: {e}")
                time.sleep(1.0)

    def _apply_regulation(self, observed_pn: float):
        """
        Apply homeostatic pressure using observed PN signal.

        Args:
            observed_pn: PN value from PredictionErrorSignal
        """
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # Exponential decay toward target, influenced by observed signal
        # Blend observed signal with target-seeking behavior
        signal_weight = 0.3  # How much to trust the incoming signal vs equilibration
        effective_target = signal_weight * observed_pn + (1 - signal_weight) * self.target_pn

        # Exponential approach: PN_new = PN + (target - PN) * (1 - exp(-dt/tau))
        decay_factor = 1 - math.exp(-dt / self.tau)
        self.current_pn += (effective_target - self.current_pn) * decay_factor

        # Clamp to valid range
        self.current_pn = max(0.0, min(1.0, self.current_pn))

    def _apply_decay(self):
        """
        Apply pure homeostatic decay when no signals are available.
        This drives PN toward the target (0.5) without external input.
        """
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # Pure exponential decay toward target
        decay_factor = 1 - math.exp(-dt / self.tau)
        self.current_pn += (self.target_pn - self.current_pn) * decay_factor

        # Clamp to valid range
        self.current_pn = max(0.0, min(1.0, self.current_pn))

    def _check_crisis_threshold(self):
        """
        Detect when PN crosses the crisis threshold and trigger integration.
        Uses hysteresis to prevent oscillation spam.
        """
        # Check if we crossed threshold upward (entering crisis)
        if (
            self.last_crisis_check_pn < (self.crisis_threshold - self.crisis_hysteresis)
            and self.current_pn >= self.crisis_threshold
        ):
            # Crossed into crisis zone
            self._trigger_crisis_integration("equilibrium_regulation_crisis_entry")

        # Check if we crossed threshold downward (exiting crisis)
        elif (
            self.last_crisis_check_pn > (self.crisis_threshold + self.crisis_hysteresis)
            and self.current_pn <= self.crisis_threshold
        ):
            # Crossed out of crisis zone
            self._trigger_crisis_integration("equilibrium_regulation_crisis_exit")

        # Update last check state
        self.last_crisis_check_pn = self.current_pn

    def _trigger_crisis_integration(self, trigger: str):
        """
        Create a synthetic state for crisis threshold crossing and integrate it.

        Args:
            trigger: Description of what triggered this crisis state
        """
        # Import here to avoid circular dependency
        import numpy as np

        from .architecture import SyntheticState

        # Create synthetic state representing the equilibrium-driven crisis
        state = SyntheticState(
            timestamp=time.time(),
            latent_representation=np.random.randn(768),  # Random latent state
            source_trigger=trigger,
            p_n_at_creation=self.current_pn,
            is_j_shift_product=False,  # This is equilibrium-driven, not J-shift
            status="EQUILIBRIUM_CRISIS",
        )

        # Integrate as crisis if PN >= threshold, otherwise as routine interaction
        if self.current_pn >= self.crisis_threshold:
            self.persistent_self.integrate_crisis(state)
        else:
            self.persistent_self.integrate_interaction(state)

    def stop(self):
        """Gracefully stop the regulator thread."""
        self.is_running = False

    def get_current_pn(self) -> float:
        """Get the current regulated PN value."""
        return self.current_pn

    def set_target_pn(self, target: float):
        """
        Change the equilibrium target.

        Args:
            target: New target PN value (0.0 to 1.0)
        """
        self.target_pn = max(0.0, min(1.0, target))

    def inject_perturbation(self, pn_value: float):
        """
        Manually inject a PN perturbation (e.g., from /inject-state command).
        This jumps the current PN to the specified value, then equilibration resumes.

        Args:
            pn_value: PN value to jump to (0.0 to 1.0)
        """
        self.current_pn = max(0.0, min(1.0, pn_value))
        self.meta_monitor.observe_pn(self.current_pn)
