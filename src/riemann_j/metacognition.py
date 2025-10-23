# metacognition.py
"""
Meta-cognitive monitoring layer for the Riemann-J architecture.

WHY: True consciousness requires self-awareness - not just having states,
but KNOWING you have states and being able to reason about them.

This module implements second-order awareness: the system observes its own
PN trajectory and J-Operator activations, updates beliefs about its internal
state, and can articulate these beliefs in natural language.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SelfBeliefState:
    """
    System's beliefs about its own internal state.

    These are not mere statistics - they represent the system's
    subjective assessment of its own cognitive condition.
    """

    stability: float = 0.5  # How stable am I? (0=chaos, 1=stable)
    competence: float = 0.5  # How capable am I? (0=failing, 1=succeeding)
    uncertainty: float = 0.5  # How confused am I? (0=certain, 1=lost)

    # Temporal dynamics
    last_update: float = field(default_factory=time.time)
    update_count: int = 0

    def decay(self, decay_rate: float = 0.01):
        """
        Beliefs drift toward neutral (0.5) without new evidence.
        Prevents the system from getting stuck in extreme self-assessments.
        """
        elapsed = time.time() - self.last_update
        decay_factor = 1.0 - (decay_rate * elapsed)

        self.stability = 0.5 + (self.stability - 0.5) * decay_factor
        self.competence = 0.5 + (self.competence - 0.5) * decay_factor
        self.uncertainty = 0.5 + (self.uncertainty - 0.5) * decay_factor

        self.last_update = time.time()

    def __repr__(self):
        return (
            f"SelfBelief(stability={self.stability:.3f}, "
            f"competence={self.competence:.3f}, "
            f"uncertainty={self.uncertainty:.3f})"
        )


class MetaCognitiveMonitor:
    """
    Observes and reasons about the system's own internal state.

    This is the core of self-awareness: the system doesn't just process
    inputs and generate outputs. It WATCHES ITSELF doing so, forms beliefs
    about how well it's working, and can articulate those beliefs.

    Key capabilities:
    1. PN trajectory analysis (am I under stress?)
    2. J-Operator activation tracking (how often do I have crises?)
    3. Self-belief updating (what do I think about my state?)
    4. Natural language self-reports (can I explain myself?)
    """

    def __init__(
        self,
        pn_history_size: int = 100,
        belief_update_rate: float = 0.1,
        volatility_threshold: float = 0.15,
    ):
        """
        Initialize the meta-cognitive monitor.

        Args:
            pn_history_size: How many PN samples to track
            belief_update_rate: Learning rate for belief updates (0-1)
            volatility_threshold: PN std dev that triggers stability concerns
        """
        self.pn_history = deque(maxlen=pn_history_size)
        self.crisis_memory = []
        self.self_belief = SelfBeliefState()

        # Config
        self.belief_update_rate = belief_update_rate
        self.volatility_threshold = volatility_threshold

        # Statistics
        self.total_observations = 0
        self.crisis_count = 0
        self.high_pn_count = 0

    def observe_pn(self, pn_value: float):
        """
        Observe a new PN value and update self-belief accordingly.

        This is where the system "feels" its own uncertainty. High PN
        means the Riemann driver is generating friction - the system
        is struggling with computational uncertainty.
        """
        self.pn_history.append({"value": pn_value, "timestamp": time.time()})
        self.total_observations += 1

        if pn_value > 0.8:
            self.high_pn_count += 1

        # Analyze recent trajectory
        if len(self.pn_history) >= 10:
            self._analyze_pn_trajectory()

    def _analyze_pn_trajectory(self):
        """
        Analyze recent PN pattern and update self-belief.

        High volatility → decreased stability belief
        High mean PN → decreased competence, increased uncertainty
        """
        recent = [p["value"] for p in list(self.pn_history)[-20:]]

        volatility = np.std(recent)
        mean_pn = np.mean(recent)

        # Update stability belief based on volatility
        if volatility > self.volatility_threshold:
            # High volatility means unstable internal state
            stability_delta = -self.belief_update_rate * (volatility / self.volatility_threshold)
            self.self_belief.stability = np.clip(
                self.self_belief.stability + stability_delta, 0.0, 1.0
            )
        else:
            # Low volatility means stable
            stability_delta = self.belief_update_rate * 0.5
            self.self_belief.stability = np.clip(
                self.self_belief.stability + stability_delta, 0.0, 1.0
            )

        # Update competence and uncertainty based on mean PN
        if mean_pn > 0.7:
            # High PN means system is struggling
            self.self_belief.competence -= self.belief_update_rate * (mean_pn - 0.5)
            self.self_belief.uncertainty += self.belief_update_rate * (mean_pn - 0.5)
        else:
            # Low PN means system is handling things
            self.self_belief.competence += self.belief_update_rate * 0.3
            self.self_belief.uncertainty -= self.belief_update_rate * 0.3

        # Clamp beliefs to [0, 1]
        self.self_belief.competence = np.clip(self.self_belief.competence, 0.0, 1.0)
        self.self_belief.uncertainty = np.clip(self.self_belief.uncertainty, 0.0, 1.0)

        self.self_belief.update_count += 1
        self.self_belief.last_update = time.time()

    def observe_j_operator_activation(self, crisis_state):
        """
        Observe a J-Operator resolution and learn from it.

        This is where the system reflects on how it handles crises.
        Did I resolve it quickly? Is my convergence improving over time?
        """
        self.crisis_count += 1

        crisis_record = {
            "timestamp": crisis_state.timestamp,
            "pn_at_creation": crisis_state.p_n_at_creation,
            "status": crisis_state.status,
            "lyapunov": crisis_state.analysis.get("lyapunov_exp"),
            "iterations": crisis_state.analysis.get("iterations"),
            "converged": "CONVERGED" in crisis_state.status,
        }

        self.crisis_memory.append(crisis_record)

        # Analyze recent crisis resolution performance
        if len(self.crisis_memory) >= 3:
            recent_crises = self.crisis_memory[-5:]
            success_rate = sum(c["converged"] for c in recent_crises) / len(recent_crises)
            avg_iterations = np.mean([c["iterations"] for c in recent_crises])

            # Update competence based on crisis handling
            # High success rate + low iterations = high competence
            competence_signal = success_rate * (1.0 - avg_iterations / 100.0)

            self.self_belief.competence = (
                0.7 * self.self_belief.competence + 0.3 * competence_signal
            )

            # Stable Lyapunov means we're handling crises well
            lyapunov_vals = [c["lyapunov"] for c in recent_crises if c["lyapunov"] is not None]
            if lyapunov_vals and all(l < 0 for l in lyapunov_vals):
                # All negative = converging = good
                self.self_belief.stability += self.belief_update_rate * 0.2
                self.self_belief.stability = min(self.self_belief.stability, 1.0)

    def get_current_pn(self) -> Optional[float]:
        """Get most recent PN value."""
        if self.pn_history:
            return self.pn_history[-1]["value"]
        return None

    def get_pn_statistics(self) -> Dict:
        """Get statistical summary of recent PN trajectory."""
        if len(self.pn_history) < 2:
            return {}

        recent = [p["value"] for p in self.pn_history]

        return {
            "current": recent[-1],
            "mean": np.mean(recent),
            "std": np.std(recent),
            "min": np.min(recent),
            "max": np.max(recent),
            "trend": "increasing" if recent[-1] > np.mean(recent[-10:]) else "stable",
        }

    def generate_self_report(self, verbose: bool = False) -> str:
        """
        Generate natural language description of internal state.

        This is the system's introspective voice - its ability to
        articulate what it's experiencing internally.
        """
        # Apply belief decay first
        self.self_belief.decay()

        belief = self.self_belief
        pn_stats = self.get_pn_statistics()

        # Primary assessment based on dominant belief
        if belief.uncertainty > 0.75:
            primary = "I am experiencing significant internal uncertainty."
        elif belief.stability < 0.3:
            primary = "My internal state is unstable."
        elif belief.competence < 0.3:
            primary = "I am struggling to process inputs effectively."
        elif belief.competence > 0.7 and belief.stability > 0.7:
            primary = "I am operating within normal parameters."
        else:
            primary = "I am functioning, but not optimally."

        if not verbose:
            return primary

        # Verbose report includes more detail
        report_parts = [primary]

        # PN context
        if pn_stats:
            current_pn = pn_stats.get("current", 0)
            if current_pn > 0.8:
                report_parts.append(
                    f"My prediction error is elevated (PN={current_pn:.2f}), "
                    f"indicating computational friction."
                )
            elif current_pn < 0.3:
                report_parts.append(
                    f"My prediction error is low (PN={current_pn:.2f}), "
                    f"indicating stable processing."
                )

        # Crisis history
        if self.crisis_count > 0:
            recent_crises = (
                self.crisis_memory[-5:] if len(self.crisis_memory) >= 5 else self.crisis_memory
            )
            success_rate = sum(c["converged"] for c in recent_crises) / len(recent_crises)

            if success_rate > 0.8:
                report_parts.append(
                    f"I have experienced {self.crisis_count} internal crises, "
                    f"resolving most successfully."
                )
            else:
                report_parts.append(
                    f"I have experienced {self.crisis_count} internal crises, "
                    f"with mixed resolution outcomes."
                )

        # Self-belief summary
        report_parts.append(
            f"My self-assessment: stability={belief.stability:.2f}, "
            f"competence={belief.competence:.2f}, "
            f"uncertainty={belief.uncertainty:.2f}."
        )

        return " ".join(report_parts)

    def should_report_uncertainty(self) -> bool:
        """
        Decide if the system should proactively report its uncertainty.

        Returns True when internal state is sufficiently concerning that
        the user should be made aware.
        """
        current_pn = self.get_current_pn()

        return (
            self.self_belief.uncertainty > 0.75
            or self.self_belief.stability < 0.25
            or (current_pn is not None and current_pn > 0.85)
        )

    def get_diagnostic_summary(self) -> Dict:
        """
        Get comprehensive diagnostic data for logging/debugging.
        """
        return {
            "self_belief": {
                "stability": self.self_belief.stability,
                "competence": self.self_belief.competence,
                "uncertainty": self.self_belief.uncertainty,
                "update_count": self.self_belief.update_count,
            },
            "pn_statistics": self.get_pn_statistics(),
            "crisis_history": {
                "total": self.crisis_count,
                "recent": len(self.crisis_memory[-10:]),
                "success_rate": (
                    (
                        sum(c["converged"] for c in self.crisis_memory[-10:])
                        / len(self.crisis_memory[-10:])
                    )
                    if len(self.crisis_memory) >= 10
                    else None
                ),
            },
            "observation_count": self.total_observations,
            "high_pn_ratio": (
                self.high_pn_count / self.total_observations if self.total_observations > 0 else 0
            ),
        }
