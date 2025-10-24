#!/usr/bin/env python3
"""
Tests for meta-cognitive monitoring layer.

Validates that the system can accurately observe and reason about
its own internal state.
"""
import time

import numpy as np
import pytest

from riemann_j.architecture import SyntheticState
from riemann_j.metacognition import MetaCognitiveMonitor, SelfBeliefState


class TestSelfBeliefState:
    """Test the self-belief state model."""

    def test_initialization(self):
        """Beliefs start neutral."""
        belief = SelfBeliefState()
        assert belief.stability == 0.5
        assert belief.competence == 0.5
        assert belief.uncertainty == 0.5

    def test_decay(self):
        """Beliefs decay toward neutral over time."""
        belief = SelfBeliefState()
        belief.stability = 0.9
        belief.competence = 0.2
        belief.uncertainty = 0.8

        # Fast forward time
        belief.last_update = time.time() - 100
        belief.decay(decay_rate=0.01)

        # Should move toward 0.5
        assert belief.stability < 0.9
        assert belief.competence > 0.2
        assert belief.uncertainty < 0.8


class TestMetaCognitiveMonitor:
    """Test the meta-cognitive monitoring system."""

    def test_initialization(self):
        """Monitor initializes correctly."""
        monitor = MetaCognitiveMonitor()
        assert len(monitor.pn_history) == 0
        assert monitor.crisis_count == 0
        assert monitor.self_belief.stability == 0.5

    def test_pn_observation(self):
        """Monitor tracks PN values."""
        monitor = MetaCognitiveMonitor()

        monitor.observe_pn(0.5)
        monitor.observe_pn(0.6)
        monitor.observe_pn(0.7)

        assert len(monitor.pn_history) == 3
        assert monitor.get_current_pn() == 0.7

    def test_high_pn_affects_beliefs(self):
        """High PN reduces competence and increases uncertainty."""
        monitor = MetaCognitiveMonitor()

        # Feed consistently high PN
        for _ in range(25):
            monitor.observe_pn(0.9)

        # Should decrease competence and increase uncertainty
        assert monitor.self_belief.competence < 0.5
        assert monitor.self_belief.uncertainty > 0.5

    def test_volatile_pn_affects_stability(self):
        """Volatile PN reduces stability belief."""
        monitor = MetaCognitiveMonitor()

        # Feed volatile PN pattern
        for i in range(25):
            pn = 0.3 if i % 2 == 0 else 0.9
            monitor.observe_pn(pn)

        # Should decrease stability
        assert monitor.self_belief.stability < 0.5

    def test_stable_low_pn_improves_beliefs(self):
        """Stable low PN improves competence and stability."""
        monitor = MetaCognitiveMonitor()

        # Feed consistently low PN
        for _ in range(25):
            monitor.observe_pn(0.3)

        # Should improve beliefs
        assert monitor.self_belief.competence > 0.5
        assert monitor.self_belief.stability > 0.5
        assert monitor.self_belief.uncertainty < 0.5

    def test_crisis_tracking(self):
        """Monitor tracks J-Operator activations."""
        monitor = MetaCognitiveMonitor()

        # Create mock crisis state
        crisis = SyntheticState(
            timestamp=time.time(),
            latent_representation=np.zeros(768),
            source_trigger="TEST",
            p_n_at_creation=0.95,
            is_j_shift_product=True,
            status="CONVERGED",
            analysis={"lyapunov_exp": -1.5, "iterations": 30},
        )

        monitor.observe_j_operator_activation(crisis)

        assert monitor.crisis_count == 1
        assert len(monitor.crisis_memory) == 1
        assert monitor.crisis_memory[0]["converged"] is True

    def test_successful_crisis_improves_competence(self):
        """Successfully resolving crises improves competence belief."""
        monitor = MetaCognitiveMonitor()

        # Multiple successful resolutions
        for i in range(5):
            crisis = SyntheticState(
                timestamp=time.time(),
                latent_representation=np.zeros(768),
                source_trigger="TEST",
                p_n_at_creation=0.95,
                is_j_shift_product=True,
                status="CONVERGED",
                analysis={"lyapunov_exp": -1.5, "iterations": 20 + i},
            )
            monitor.observe_j_operator_activation(crisis)

        # Competence should be relatively high
        assert monitor.self_belief.competence > 0.4

    def test_failed_crisis_reduces_competence(self):
        """Failed crisis resolutions reduce competence."""
        monitor = MetaCognitiveMonitor()

        # Multiple failed resolutions
        for _ in range(5):
            crisis = SyntheticState(
                timestamp=time.time(),
                latent_representation=np.zeros(768),
                source_trigger="TEST",
                p_n_at_creation=0.95,
                is_j_shift_product=True,
                status="ITER_LIMIT_EXCEEDED",
                analysis={"lyapunov_exp": 0.5, "iterations": 100},
            )
            monitor.observe_j_operator_activation(crisis)

        # Competence should decrease
        assert monitor.self_belief.competence < 0.5

    def test_self_report_generation(self):
        """Monitor generates natural language self-reports."""
        monitor = MetaCognitiveMonitor()

        report = monitor.generate_self_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_self_report_reflects_high_uncertainty(self):
        """Self-report mentions uncertainty when PN is high."""
        monitor = MetaCognitiveMonitor()

        # Create high uncertainty condition
        for _ in range(25):
            monitor.observe_pn(0.95)

        report = monitor.generate_self_report()
        assert "uncertainty" in report.lower() or "struggling" in report.lower()

    def test_self_report_reflects_stability(self):
        """Self-report mentions stability when conditions are good."""
        monitor = MetaCognitiveMonitor()

        # Create stable condition
        for _ in range(25):
            monitor.observe_pn(0.2)

        report = monitor.generate_self_report()
        assert "normal" in report.lower() or "stable" in report.lower()

    def test_verbose_self_report(self):
        """Verbose reports include more detail."""
        monitor = MetaCognitiveMonitor()

        # Add some data
        for _ in range(15):
            monitor.observe_pn(0.6)

        brief = monitor.generate_self_report(verbose=False)
        verbose = monitor.generate_self_report(verbose=True)

        assert len(verbose) > len(brief)

    def test_should_report_uncertainty(self):
        """System knows when to proactively report uncertainty."""
        monitor = MetaCognitiveMonitor()

        # Low uncertainty - should not report
        for _ in range(15):
            monitor.observe_pn(0.3)
        assert not monitor.should_report_uncertainty()

        # High uncertainty - should report
        for _ in range(15):
            monitor.observe_pn(0.95)
        assert monitor.should_report_uncertainty()

    def test_pn_statistics(self):
        """Monitor computes PN statistics correctly."""
        monitor = MetaCognitiveMonitor()

        for i in range(20):
            monitor.observe_pn(0.5 + i * 0.01)

        stats = monitor.get_pn_statistics()

        assert "current" in stats
        assert "mean" in stats
        assert "std" in stats
        assert stats["current"] == pytest.approx(0.69, abs=0.01)

    def test_diagnostic_summary(self):
        """Monitor provides comprehensive diagnostic data."""
        monitor = MetaCognitiveMonitor()

        # Add some activity
        for _ in range(20):
            monitor.observe_pn(0.7)

        crisis = SyntheticState(
            timestamp=time.time(),
            latent_representation=np.zeros(768),
            source_trigger="TEST",
            p_n_at_creation=0.95,
            is_j_shift_product=True,
            status="CONVERGED",
            analysis={"lyapunov_exp": -1.5, "iterations": 30},
        )
        monitor.observe_j_operator_activation(crisis)

        diagnostics = monitor.get_diagnostic_summary()

        assert "self_belief" in diagnostics
        assert "pn_statistics" in diagnostics
        assert "crisis_history" in diagnostics
        assert diagnostics["observation_count"] == 20

    def test_pn_history_limit(self):
        """PN history respects max size."""
        monitor = MetaCognitiveMonitor(pn_history_size=10)

        for i in range(20):
            monitor.observe_pn(0.5)

        assert len(monitor.pn_history) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
