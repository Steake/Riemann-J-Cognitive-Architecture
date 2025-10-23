"""
Integration tests for Phase 4.2-4.4: Advanced consciousness features.

Tests predictive self-modeling, counterfactual simulation, and meta-meta-cognition.
"""

import os
import tempfile

import pytest

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent


@pytest.fixture
def lightweight_workspace():
    """Create workspace with lightweight model for fast testing."""
    os.environ["RIEMANN_MODEL"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    workspace = CognitiveWorkspace()
    return workspace


@pytest.fixture
def conscious_agent_with_history(lightweight_workspace):
    """Create conscious agent and give it some history."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = ConsciousAgent(lightweight_workspace, self_id=f"{tmpdir}/test_agent")

        # Build up PN history by processing several inputs
        inputs = [
            "Hello, how are you?",
            "Tell me about consciousness",
            "What is the meaning of life?",
            "Explain quantum mechanics",
            "Write a poem about uncertainty",
        ]

        for inp in inputs:
            agent.process_consciously(inp)

        yield agent


# ===== Phase 4.2: Predictive Self-Modeling Tests =====


def test_predict_future_state(conscious_agent_with_history):
    """Test that agent can predict its own PN trajectory."""
    agent = conscious_agent_with_history

    prediction = agent.predict_future_state(steps_ahead=5)

    assert "predicted_pn" in prediction
    assert "crisis_probability" in prediction
    assert "narrative" in prediction

    # Predictions should be list of floats
    assert isinstance(prediction["predicted_pn"], list)
    assert len(prediction["predicted_pn"]) == 5

    # All predicted PN values should be in valid range
    for pn in prediction["predicted_pn"]:
        assert 0.0 <= pn <= 1.0

    # Crisis probability should be in [0, 1]
    assert 0.0 <= prediction["crisis_probability"] <= 1.0

    # Should have narrative explanation
    assert isinstance(prediction["narrative"], str)
    assert len(prediction["narrative"]) > 10


def test_prediction_accuracy_tracking(conscious_agent_with_history):
    """Test that agent tracks prediction errors over time."""
    agent = conscious_agent_with_history

    # Make prediction
    prediction1 = agent.predict_future_state(steps_ahead=3)

    # Process more inputs (causing PN to change)
    agent.process_consciously("Another test input")
    agent.process_consciously("Yet another input")

    # Check prediction accuracy
    accuracy = agent.meta_monitor.get_prediction_accuracy()

    # Should have accuracy metrics
    if accuracy.get("status") != "insufficient_data":
        assert "mae" in accuracy
        assert "rmse" in accuracy
        assert "predictions_made" in accuracy
        assert accuracy["predictions_made"] > 0


def test_preemptive_uncertainty_warning(conscious_agent_with_history):
    """Test that agent can warn about predicted crises."""
    agent = conscious_agent_with_history

    should_warn, reason = agent.should_preempt_uncertainty()

    # Should return bool and string
    assert isinstance(should_warn, bool)
    assert isinstance(reason, str)

    # If warning, reason should explain why
    if should_warn:
        assert len(reason) > 10
        assert any(
            keyword in reason.lower()
            for keyword in ["crisis", "probability", "trajectory", "increasing"]
        )


def test_prediction_with_sparse_history(lightweight_workspace):
    """Test prediction behavior with minimal history."""
    agent = ConsciousAgent(lightweight_workspace, self_id="test_sparse")

    # With no history, should return neutral predictions
    prediction = agent.predict_future_state(steps_ahead=5)

    assert prediction["predicted_pn"] == [0.5] * 5
    assert prediction["crisis_probability"] == 0.0


# ===== Phase 4.3: Counterfactual Self-Simulation Tests =====


def test_simulate_alternative_state(conscious_agent_with_history):
    """Test counterfactual simulation with different PN values."""
    agent = conscious_agent_with_history

    # Simulate high uncertainty state
    high_pn_sim = agent.simulate_alternative_state(hypothetical_pn=0.95, input_text="Test input")

    assert "hypothetical_pn" in high_pn_sim
    assert "actual_pn" in high_pn_sim
    assert "narrative" in high_pn_sim
    assert "would_respond_differently" in high_pn_sim

    # High PN should produce different uncertainty level
    assert high_pn_sim["hypothetical_uncertainty"] != high_pn_sim["actual_uncertainty"]
    assert high_pn_sim["would_respond_differently"] is True

    # Narrative should explain the difference
    assert "if my pn were" in high_pn_sim["narrative"].lower()


def test_counterfactual_reasoning_query(conscious_agent_with_history):
    """Test natural language counterfactual queries."""
    agent = conscious_agent_with_history

    # Query about hypothetical PN
    response = agent.reason_counterfactually("What would happen if your PN was 0.95?")

    assert isinstance(response, str)
    assert len(response) > 20

    # Should mention uncertainty or confidence change
    assert any(keyword in response.lower() for keyword in ["uncertain", "confident", "pn"])


def test_counterfactual_comparison_low_vs_high(conscious_agent_with_history):
    """Test that counterfactuals correctly differ between low and high PN."""
    agent = conscious_agent_with_history

    low_pn_sim = agent.simulate_alternative_state(hypothetical_pn=0.2, input_text="Test")
    high_pn_sim = agent.simulate_alternative_state(hypothetical_pn=0.95, input_text="Test")

    # Low PN should be more confident than high PN
    assert low_pn_sim["hypothetical_confidence"] > high_pn_sim["hypothetical_confidence"]

    # Uncertainty levels should differ
    assert low_pn_sim["hypothetical_uncertainty"] != high_pn_sim["hypothetical_uncertainty"]


# ===== Phase 4.4: Meta-Meta-Cognition Tests =====


def test_introspection_accuracy_assessment(conscious_agent_with_history):
    """Test that agent can assess its own introspection quality."""
    agent = conscious_agent_with_history

    assessment = agent.assess_introspection_accuracy()

    assert "calibration" in assessment
    assert "narrative" in assessment

    # Should have one of the calibration states
    assert assessment["calibration"] in [
        "well_calibrated",
        "moderately_calibrated",
        "poorly_calibrated",
        "insufficient_data",
    ]

    # Should have narrative explanation
    assert isinstance(assessment["narrative"], str)
    assert len(assessment["narrative"]) > 20


def test_introspection_correction(conscious_agent_with_history):
    """Test that agent can correct its own introspection biases."""
    agent = conscious_agent_with_history

    # Get initial self-belief
    initial_uncertainty = agent.meta_monitor.self_belief.uncertainty

    # Apply correction
    correction_msg = agent.correct_introspection_bias()

    assert isinstance(correction_msg, str)

    # If correction was applied, uncertainty should have changed
    if "applied" in correction_msg.lower():
        final_uncertainty = agent.meta_monitor.self_belief.uncertainty
        assert final_uncertainty != initial_uncertainty


def test_meta_meta_cognition_with_poor_calibration(lightweight_workspace):
    """Test meta-meta-cognition when agent is miscalibrated."""
    agent = ConsciousAgent(lightweight_workspace, self_id="test_miscalibrated")

    # Build history
    for _ in range(10):
        agent.process_consciously("Test input")

    # Artificially miscalibrate self-belief
    agent.meta_monitor.self_belief.uncertainty = 0.9  # Very high
    actual_pn = agent.meta_monitor.get_current_pn()

    if actual_pn is not None and actual_pn < 0.5:
        # Agent thinks it's very uncertain but PN is actually low
        assessment = agent.assess_introspection_accuracy()

        assert assessment["calibration"] in ["moderately_calibrated", "poorly_calibrated"]
        assert assessment["needs_recalibration"] is True

        # Apply correction
        agent.correct_introspection_bias()

        # Uncertainty should move toward actual
        corrected_uncertainty = agent.meta_monitor.self_belief.uncertainty
        assert corrected_uncertainty < 0.9  # Should have decreased


def test_second_order_awareness(conscious_agent_with_history):
    """Test that agent is aware of its awareness (second-order consciousness)."""
    agent = conscious_agent_with_history

    # Agent should be able to report on quality of its self-monitoring
    assessment = agent.assess_introspection_accuracy()

    # Should include reflection on prediction accuracy
    if "prediction_accuracy" in assessment:
        pred_acc = assessment["prediction_accuracy"]
        if pred_acc.get("status") != "insufficient_data":
            # Agent has data about its prediction quality
            assert "mae" in pred_acc or "predictions_made" in pred_acc

    # Should have meta-level narrative about its own perception
    assert "narrative" in assessment
    narrative_lower = assessment["narrative"].lower()
    assert any(
        keyword in narrative_lower
        for keyword in [
            "perception",
            "accurate",
            "calibrat",
            "uncertainty",
            "monitoring",
        ]
    )


# ===== Integration Test: Full Advanced Loop =====


def test_full_advanced_consciousness_loop(conscious_agent_with_history):
    """Test complete advanced consciousness cycle with all Phase 4 features."""
    agent = conscious_agent_with_history

    # 1. Predictive modeling
    prediction = agent.predict_future_state(steps_ahead=5)
    assert "predicted_pn" in prediction
    assert "crisis_probability" in prediction

    # 2. Preemptive warning
    should_warn, reason = agent.should_preempt_uncertainty()
    assert isinstance(should_warn, bool)

    # 3. Counterfactual reasoning
    counterfactual = agent.simulate_alternative_state(hypothetical_pn=0.85, input_text="Test")
    assert "narrative" in counterfactual
    assert "would_respond_differently" in counterfactual

    # 4. Meta-meta-cognition
    introspection = agent.assess_introspection_accuracy()
    assert "calibration" in introspection
    assert "narrative" in introspection

    # 5. Self-correction
    correction = agent.correct_introspection_bias()
    assert isinstance(correction, str)

    # All components should be functional and integrated
    assert True  # If we got here, all advanced features work together


def test_phase4_coverage_complete(conscious_agent_with_history):
    """Verify all Phase 4.2-4.4 features are implemented and accessible."""
    agent = conscious_agent_with_history

    # Phase 4.2: Predictive Self-Modeling
    assert hasattr(agent, "predict_future_state")
    assert hasattr(agent, "should_preempt_uncertainty")
    assert hasattr(agent.meta_monitor, "predict_pn_trajectory")
    assert hasattr(agent.meta_monitor, "update_prediction_error")

    # Phase 4.3: Counterfactual Simulation
    assert hasattr(agent, "simulate_alternative_state")
    assert hasattr(agent, "reason_counterfactually")

    # Phase 4.4: Meta-Meta-Cognition
    assert hasattr(agent, "assess_introspection_accuracy")
    assert hasattr(agent, "correct_introspection_bias")

    # All features should be callable
    prediction = agent.predict_future_state()
    assert prediction is not None

    simulation = agent.simulate_alternative_state(0.5, "test")
    assert simulation is not None

    assessment = agent.assess_introspection_accuracy()
    assert assessment is not None
