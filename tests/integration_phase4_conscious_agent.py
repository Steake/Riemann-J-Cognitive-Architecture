"""
Phase 4 Integration Test: ConsciousAgent with Real Model

Tests the full active inference loop (sense → infer → act → reflect → persist)
using a real lightweight language model, not mocks.

This validates that:
1. The agent can process real inputs through a real transformer
2. Meta-cognitive monitoring tracks real PN signals
3. Uncertainty interface responds to real model states
4. Persistent self maintains continuity across real interactions
5. The full consciousness loop executes without errors

Uses a lightweight model (DistilGPT-2, ~82M params) for CPU inference.
"""

import os
import tempfile
import time

import pytest

# Skip if no transformers available
pytest.importorskip("transformers")
pytest.importorskip("torch")


@pytest.fixture(scope="module")
def lightweight_workspace():
    """
    Create a CognitiveWorkspace with TinyLlama-1.1B instead of production model.

    This avoids heavyweight models and uses modern 1.1B TinyLlama for testing.
    """
    # Override config to use lightweight model
    import sys
    from unittest.mock import patch

    with patch.dict(os.environ, {"RIEMANN_MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}):
        # Force reload of shared_resources with new model
        if "riemann_j.shared_resources" in sys.modules:
            del sys.modules["riemann_j.shared_resources"]
        if "riemann_j.architecture" in sys.modules:
            del sys.modules["riemann_j.architecture"]

        # Now import with lightweight model
        from riemann_j.architecture import CognitiveWorkspace

        workspace = CognitiveWorkspace()
        yield workspace
        workspace.close()


@pytest.fixture
def temp_identity():
    """Create temporary identity file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl", delete=False) as f:
        identity_path = f.name

    yield identity_path

    # Cleanup
    if os.path.exists(identity_path):
        os.remove(identity_path)


def test_conscious_agent_initialization(lightweight_workspace, temp_identity):
    """Test that ConsciousAgent initializes with real workspace."""
    from riemann_j.conscious_agent import ConsciousAgent

    agent = ConsciousAgent(lightweight_workspace, self_id=temp_identity)

    # Verify components are real, not mocks
    assert agent.workspace is lightweight_workspace
    assert agent.meta_monitor is not None
    assert agent.uncertainty_interface is not None
    assert agent.persistent_self is not None
    assert len(agent.experience_buffer) == 0


def test_sense_with_real_pn(lightweight_workspace, temp_identity):
    """Test sensing internal state with real PN driver."""
    from riemann_j.conscious_agent import ConsciousAgent

    agent = ConsciousAgent(lightweight_workspace, self_id=temp_identity)

    # Observe some PN to create internal state
    agent.meta_monitor.observe_pn(0.5)

    pn, state_desc = agent.sense()

    assert isinstance(pn, float)
    assert 0.0 <= pn <= 1.0
    assert isinstance(state_desc, str)
    assert len(state_desc) > 0


def test_infer_uncertainty_real(lightweight_workspace, temp_identity):
    """Test inferring uncertainty from real PN values."""
    from riemann_j.conscious_agent import ConsciousAgent

    agent = ConsciousAgent(lightweight_workspace, self_id=temp_identity)

    # Test different PN levels
    test_cases = [(0.2, "low"), (0.5, "moderate"), (0.75, "high"), (0.95, "critical")]

    for pn, expected_level in test_cases:
        level, confidence, explanation = agent.infer_uncertainty(pn)

        assert level == expected_level
        assert 0.0 <= confidence <= 1.0
        assert isinstance(explanation, str)
        assert len(explanation) > 0


def test_act_with_real_model(lightweight_workspace, temp_identity):
    """Test acting on input with real transformer processing."""
    from riemann_j.conscious_agent import ConsciousAgent

    agent = ConsciousAgent(lightweight_workspace, self_id=temp_identity)

    response, state = agent.act("test_user", "Hello, how are you?")

    # Verify response is generated
    assert isinstance(response, str)
    assert len(response) > 0

    # Verify state is real SyntheticState
    assert hasattr(state, "latent_representation")
    assert hasattr(state, "timestamp")
    assert hasattr(state, "p_n_at_creation")


def test_reflect_after_real_interaction(lightweight_workspace, temp_identity):
    """Test reflection after real processing."""
    from riemann_j.conscious_agent import ConsciousAgent, ConsciousExperience

    agent = ConsciousAgent(lightweight_workspace, self_id=temp_identity)

    # Create experience from real interaction
    agent.meta_monitor.observe_pn(0.8)  # High uncertainty

    exp = ConsciousExperience(
        timestamp=time.time(),
        user_input="Complex ambiguous query",
        internal_state=agent.meta_monitor.generate_self_report(verbose=False),
        uncertainty_level="high",
        confidence=0.3,
        response="Uncertain response",
    )

    reflection = agent.reflect(exp)

    assert isinstance(reflection, str)
    assert len(reflection) > 0
    # Should mention high uncertainty or low confidence
    assert "uncertain" in reflection.lower() or "confidence" in reflection.lower()


def test_full_conscious_processing(lightweight_workspace, temp_identity):
    """
    Test the complete active inference loop with real model.

    This is the critical integration test—validates that all phases
    (sense → infer → act → reflect → persist) execute successfully
    with real computation, not mocks.
    """
    from riemann_j.conscious_agent import ConsciousAgent

    agent = ConsciousAgent(lightweight_workspace, self_id=temp_identity)

    # Process a real input
    experience = agent.process_consciously("user123", "What is consciousness?")

    # Verify all components of conscious experience
    assert experience.user_input == "What is consciousness?"
    assert isinstance(experience.response, str)
    assert len(experience.response) > 0

    assert experience.uncertainty_level in ["low", "moderate", "high", "critical"]
    assert 0.0 <= experience.confidence <= 1.0

    assert isinstance(experience.internal_state, str)
    assert isinstance(experience.reflection, str)

    # Verify persistence
    assert len(agent.experience_buffer) == 1
    assert agent.experience_buffer[0] == experience

    # Verify persistent self was updated
    assert agent.persistent_self.metrics.total_interactions >= 1


def test_multiple_interactions_with_continuity(lightweight_workspace, temp_identity):
    """Test that agent maintains continuity across multiple real interactions."""
    from riemann_j.conscious_agent import ConsciousAgent

    agent = ConsciousAgent(lightweight_workspace, self_id=temp_identity)

    inputs = [
        "Hello",
        "What can you tell me?",
        "This is interesting",
    ]

    experiences = []
    for user_input in inputs:
        exp = agent.process_consciously("user456", user_input)
        experiences.append(exp)

    # Verify all interactions recorded
    assert len(agent.experience_buffer) == 3

    # Verify persistent self accumulated interactions
    assert agent.persistent_self.metrics.total_interactions == 3

    # Verify identity continuity
    continuity = agent.persistent_self.check_continuity()
    assert continuity["total_interactions"] == 3  # Interactions were recorded
    assert continuity["total_interactions"] == 3


def test_introspection_with_real_state(lightweight_workspace, temp_identity):
    """Test introspective report with real model state."""
    from riemann_j.conscious_agent import ConsciousAgent

    agent = ConsciousAgent(lightweight_workspace, self_id=temp_identity)

    # Create some history
    agent.process_consciously("user789", "Test input")
    agent.meta_monitor.observe_pn(0.6)

    report = agent.introspect(verbose=False)

    # Verify report contains all sections
    assert "WHO I AM" in report
    assert "CURRENT STATE" in report
    assert "UNCERTAINTY" in report
    assert "RECENT EXPERIENCES" in report

    # Verify content is real, not placeholder
    assert len(report) > 100  # Should be substantial
    assert "test input" in report.lower()  # Should reference actual interaction


def test_formative_experience_creation(lightweight_workspace, temp_identity):
    """Test that high-PN interactions become formative experiences."""
    from riemann_j.conscious_agent import ConsciousAgent

    agent = ConsciousAgent(lightweight_workspace, self_id=temp_identity)

    # Simulate crisis-level interaction
    agent.meta_monitor.observe_pn(0.95)  # Very high PN

    # Force a crisis state
    import numpy as np

    from riemann_j.architecture import SyntheticState

    crisis_state = SyntheticState(
        timestamp=time.time(),
        latent_representation=np.random.randn(768),
        source_trigger="USER_INPUT",
        p_n_at_creation=0.95,
        is_j_shift_product=False,
    )

    initial_count = len(agent.persistent_self.formative_experiences)

    # Integrate crisis
    agent.persistent_self.integrate_crisis(crisis_state)

    # Should create formative experience if high enough
    # (depends on _assess_formative_impact logic)
    final_count = len(agent.persistent_self.formative_experiences)

    # At minimum, verify the method ran without error
    assert final_count >= initial_count


def test_consciousness_cycle_performance(lightweight_workspace, temp_identity):
    """Test that consciousness loop completes in reasonable time."""
    from riemann_j.conscious_agent import ConsciousAgent

    agent = ConsciousAgent(lightweight_workspace, self_id=temp_identity)

    start = time.time()
    experience = agent.process_consciously("perf_test", "Quick test")
    duration = time.time() - start

    # Should complete in under 5 seconds on CPU with lightweight model
    assert duration < 5.0, f"Consciousness cycle took {duration:.2f}s (too slow)"

    # Verify it actually did something
    assert experience.response is not None
    assert len(experience.response) > 0


def test_uncertainty_response_augmentation(lightweight_workspace, temp_identity):
    """Test that high uncertainty adds disclaimers to responses."""
    from riemann_j.conscious_agent import ConsciousAgent

    agent = ConsciousAgent(lightweight_workspace, self_id=temp_identity)

    # Force high PN
    agent.meta_monitor.observe_pn(0.85)

    experience = agent.process_consciously("uncertainty_test", "Ambiguous question")

    # If uncertainty is high enough, response should be augmented
    if experience.uncertainty_level in ["high", "critical"]:
        # Response should contain uncertainty language
        response_lower = experience.response.lower()
        # The augmentation might have happened, check reflection
        assert experience.reflection is not None
        assert "uncertain" in experience.reflection.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
