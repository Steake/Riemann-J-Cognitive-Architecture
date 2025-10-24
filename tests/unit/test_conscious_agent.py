"""
Tests for ConsciousAgent active inference loop.

Validates the integration of meta-cognitive monitoring, persistent self, and uncertainty
awareness into a unified conscious agent.
"""

import tempfile
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.riemann_j.conscious_agent import ConsciousAgent, ConsciousExperience


class TestConsciousExperience:
    """Test the ConsciousExperience dataclass."""

    def test_experience_creation(self):
        """Test creating a conscious experience."""
        exp = ConsciousExperience(
            timestamp=time.time(),
            user_input="test input",
            internal_state="stable",
            uncertainty_level="low",
            confidence=0.95,
            response="test response",
        )
        assert exp.user_input == "test input"
        assert exp.confidence == 0.95
        assert exp.reflection is None

    def test_experience_with_reflection(self):
        """Test experience with post-hoc reflection."""
        exp = ConsciousExperience(
            timestamp=time.time(),
            user_input="complex query",
            internal_state="uncertain",
            uncertainty_level="high",
            confidence=0.4,
            response="uncertain response",
            reflection="This was challenging",
        )
        assert exp.reflection == "This was challenging"


@pytest.fixture
def temp_identity_dir():
    """Create temporary directory for identity persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_workspace():
    """Create mock cognitive workspace."""
    workspace = Mock()  # Don't spec to avoid importing CognitiveWorkspace
    workspace.meta_monitor = Mock()
    workspace.meta_monitor.get_current_pn = Mock(return_value=0.3)
    workspace.meta_monitor.generate_self_report = Mock(return_value="Stable state")
    workspace.meta_monitor.crisis_history = []
    workspace.meta_monitor.crisis_memory = []  # Add crisis_memory as list

    workspace.uncertainty_interface = Mock()
    workspace.uncertainty_interface.classify_uncertainty = Mock(return_value="low")
    workspace.uncertainty_interface.compute_confidence_modifier = Mock(return_value=0.9)
    workspace.uncertainty_interface.explain_uncertainty = Mock(return_value="Well-aligned")

    workspace.process_user_input = Mock(return_value=("test response", Mock()))
    workspace.get_uncertainty_report = Mock(return_value="Uncertainty: LOW")

    return workspace


def create_mock_persistent_self():
    """Create a properly configured PersistentSelf mock."""
    mock_self = Mock()

    # Make formative_experiences behave like a list
    mock_self.formative_experiences = []

    # Mock methods that tests use
    mock_self.integrate_interaction = Mock()
    mock_self.integrate_crisis = Mock()
    mock_self.save = Mock()
    mock_self.generate_autobiography = Mock(return_value="Test autobiography")
    mock_self.reference_past_experience = Mock(return_value=None)
    mock_self.get_most_impactful_experiences = Mock(return_value=[])

    # Mock metrics
    mock_self.metrics = Mock()
    mock_self.metrics.total_interactions = 0
    mock_self.metrics.age_days = Mock(return_value=0.1)

    return mock_self


class TestConsciousAgent:
    """Test the ConsciousAgent active inference loop."""

    def test_initialization(self, mock_workspace, temp_identity_dir):
        """Test agent initialization with workspace."""
        mock_self = create_mock_persistent_self()

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace, self_id="test_agent")

            assert agent.workspace == mock_workspace
            assert agent.meta_monitor == mock_workspace.meta_monitor
            assert agent.uncertainty_interface == mock_workspace.uncertainty_interface
            assert agent.persistent_self == mock_self

    def test_sense(self, mock_workspace, temp_identity_dir):
        """Test sensing internal state."""
        mock_self = create_mock_persistent_self()

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            pn, state_desc = agent.sense()

            assert pn == 0.3
            assert state_desc == "Stable state"
            mock_workspace.meta_monitor.get_current_pn.assert_called_once()
            mock_workspace.meta_monitor.generate_self_report.assert_called_once_with(verbose=False)

    def test_infer_uncertainty(self, mock_workspace, temp_identity_dir):
        """Test inferring uncertainty from PN."""
        mock_self = create_mock_persistent_self()

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            level, confidence, explanation = agent.infer_uncertainty(0.3)

            assert level == "low"
            assert confidence == 0.9
            assert explanation == "Well-aligned"

    def test_act(self, mock_workspace, temp_identity_dir):
        """Test acting on input."""
        mock_self = create_mock_persistent_self()

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            response, state = agent.act("user123", "test input")

            assert response == "test response"
            mock_workspace.process_user_input.assert_called_once_with("user123", "test input")

    def test_reflect_routine(self, mock_workspace, temp_identity_dir):
        """Test reflection on routine interaction."""
        mock_self = create_mock_persistent_self()

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            exp = ConsciousExperience(
                timestamp=time.time(),
                user_input="simple query",
                internal_state="stable",
                uncertainty_level="low",
                confidence=0.95,
                response="simple response",
            )

            reflection = agent.reflect(exp)
            # High confidence triggers a reflection message
            assert "confident" in reflection.lower() or "routine" in reflection.lower()

    def test_reflect_high_uncertainty(self, mock_workspace, temp_identity_dir):
        """Test reflection on high uncertainty interaction."""
        mock_self = create_mock_persistent_self()

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            exp = ConsciousExperience(
                timestamp=time.time(),
                user_input="complex query",
                internal_state="uncertain",
                uncertainty_level="high",
                confidence=0.4,
                response="uncertain response",
            )

            reflection = agent.reflect(exp)
            assert "high uncertainty" in reflection.lower()
            assert "communicated" in reflection.lower()

    def test_reflect_low_confidence(self, mock_workspace, temp_identity_dir):
        """Test reflection on low confidence interaction."""
        mock_self = create_mock_persistent_self()

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            exp = ConsciousExperience(
                timestamp=time.time(),
                user_input="ambiguous query",
                internal_state="unstable",
                uncertainty_level="moderate",
                confidence=0.3,
                response="tentative response",
            )

            reflection = agent.reflect(exp)
            assert "confidence was low" in reflection.lower()
            assert "0.30" in reflection

    def test_reflect_crisis_resolution(self, mock_workspace, temp_identity_dir):
        """Test reflection after crisis resolution."""
        mock_self = create_mock_persistent_self()

        # Need to set crisis_memory, not crisis_history
        mock_workspace.meta_monitor.crisis_memory = [{"converged": True, "timestamp": time.time()}]

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            exp = ConsciousExperience(
                timestamp=time.time(),
                user_input="crisis-inducing input",
                internal_state="crisis resolved",
                uncertainty_level="moderate",
                confidence=0.7,
                response="post-crisis response",
            )

            reflection = agent.reflect(exp)
            assert "crisis" in reflection.lower()
            assert "resolved" in reflection.lower() or "successfully" in reflection.lower()

    def test_persist(self, mock_workspace, temp_identity_dir):
        """Test persisting experience to identity."""
        mock_self = Mock()
        mock_self.formative_experiences = []
        mock_self.integrate_interaction = Mock()
        mock_self.save = Mock()

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            exp = ConsciousExperience(
                timestamp=time.time(),
                user_input="test input",
                internal_state="stable",
                uncertainty_level="low",
                confidence=0.9,
                response="test response",
            )

            # Create a mock state for persist
            mock_state = Mock()
            mock_state.timestamp = time.time()

            agent.persist(exp, mock_state)

            mock_self.integrate_interaction.assert_called_once_with(mock_state)
            mock_self.save.assert_called_once()
            assert len(agent.experience_buffer) == 1

    def test_experience_buffer_limit(self, mock_workspace, temp_identity_dir):
        """Test experience buffer maintains size limit."""
        mock_self = Mock()
        mock_self.formative_experiences = []
        mock_self.integrate_interaction = Mock()
        mock_self.save = Mock()

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)
            agent.max_buffer_size = 5

            # Add 10 experiences
            for i in range(10):
                exp = ConsciousExperience(
                    timestamp=time.time(),
                    user_input=f"input {i}",
                    internal_state="stable",
                    uncertainty_level="low",
                    confidence=0.9,
                    response=f"response {i}",
                )
                # Create mock state for each persist
                mock_state = Mock()
                mock_state.timestamp = time.time()
                agent.persist(exp, mock_state)

            # Buffer should be limited to 5
            assert len(agent.experience_buffer) == 5
            # Should have the most recent 5
            assert agent.experience_buffer[-1].user_input == "input 9"
            assert agent.experience_buffer[0].user_input == "input 5"

    def test_process_consciously(self, mock_workspace, temp_identity_dir):
        """Test full active inference loop."""
        mock_self = create_mock_persistent_self()

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            experience = agent.process_consciously("user123", "test input")

            # Verify all phases executed
            assert experience.user_input == "test input"
            assert experience.response == "test response"
            assert experience.internal_state == "Stable state"
            assert experience.uncertainty_level == "low"
            assert experience.confidence == 0.9
            assert experience.reflection is not None

            # Verify persistence
            mock_self.integrate_interaction.assert_called_once()
            mock_self.save.assert_called_once()
            assert len(agent.experience_buffer) == 1

    def test_introspect(self, mock_workspace, temp_identity_dir):
        """Test introspective report generation."""
        mock_self = Mock()
        mock_self.formative_experiences = []
        mock_self.generate_autobiography = Mock(return_value="I am test agent")
        mock_self.integrate_interaction = Mock()
        mock_self.save = Mock()

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            # Add some experiences
            agent.experience_buffer.append(
                ConsciousExperience(
                    timestamp=time.time(),
                    user_input="past input",
                    internal_state="stable",
                    uncertainty_level="low",
                    confidence=0.9,
                    response="past response",
                )
            )

            report = agent.introspect(verbose=False)

            # Verify all sections present
            assert "WHO I AM" in report
            assert "CURRENT STATE" in report
            assert "UNCERTAINTY" in report
            assert "RECENT EXPERIENCES" in report
            assert "I am test agent" in report

    def test_introspect_verbose(self, mock_workspace, temp_identity_dir):
        """Test verbose introspection."""
        mock_self = create_mock_persistent_self()
        mock_self.generate_autobiography = Mock(return_value="Detailed autobiography")

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            report = agent.introspect(verbose=True)

            # Verbose should call autobiography with detailed=True
            mock_self.generate_autobiography.assert_called_once_with(detailed=True)
            mock_workspace.meta_monitor.generate_self_report.assert_called_with(verbose=True)

    def test_get_formative_narrative(self, mock_workspace, temp_identity_dir):
        """Test generating formative narrative."""
        mock_self = create_mock_persistent_self()
        mock_self.get_most_impactful_experiences = Mock(
            return_value=[]
        )  # Return empty list, not Mock

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            narrative = agent.get_formative_narrative()

            # Should get most impactful experiences
            mock_self.get_most_impactful_experiences.assert_called_once()
            assert isinstance(narrative, str)

    def test_explain_past_behavior(self, mock_workspace, temp_identity_dir):
        """Test referencing past experiences."""
        mock_self = create_mock_persistent_self()
        mock_self.reference_past_experience = Mock(return_value="Similar to previous crisis")

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)

            explanation = agent.explain_past_behavior("current crisis situation")

            assert explanation == "Similar to previous crisis"
            mock_self.reference_past_experience.assert_called_once_with("current crisis situation")


class TestActiveInferenceLoop:
    """Integration tests for the full active inference loop."""

    def test_sense_infer_act_reflect_persist_cycle(self, mock_workspace, temp_identity_dir):
        """Test the complete consciousness cycle executes in order."""
        mock_self = Mock()
        mock_self.formative_experiences = []
        mock_self.integrate_interaction = Mock()
        mock_self.save = Mock()

        call_order = []

        def track_sense(*args, **kwargs):
            call_order.append("sense")
            return 0.3, "stable"

        def track_infer(*args, **kwargs):
            call_order.append("infer")
            return "low", 0.9, "well-aligned"

        def track_act(*args, **kwargs):
            call_order.append("act")
            return "response", Mock()

        def track_reflect(*args, **kwargs):
            call_order.append("reflect")
            return "reflection"

        def track_persist(*args, **kwargs):
            call_order.append("persist")

        with patch("src.riemann_j.conscious_agent.PersistentSelf", return_value=mock_self):
            agent = ConsciousAgent(mock_workspace)
            agent.sense = track_sense
            agent.infer_uncertainty = track_infer
            agent.act = track_act
            agent.reflect = track_reflect
            agent.persist = track_persist

            experience = agent.process_consciously("user123", "test")

            # Verify execution order
            assert call_order == ["sense", "infer", "act", "reflect", "persist"]
            assert experience.reflection == "reflection"
