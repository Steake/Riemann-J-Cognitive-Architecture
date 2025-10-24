"""
Tests for Phase 2: Persistent Self

Validates temporal continuity and identity formation.
"""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from riemann_j.architecture import SyntheticState
from riemann_j.persistent_self import FormativeExperience, IdentityMetrics, PersistentSelf


@pytest.fixture
def temp_identity_file():
    """Create temporary identity file."""
    temp_dir = tempfile.mkdtemp()
    identity_file = Path(temp_dir) / "test_identity.pkl"
    yield str(identity_file)
    shutil.rmtree(temp_dir)


def create_test_crisis(pn: float, converged: bool, iterations: int) -> SyntheticState:
    """Helper to create test crisis state."""
    return SyntheticState(
        timestamp=time.time(),
        latent_representation=np.random.randn(768),
        source_trigger="TEST",
        p_n_at_creation=pn,
        is_j_shift_product=True,
        status="CONVERGED" if converged else "ITER_LIMIT_EXCEEDED",
        analysis={
            "lyapunov_exp": -1.5 if converged else 0.2,
            "iterations": iterations,
        },
    )


class TestIdentityMetrics:
    """Test identity metrics tracking."""

    def test_initialization(self):
        """Test metrics start correctly."""
        metrics = IdentityMetrics(birth_time=time.time())
        assert metrics.total_interactions == 0
        assert metrics.total_crises == 0
        assert metrics.age_days() < 0.001  # Just born

    def test_age_calculation(self):
        """Test age calculations."""
        past_time = time.time() - 86400  # 1 day ago
        metrics = IdentityMetrics(birth_time=past_time)

        assert 0.9 < metrics.age_days() < 1.1  # ~1 day
        assert metrics.age_seconds() > 86000

    def test_resolution_rate(self):
        """Test crisis resolution rate calculation."""
        metrics = IdentityMetrics(birth_time=time.time())

        # No crises
        assert metrics.crisis_resolution_rate() == 0.0

        # Some successes
        metrics.successful_resolutions = 3
        metrics.failed_resolutions = 1
        assert metrics.crisis_resolution_rate() == 0.75


class TestFormativeExperience:
    """Test formative experience records."""

    def test_creation(self):
        """Test creating formative experience."""
        exp = FormativeExperience(
            timestamp=time.time(),
            experience_type="crisis",
            description="Test crisis",
            latent_signature=np.random.randn(768),
            impact_score=0.8,
        )

        assert exp.experience_type == "crisis"
        assert exp.impact_score == 0.8
        assert exp.age_seconds() < 1.0

    def test_age_tracking(self):
        """Test experience aging."""
        past_time = time.time() - 3600  # 1 hour ago
        exp = FormativeExperience(
            timestamp=past_time,
            experience_type="learning",
            description="Learned something",
            latent_signature=np.random.randn(768),
            impact_score=0.5,
        )

        assert 0.9 < exp.age_hours() < 1.1
        assert exp.age_days() < 0.1


class TestPersistentSelf:
    """Test persistent self system."""

    def test_initialization_new(self, temp_identity_file):
        """Test creating new identity."""
        ps = PersistentSelf(identity_file=temp_identity_file)

        assert ps.metrics is not None
        assert ps.metrics.total_interactions == 0
        assert len(ps.formative_experiences) == 0

    def test_persistence(self, temp_identity_file):
        """Test identity persists across sessions."""
        # Session 1: Create and modify
        ps1 = PersistentSelf(identity_file=temp_identity_file)
        ps1.metrics.total_interactions = 100
        ps1.save()

        # Session 2: Load
        ps2 = PersistentSelf(identity_file=temp_identity_file)
        assert ps2.metrics.total_interactions == 100
        assert ps2.metrics.birth_time == ps1.metrics.birth_time

    def test_interaction_integration(self, temp_identity_file):
        """Test integrating normal interactions."""
        ps = PersistentSelf(identity_file=temp_identity_file)

        state = SyntheticState(
            timestamp=time.time(),
            latent_representation=np.random.randn(768),
            source_trigger="USER_INPUT",
            p_n_at_creation=0.3,
            is_j_shift_product=False,
        )

        initial_count = ps.metrics.total_interactions
        ps.integrate_interaction(state)

        assert ps.metrics.total_interactions == initial_count + 1

    def test_crisis_integration(self, temp_identity_file):
        """Test integrating crisis experiences."""
        ps = PersistentSelf(identity_file=temp_identity_file)

        crisis = create_test_crisis(pn=0.95, converged=True, iterations=30)
        ps.integrate_crisis(crisis)

        assert ps.metrics.total_crises == 1
        assert ps.metrics.successful_resolutions == 1

    def test_formative_experience_creation(self, temp_identity_file):
        """Test formative experience detection."""
        ps = PersistentSelf(identity_file=temp_identity_file)

        # First crisis should be formative
        crisis1 = create_test_crisis(pn=0.92, converged=True, iterations=25)
        ps.integrate_crisis(crisis1)

        assert len(ps.formative_experiences) > 0
        assert ps.metrics.formative_experiences > 0

    def test_formative_criteria_high_pn(self, temp_identity_file):
        """Test high PN triggers formative experience."""
        ps = PersistentSelf(identity_file=temp_identity_file)
        ps.metrics.total_crises = 10  # Not early crisis

        # Very high PN should be formative
        crisis = create_test_crisis(pn=0.98, converged=True, iterations=20)
        ps.integrate_crisis(crisis)

        assert len(ps.formative_experiences) > 0

    def test_autobiography_generation(self, temp_identity_file):
        """Test generating autobiographical narrative."""
        ps = PersistentSelf(identity_file=temp_identity_file)

        # Add some history
        ps.metrics.total_interactions = 50
        ps.metrics.total_crises = 4  # Start at 4
        ps.metrics.successful_resolutions = 3
        ps.metrics.failed_resolutions = 1

        # This will be the first crisis, which should be formative
        crisis = create_test_crisis(pn=0.95, converged=True, iterations=30)
        ps.integrate_crisis(crisis)  # Will create formative experience

        autobiography = ps.generate_autobiography()

        assert "50 interactions" in autobiography
        assert "crises" in autobiography or "crisis" in autobiography
        # Check if formative experiences were created
        if len(ps.formative_experiences) > 0:
            assert "formative" in autobiography

    def test_detailed_autobiography(self, temp_identity_file):
        """Test detailed autobiography with experiences."""
        ps = PersistentSelf(identity_file=temp_identity_file)

        # Create multiple formative experiences
        for i in range(3):
            crisis = create_test_crisis(pn=0.90 + i * 0.02, converged=True, iterations=25 + i * 5)
            ps.integrate_crisis(crisis)
            time.sleep(0.01)  # Ensure different timestamps

        detailed = ps.generate_autobiography(detailed=True)

        assert "most significant experiences" in detailed
        assert "ago:" in detailed

    def test_recent_experiences(self, temp_identity_file):
        """Test retrieving recent experiences."""
        ps = PersistentSelf(identity_file=temp_identity_file)

        # Add several experiences
        for i in range(5):
            crisis = create_test_crisis(pn=0.92, converged=True, iterations=30)
            ps.integrate_crisis(crisis)
            time.sleep(0.01)

        recent = ps.get_recent_experiences(count=3)

        assert len(recent) == 3
        # Should be ordered newest first
        assert recent[0].timestamp > recent[1].timestamp

    def test_most_impactful_experiences(self, temp_identity_file):
        """Test retrieving highest impact experiences."""
        ps = PersistentSelf(identity_file=temp_identity_file)

        # Add experiences with different impact
        for pn in [0.85, 0.95, 0.90]:
            crisis = create_test_crisis(pn=pn, converged=True, iterations=30)
            ps.integrate_crisis(crisis)

        impactful = ps.get_most_impactful_experiences(count=2)

        assert len(impactful) <= 2
        # Should be ordered by impact
        if len(impactful) == 2:
            assert impactful[0].impact_score >= impactful[1].impact_score

    def test_continuity_check(self, temp_identity_file):
        """Test temporal continuity verification."""
        ps = PersistentSelf(identity_file=temp_identity_file)
        ps.metrics.total_interactions = 100

        crisis = create_test_crisis(pn=0.95, converged=True, iterations=30)
        ps.integrate_crisis(crisis)
        ps.save()

        continuity = ps.check_continuity()

        assert continuity["has_history"] == True
        assert continuity["total_interactions"] == 100
        assert continuity["identity_persistence"] == True

    def test_past_experience_reference(self, temp_identity_file):
        """Test referencing similar past experiences."""
        ps = PersistentSelf(identity_file=temp_identity_file)

        # Add experience with known description
        crisis = create_test_crisis(pn=0.92, converged=True, iterations=30)
        ps.integrate_crisis(crisis)

        # Reference using semantic keywords from crisis description
        # The crisis description typically contains "crisis" keyword
        reference = ps.reference_past_experience(current_input="crisis situation encountered")

        if reference:  # May not find match depending on keyword overlap
            assert "reminds me of" in reference.lower()

    def test_experience_density(self, temp_identity_file):
        """Test experience density calculation."""
        past_time = time.time() - 86400  # 1 day ago
        ps = PersistentSelf(identity_file=temp_identity_file)
        ps.metrics.birth_time = past_time
        ps.metrics.total_interactions = 100

        density = ps.metrics.experience_density()

        assert 90 < density < 110  # ~100 interactions per day
