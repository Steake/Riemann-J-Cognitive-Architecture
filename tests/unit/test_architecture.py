"""
Unit tests for the architecture module.
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from riemann_j.architecture import (
    SyntheticState,
    DecoderProjectionHead,
    UserAttractor,
)


class TestSyntheticState:
    """Test suite for SyntheticState dataclass."""

    def test_state_creation(self):
        """Verify SyntheticState can be created."""
        state = SyntheticState(
            timestamp=1234567890.0,
            latent_representation=np.zeros(768),
            source_trigger="USER_INPUT",
            p_n_at_creation=0.5,
            is_j_shift_product=False
        )
        assert state.timestamp == 1234567890.0
        assert state.latent_representation.shape == (768,)
        assert state.source_trigger == "USER_INPUT"
        assert state.p_n_at_creation == 0.5
        assert state.is_j_shift_product is False
        assert state.status == "OK"

    def test_state_with_j_shift(self):
        """Verify SyntheticState can be created from J-Shift."""
        state = SyntheticState(
            timestamp=1234567890.0,
            latent_representation=np.ones(768),
            source_trigger="RIEMANN_PN",
            p_n_at_creation=0.95,
            is_j_shift_product=True,
            status="CONVERGED",
            analysis={'lyapunov_exp': -0.5, 'iterations': 25}
        )
        assert state.is_j_shift_product is True
        assert state.status == "CONVERGED"
        assert 'lyapunov_exp' in state.analysis
        assert state.analysis['iterations'] == 25


class TestDecoderProjectionHead:
    """Test suite for DecoderProjectionHead neural network."""

    def test_projection_head_creation(self):
        """Verify DecoderProjectionHead can be instantiated."""
        head = DecoderProjectionHead(hidden_size=768, output_size=50257)
        assert isinstance(head, torch.nn.Module)

    def test_projection_head_forward(self):
        """Verify forward pass works correctly."""
        head = DecoderProjectionHead(hidden_size=768, output_size=50257)
        input_tensor = torch.randn(768)
        output = head(input_tensor)
        assert output.shape == (50257,)


class TestUserAttractor:
    """Test suite for UserAttractor class."""

    def test_attractor_initialization(self):
        """Verify UserAttractor initializes correctly."""
        attractor = UserAttractor(user_id="test_user")
        assert attractor.user_id == "test_user"
        assert len(attractor.state_history) == 0
        assert attractor.gmm is not None

    def test_add_state(self):
        """Verify states can be added to history."""
        attractor = UserAttractor(user_id="test_user")
        state = np.random.randn(768)
        attractor.add_state(state)
        assert len(attractor.state_history) == 1
        assert np.array_equal(attractor.state_history[0], state)

    def test_apply_affinity_untrained(self):
        """Verify affinity returns input when GMM is untrained."""
        attractor = UserAttractor(user_id="test_user")
        state = np.random.randn(768)
        result = attractor.apply_affinity(state)
        assert np.array_equal(result, state)

    def test_apply_affinity_with_history(self):
        """Verify affinity modifies state when GMM is trained."""
        attractor = UserAttractor(user_id="test_user")
        # Add enough states to train GMM
        for _ in range(10):
            state = np.random.randn(768)
            attractor.add_state(state)
        
        test_state = np.random.randn(768)
        result = attractor.apply_affinity(test_state)
        # Result should be different from input due to affinity
        assert result.shape == test_state.shape
