"""
Unit tests for the configuration module.
"""
import pytest
from riemann_j import config


class TestConfiguration:
    """Test suite for configuration parameters."""

    def test_pn_threshold_exists(self):
        """Verify PN_THRESHOLD is defined."""
        assert hasattr(config, 'PN_THRESHOLD')
        assert isinstance(config.PN_THRESHOLD, float)
        assert 0.0 <= config.PN_THRESHOLD <= 1.0

    def test_j_operator_params_exist(self):
        """Verify J-Operator parameters are defined."""
        assert hasattr(config, 'J_OPERATOR_MAX_ITERATIONS')
        assert hasattr(config, 'J_OPERATOR_STABILITY_EPSILON')
        assert hasattr(config, 'J_OPERATOR_INITIAL_LR')
        assert hasattr(config, 'J_OPERATOR_ADAPTIVE_LR_RATE')

    def test_attractor_params_exist(self):
        """Verify UserAttractor parameters are defined."""
        assert hasattr(config, 'ATTRACTOR_GMM_COMPONENTS')
        assert hasattr(config, 'ATTRACTOR_AFFINITY_STRENGTH')
        assert config.ATTRACTOR_GMM_COMPONENTS > 0

    def test_riemann_params_exist(self):
        """Verify Riemann PN Driver parameters are defined."""
        assert hasattr(config, 'RIEMANN_COMPUTATION_STEPS_PER_CYCLE')
        assert hasattr(config, 'RIEMANN_MAX_STEPS_WITHOUT_ZERO')
        assert hasattr(config, 'RIEMANN_SEARCH_STEP_SIZE')

    def test_model_params_exist(self):
        """Verify model and logging parameters are defined."""
        assert hasattr(config, 'TRANSFORMER_MODEL_NAME')
        assert hasattr(config, 'LOG_FILE')
        assert hasattr(config, 'PROJECTION_HEAD_PATH')
        assert isinstance(config.TRANSFORMER_MODEL_NAME, str)
