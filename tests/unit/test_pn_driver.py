"""
Unit tests for the PN Driver module.
"""
import pytest
import time
from unittest.mock import Mock, patch
from riemann_j.pn_driver import PredictionErrorSignal, PNDriverRiemannZeta


class TestPredictionErrorSignal:
    """Test suite for PredictionErrorSignal dataclass."""

    def test_signal_creation(self):
        """Verify PredictionErrorSignal can be created with valid data."""
        signal = PredictionErrorSignal(
            timestamp=time.time(),
            source="test",
            priority=50,
            p_n=0.5,
            uncompressed_data={'test': 'data'}
        )
        assert signal.source == "test"
        assert signal.priority == 50
        assert signal.p_n == 0.5
        assert signal.uncompressed_data == {'test': 'data'}

    def test_signal_pn_range(self):
        """Verify p_n values are within expected range."""
        signal = PredictionErrorSignal(
            timestamp=time.time(),
            source="test",
            priority=50,
            p_n=0.75,
            uncompressed_data={}
        )
        assert 0.0 <= signal.p_n <= 1.0


class TestPNDriverRiemannZeta:
    """Test suite for PNDriverRiemannZeta thread class."""

    def test_driver_initialization(self):
        """Verify PN Driver initializes correctly."""
        driver = PNDriverRiemannZeta()
        assert driver.current_t == 14.1347
        assert driver.steps_since_last_zero == 0
        assert driver.is_running is True
        assert driver.daemon is True

    def test_calculate_pn_zero_steps(self):
        """Verify PN calculation with zero steps."""
        driver = PNDriverRiemannZeta()
        driver.steps_since_last_zero = 0
        p_n = driver._calculate_pn()
        assert 0.0 <= p_n <= 1.0
        assert p_n < 0.1  # Should be very low with zero steps

    def test_calculate_pn_max_steps(self):
        """Verify PN calculation approaches 1.0 with maximum steps."""
        driver = PNDriverRiemannZeta()
        driver.steps_since_last_zero = 1_000_000
        p_n = driver._calculate_pn()
        assert 0.0 <= p_n <= 1.0
        assert p_n > 0.9  # Should be very high with max steps

    def test_calculate_pn_sigmoid_behavior(self):
        """Verify PN calculation follows sigmoid curve."""
        driver = PNDriverRiemannZeta()
        
        driver.steps_since_last_zero = 0
        p_n_low = driver._calculate_pn()
        
        driver.steps_since_last_zero = 500_000
        p_n_mid = driver._calculate_pn()
        
        driver.steps_since_last_zero = 1_000_000
        p_n_high = driver._calculate_pn()
        
        assert p_n_low < p_n_mid < p_n_high
