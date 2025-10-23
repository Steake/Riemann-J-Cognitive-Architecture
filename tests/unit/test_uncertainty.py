"""Tests for Phase 3: Uncertainty Interface"""

import pytest

from riemann_j.uncertainty import UncertaintyInterface, UncertaintyReport


class TestUncertaintyInterface:
    """Test uncertainty interface."""

    def test_initialization(self):
        """Test interface initializes."""
        ui = UncertaintyInterface()
        assert ui._last_pn == 0.0

    def test_manual_pn_setting(self):
        """Test manually setting PN."""
        ui = UncertaintyInterface()
        ui.set_pn_manually(0.75)
        assert ui._last_pn == 0.75

    def test_uncertainty_classification(self):
        """Test PN classification into levels."""
        ui = UncertaintyInterface()

        assert ui.classify_uncertainty(0.2) == "low"
        assert ui.classify_uncertainty(0.6) == "moderate"
        assert ui.classify_uncertainty(0.8) == "high"
        assert ui.classify_uncertainty(0.95) == "critical"

    def test_should_communicate(self):
        """Test communication threshold."""
        ui = UncertaintyInterface()

        assert not ui.should_communicate_uncertainty(0.5)  # Low - don't communicate
        assert ui.should_communicate_uncertainty(0.85)  # High - communicate

    def test_explain_uncertainty_brief(self):
        """Test brief uncertainty explanations."""
        ui = UncertaintyInterface()

        low_exp = ui.explain_uncertainty(0.3, verbose=False)
        assert "well-aligned" in low_exp.lower()

        high_exp = ui.explain_uncertainty(0.85, verbose=False)
        assert "certain" in high_exp.lower()  # "less certain" is valid

    def test_explain_uncertainty_verbose(self):
        """Test verbose uncertainty explanations."""
        ui = UncertaintyInterface()

        critical_exp = ui.explain_uncertainty(0.95, verbose=True)
        assert "PN=" in critical_exp
        assert "fundamental" in critical_exp.lower()

    def test_confidence_modifier(self):
        """Test PN to confidence conversion."""
        ui = UncertaintyInterface()

        # Low PN = high confidence
        assert ui.compute_confidence_modifier(0.0) == 1.0
        assert ui.compute_confidence_modifier(0.1) > 0.8

        # High PN = low confidence
        assert ui.compute_confidence_modifier(0.9) < 0.2
        assert ui.compute_confidence_modifier(1.0) == 0.0

    def test_uncertainty_report(self):
        """Test generating uncertainty report."""
        ui = UncertaintyInterface()
        ui.set_pn_manually(0.85)

        report = ui.generate_uncertainty_report()

        assert isinstance(report, UncertaintyReport)
        assert report.pn_value == 0.85
        assert report.uncertainty_level == "high"
        assert report.should_communicate == True
        assert 0.0 <= report.confidence_modifier <= 1.0

    def test_augment_response_low_uncertainty(self):
        """Test response augmentation with low uncertainty."""
        ui = UncertaintyInterface()

        response = "This is my answer."
        augmented = ui.augment_response(response, pn_value=0.3)

        # Low uncertainty - no annotation added
        assert augmented == response

    def test_augment_response_high_uncertainty(self):
        """Test response augmentation with high uncertainty."""
        ui = UncertaintyInterface()

        response = "This is my answer."
        augmented = ui.augment_response(response, pn_value=0.85)

        # High uncertainty - annotation added
        assert "[Note:" in augmented
        assert "Confidence:" in augmented
        assert augmented.startswith(response)

    def test_diagnostic_info(self):
        """Test getting diagnostic information."""
        ui = UncertaintyInterface()
        ui.set_pn_manually(0.75)

        diag = ui.get_diagnostic_info()

        assert diag["current_pn"] == 0.75
        assert diag["uncertainty_level"] == "high"
        assert "confidence_modifier" in diag
        assert "should_communicate" in diag
