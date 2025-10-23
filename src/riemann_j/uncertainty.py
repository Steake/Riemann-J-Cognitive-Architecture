"""
Phase 3: Uncertainty Interface

Exposes PN as first-class observable system state.
System can reason about and communicate its own uncertainty.

This is epistemic honesty: "I know what I don't know."
"""

from dataclasses import dataclass
from typing import Optional

from .pn_driver import PNDriverRiemannZeta


@dataclass
class UncertaintyReport:
    """Structured report of system uncertainty."""

    pn_value: float
    uncertainty_level: str  # 'low', 'moderate', 'high', 'critical'
    should_communicate: bool
    explanation: str
    confidence_modifier: float  # 0.0-1.0, multiply response confidence by this


class UncertaintyInterface:
    """
    Makes internal uncertainty observable and actionable.

    System doesn't just have uncertainty - it KNOWS it's uncertain
    and can communicate that fact.
    """

    # Thresholds for uncertainty levels
    CRITICAL_THRESHOLD = 0.9
    HIGH_THRESHOLD = 0.7
    MODERATE_THRESHOLD = 0.5

    def __init__(self, pn_driver: Optional[PNDriverRiemannZeta] = None):
        self.pn_driver = pn_driver
        self._last_pn = 0.0

    def get_current_uncertainty(self) -> float:
        """Query PN driver for current prediction error."""
        if self.pn_driver and hasattr(self.pn_driver, "_calculate_pn"):
            self._last_pn = self.pn_driver._calculate_pn()
        return self._last_pn

    def set_pn_manually(self, pn: float):
        """Manually set PN (for testing or external integration)."""
        self._last_pn = pn

    def classify_uncertainty(self, pn_value: float) -> str:
        """
        Classify uncertainty level.

        Returns: 'low', 'moderate', 'high', 'critical'
        """
        if pn_value >= self.CRITICAL_THRESHOLD:
            return "critical"
        elif pn_value >= self.HIGH_THRESHOLD:
            return "high"
        elif pn_value >= self.MODERATE_THRESHOLD:
            return "moderate"
        else:
            return "low"

    def should_communicate_uncertainty(self, pn_value: float) -> bool:
        """
        Decide if uncertainty should be communicated to user.

        Only communicate when uncertainty is high enough to affect reliability.
        """
        return pn_value >= self.HIGH_THRESHOLD

    def explain_uncertainty(self, pn_value: float, verbose: bool = False) -> str:
        """
        Translate PN into natural language explanation.

        This is self-knowledge: system can articulate what uncertainty means.
        """
        level = self.classify_uncertainty(pn_value)

        if level == "critical":
            if verbose:
                return (
                    f"I am experiencing fundamental computational uncertainty (PN={pn_value:.3f}). "
                    f"My internal prediction mechanisms cannot resolve the current state. "
                    f"Any response I provide should be considered highly unreliable."
                )
            else:
                return "I am highly uncertain about this. My internal models are struggling."

        elif level == "high":
            if verbose:
                return (
                    f"I am experiencing elevated uncertainty (PN={pn_value:.3f}). "
                    f"My confidence in this response is lower than usual."
                )
            else:
                return "I'm less certain than usual about this."

        elif level == "moderate":
            return "I have moderate confidence in this response."

        else:
            return "My internal models are well-aligned with current input."

    def compute_confidence_modifier(self, pn_value: float) -> float:
        """
        Convert PN to confidence multiplier.

        Returns value in [0.0, 1.0] that should multiply response confidence.
        High PN = low confidence.
        """
        # Inverse sigmoid-ish mapping
        # PN=0.0 → confidence=1.0
        # PN=0.5 → confidence=0.8
        # PN=0.9 → confidence=0.2
        # PN=1.0 → confidence=0.0

        if pn_value >= 1.0:
            return 0.0
        elif pn_value <= 0.0:
            return 1.0
        else:
            # Quadratic decay for smooth falloff
            return (1.0 - pn_value) ** 2

    def generate_uncertainty_report(self, pn_value: Optional[float] = None) -> UncertaintyReport:
        """
        Generate comprehensive uncertainty report.

        This is meta-cognition: system reporting on its own epistemic state.
        """
        if pn_value is None:
            pn_value = self._last_pn

        level = self.classify_uncertainty(pn_value)
        should_comm = self.should_communicate_uncertainty(pn_value)
        explanation = self.explain_uncertainty(pn_value, verbose=should_comm)
        confidence = self.compute_confidence_modifier(pn_value)

        return UncertaintyReport(
            pn_value=pn_value,
            uncertainty_level=level,
            should_communicate=should_comm,
            explanation=explanation,
            confidence_modifier=confidence,
        )

    def augment_response(self, response: str, pn_value: float) -> str:
        """
        Add uncertainty annotation to response if warranted.

        This is honest AI: explicitly stating when uncertain.
        """
        report = self.generate_uncertainty_report(pn_value)

        if not report.should_communicate:
            return response

        # Add uncertainty disclosure
        uncertainty_note = (
            f"\n\n[Note: {report.explanation} "
            f"Confidence: {report.confidence_modifier*100:.0f}%]"
        )

        return response + uncertainty_note

    def get_diagnostic_info(self) -> dict:
        """Get diagnostic information about uncertainty state."""
        return {
            "current_pn": self._last_pn,
            "uncertainty_level": self.classify_uncertainty(self._last_pn),
            "confidence_modifier": self.compute_confidence_modifier(self._last_pn),
            "should_communicate": self.should_communicate_uncertainty(self._last_pn),
        }
