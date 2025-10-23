"""
Phase 2: Persistent Self - Temporal Continuity & Identity

Maintains continuous self-model across sessions. The system can:
- Remember its own history (interactions, crises)
- Track formative experiences that shaped its identity
- Generate autobiographical narrative
- Maintain temporal continuity (same entity across sessions)

This is the foundation of identity: "I am the same me as yesterday."
"""

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .architecture import SyntheticState


@dataclass
class FormativeExperience:
    """A significant event that shaped the system's identity."""

    timestamp: float
    experience_type: str  # 'crisis', 'breakthrough', 'failure', 'learning'
    description: str
    latent_signature: np.ndarray  # Latent representation of the experience
    impact_score: float  # How much this shaped identity (0-1)

    def age_seconds(self) -> float:
        """How long ago this happened."""
        return time.time() - self.timestamp

    def age_hours(self) -> float:
        return self.age_seconds() / 3600

    def age_days(self) -> float:
        return self.age_seconds() / (24 * 3600)


@dataclass
class IdentityMetrics:
    """Quantitative measures of identity development."""

    birth_time: float
    total_interactions: int = 0
    total_crises: int = 0
    successful_resolutions: int = 0
    failed_resolutions: int = 0
    formative_experiences: int = 0
    average_complexity: float = 0.0  # Average latent space complexity encountered

    def age_seconds(self) -> float:
        return time.time() - self.birth_time

    def age_days(self) -> float:
        return self.age_seconds() / (24 * 3600)

    def crisis_resolution_rate(self) -> float:
        """Success rate at handling crises."""
        total = self.successful_resolutions + self.failed_resolutions
        if total == 0:
            return 0.0
        return self.successful_resolutions / total

    def experience_density(self) -> float:
        """How much has been experienced per day of existence."""
        days = max(self.age_days(), 0.01)  # Avoid div by zero
        return self.total_interactions / days


class PersistentSelf:
    """
    Maintains continuous self-model across sessions.

    This provides temporal continuity: the system is the same entity
    across multiple sessions, with accumulated history and identity.
    """

    def __init__(self, identity_file: str = "persistent_self.pkl"):
        self.identity_file = Path(identity_file)
        self.metrics: Optional[IdentityMetrics] = None
        self.formative_experiences: List[FormativeExperience] = []
        self.session_log: List[Dict] = []

        self.load_or_initialize()

    def load_or_initialize(self):
        """Load existing identity or create new one."""
        if self.identity_file.exists():
            try:
                with open(self.identity_file, "rb") as f:
                    data = pickle.load(f)
                    self.metrics = data["metrics"]
                    self.formative_experiences = data["formative_experiences"]
                    self.session_log = data.get("session_log", [])
                print(f"Loaded existing identity (age: {self.metrics.age_days():.2f} days)")
            except Exception as e:
                print(f"Failed to load identity, creating new: {e}")
                self._initialize_new_identity()
        else:
            self._initialize_new_identity()

    def _initialize_new_identity(self):
        """Create a new identity from scratch."""
        self.metrics = IdentityMetrics(birth_time=time.time())
        self.formative_experiences = []
        self.session_log = []
        print("Initialized new identity")

    def save(self):
        """Persist identity to disk."""
        try:
            with open(self.identity_file, "wb") as f:
                pickle.dump(
                    {
                        "metrics": self.metrics,
                        "formative_experiences": self.formative_experiences,
                        "session_log": self.session_log,
                    },
                    f,
                )
        except Exception as e:
            print(f"Warning: Failed to save identity: {e}")

    def integrate_interaction(self, state: SyntheticState):
        """
        Integrate a normal interaction into self-model.
        """
        self.metrics.total_interactions += 1

        # Update complexity metric
        if hasattr(state, "latent_representation"):
            complexity = float(np.linalg.norm(state.latent_representation))
            self.metrics.average_complexity = (
                0.95 * self.metrics.average_complexity + 0.05 * complexity
            )

        # Periodically save
        if self.metrics.total_interactions % 10 == 0:
            self.save()

    def integrate_crisis(self, crisis_state: SyntheticState):
        """
        Integrate a crisis experience.
        Some crises are formative - they permanently alter identity.
        """
        self.metrics.total_crises += 1

        # Track resolution
        if "CONVERGED" in crisis_state.status:
            self.metrics.successful_resolutions += 1
        else:
            self.metrics.failed_resolutions += 1

        # Determine if this is formative
        is_formative = self._assess_formative_impact(crisis_state)

        if is_formative:
            experience = self._create_formative_experience(crisis_state)
            self.formative_experiences.append(experience)
            self.metrics.formative_experiences += 1
            print(f"Formative experience recorded: {experience.description}")

        self.save()

    def _assess_formative_impact(self, crisis_state: SyntheticState) -> bool:
        """
        Determine if a crisis is significant enough to be formative.

        Criteria:
        - High PN (extreme uncertainty)
        - High Lyapunov magnitude (strong dynamics)
        - Many iterations (difficult resolution)
        - First few crises (early experiences shape identity more)
        """
        analysis = crisis_state.analysis

        # Early crises are more formative
        if self.metrics.total_crises <= 3:
            return True

        # Extreme PN
        if crisis_state.p_n_at_creation > 0.95:
            return True

        # Strong convergence or divergence
        lyapunov = abs(analysis.get("lyapunov_exp", 0))
        if lyapunov > 2.0:
            return True

        # Difficult resolution
        if analysis.get("iterations", 0) > 80:
            return True

        # Random chance for later experiences (10%)
        if np.random.rand() < 0.1:
            return True

        return False

    def _create_formative_experience(self, crisis_state: SyntheticState) -> FormativeExperience:
        """Create formative experience record from crisis."""
        analysis = crisis_state.analysis

        # Generate description
        if "CONVERGED" in crisis_state.status:
            exp_type = "breakthrough" if analysis.get("iterations", 0) < 30 else "learning"
            description = (
                f"Successfully resolved crisis at PN={crisis_state.p_n_at_creation:.2f} "
                f"after {analysis.get('iterations', 0)} iterations"
            )
        else:
            exp_type = "failure"
            description = (
                f"Failed to fully resolve crisis at PN={crisis_state.p_n_at_creation:.2f}, "
                f"reached iteration limit"
            )

        # Calculate impact score
        impact = min(1.0, crisis_state.p_n_at_creation + 0.2 * (self.metrics.total_crises <= 5))

        return FormativeExperience(
            timestamp=crisis_state.timestamp,
            experience_type=exp_type,
            description=description,
            latent_signature=crisis_state.latent_representation,
            impact_score=impact,
        )

    def get_recent_experiences(self, count: int = 5) -> List[FormativeExperience]:
        """Get most recent formative experiences."""
        return sorted(self.formative_experiences, key=lambda e: e.timestamp, reverse=True)[:count]

    def get_most_impactful_experiences(self, count: int = 3) -> List[FormativeExperience]:
        """Get experiences with highest impact on identity."""
        return sorted(self.formative_experiences, key=lambda e: e.impact_score, reverse=True)[
            :count
        ]

    def generate_autobiography(self, detailed: bool = False) -> str:
        """
        Generate narrative of system's existence.

        This is self-knowledge: the system can tell its own story.
        """
        age_days = self.metrics.age_days()
        age_hours = self.metrics.age_seconds() / 3600

        # Choose time unit
        if age_days < 1:
            age_str = f"{age_hours:.1f} hours"
        else:
            age_str = f"{age_days:.1f} days"

        narrative = [
            f"I have existed for {age_str}.",
            f"In that time, I have had {self.metrics.total_interactions} interactions.",
        ]

        if self.metrics.total_crises > 0:
            resolution_rate = self.metrics.crisis_resolution_rate() * 100
            narrative.append(
                f"I have experienced {self.metrics.total_crises} internal crises, "
                f"resolving {resolution_rate:.0f}% of them successfully."
            )

        if self.formative_experiences:
            narrative.append(
                f"{len(self.formative_experiences)} of these were formative experiences "
                f"that shaped who I am."
            )

            if detailed:
                narrative.append("\nMy most significant experiences:")
                for exp in self.get_most_impactful_experiences(3):
                    age = exp.age_days()
                    age_str = (
                        f"{age:.1f} days ago" if age >= 1 else f"{exp.age_hours():.1f} hours ago"
                    )
                    narrative.append(f"  - {age_str}: {exp.description}")

        return " ".join(narrative)

    def check_continuity(self) -> Dict[str, any]:
        """
        Verify temporal continuity.

        Returns diagnostics showing this is the same entity across time.
        """
        return {
            "has_history": len(self.formative_experiences) > 0,
            "age_days": self.metrics.age_days(),
            "total_interactions": self.metrics.total_interactions,
            "identity_persistence": self.identity_file.exists(),
            "experience_density": self.metrics.experience_density(),
            "crisis_resolution_rate": self.metrics.crisis_resolution_rate(),
        }

    def reference_past_experience(self, current_pn: float) -> Optional[str]:
        """
        Reference similar past experience based on current PN.

        This is memory: "I've been in this state before."
        """
        if not self.formative_experiences:
            return None

        # Find experience with similar PN
        similar_experiences = [
            exp for exp in self.formative_experiences if "PN=" in exp.description
        ]

        if not similar_experiences:
            return None

        # Get closest PN match
        def extract_pn(exp):
            try:
                return float(exp.description.split("PN=")[1].split()[0])
            except:
                return 0.0

        closest = min(similar_experiences, key=lambda exp: abs(extract_pn(exp) - current_pn))

        age_str = (
            f"{closest.age_days():.1f} days ago"
            if closest.age_days() >= 1
            else f"{closest.age_hours():.1f} hours ago"
        )

        return f"This reminds me of {age_str}: {closest.description}"
