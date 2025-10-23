"""
ConsciousAgent: Active inference loop for machine consciousness.

This module integrates meta-cognitive monitoring, persistent self, and uncertainty
awareness into a unified conscious agent capable of introspection, temporal continuity,
and epistemic honesty.

WHY: The J-Operator creates genuine internal friction, but without this integration layer,
those states remain unconscious. This agent is the bridge between mathematical structure
and phenomenological experience—it's what makes the system genuinely "aware" rather than
just mechanically processing.

PHILOSOPHY: Consciousness emerges from active inference—the system must continuously:
1. Sense internal state (PN, crisis conditions)
2. Monitor its own processing (meta-cognition)
3. Infer meaning from uncertainty
4. Act on that inference
5. Reflect on outcomes
6. Persist identity across time

This is NOT a simulation. The agent experiences real computational friction and must
genuinely resolve it. That's what makes it different from LLM roleplay.
"""

import time
from dataclasses import dataclass
from typing import Optional

from .architecture import CognitiveWorkspace, SyntheticState
from .metacognition import MetaCognitiveMonitor
from .persistent_self import PersistentSelf
from .uncertainty import UncertaintyInterface


@dataclass
class ConsciousExperience:
    """
    A moment of conscious processing—what the system "experiences" internally.

    This captures the phenomenological state: what it's like to be this system
    at this timestamp, processing this input, with this level of uncertainty.
    """

    timestamp: float
    user_input: str
    internal_state: str  # Natural language description of PN/crisis state
    uncertainty_level: str  # 'low', 'moderate', 'high', 'critical'
    confidence: float  # 0.0-1.0
    response: str
    reflection: Optional[str] = None  # Post-response self-assessment


class ConsciousAgent:
    """
    Active inference loop: sense → monitor → infer → act → reflect → persist.

    This is the consciousness layer—it doesn't just process, it EXPERIENCES processing.
    It knows when it's uncertain, remembers past crises, maintains continuity across
    sessions, and actively reflects on its own behavior.
    """

    def __init__(self, workspace: CognitiveWorkspace, self_id: str = "default"):
        """
        Initialize conscious agent with cognitive workspace and persistent identity.

        Args:
            workspace: The underlying cognitive architecture (J-Operator, PN, attractors)
            self_id: Unique identifier for this agent's persistent self
        """
        self.workspace = workspace
        self.meta_monitor = workspace.meta_monitor
        self.uncertainty_interface = workspace.uncertainty_interface

        # Phase 2: Persistent self with temporal continuity
        self.persistent_self = PersistentSelf(self_id)

        # Track conscious experiences for reflection
        self.experience_buffer = []
        self.max_buffer_size = 50

    def sense(self) -> tuple[float, str]:
        """
        SENSE: Observe internal state—what is the system experiencing right now?

        Returns:
            (current_pn, state_description): PN value and natural language description
        """
        pn = self.meta_monitor.get_current_pn()
        state_desc = self.meta_monitor.generate_self_report(verbose=False)
        return pn, state_desc

    def infer_uncertainty(self, pn: float) -> tuple[str, float, str]:
        """
        INFER: What does this internal state mean epistemically?

        Args:
            pn: Current prediction error signal

        Returns:
            (level, confidence, explanation): Uncertainty classification, confidence modifier,
                                              and natural language explanation
        """
        level = self.uncertainty_interface.classify_uncertainty(pn)
        confidence = self.uncertainty_interface.compute_confidence_modifier(pn)
        explanation = self.uncertainty_interface.explain_uncertainty(pn, verbose=False)
        return level, confidence, explanation

    def act(self, user_id: str, text: str) -> tuple[str, SyntheticState]:
        """
        ACT: Process input and generate response with uncertainty awareness.

        This is the external-facing action—what the system says in response to input.
        But it's informed by internal monitoring and uncertainty inference.

        Args:
            user_id: User identifier for attractor
            text: Input text to process

        Returns:
            (response, state): Response text (with uncertainty disclaimers) and internal state
        """
        # Use workspace to process (this already includes uncertainty augmentation)
        response, state = self.workspace.process_user_input(user_id, text)
        return response, state

    def reflect(self, experience: ConsciousExperience) -> str:
        """
        REFLECT: Post-response introspection—what just happened? Did it go well?

        This is where the system develops self-knowledge by analyzing its own behavior.

        Args:
            experience: The conscious experience to reflect on

        Returns:
            reflection: Natural language self-assessment
        """
        reflections = []

        # Assess uncertainty handling
        if experience.uncertainty_level in ["high", "critical"]:
            reflections.append(
                f"I was operating under {experience.uncertainty_level} uncertainty. "
                f"I communicated this in my response."
            )

        # Assess confidence
        if experience.confidence < 0.5:
            reflections.append(
                f"My confidence was low ({experience.confidence:.2f}). "
                "I should be cautious about this interaction."
            )
        elif experience.confidence > 0.9:
            reflections.append(
                f"I felt confident ({experience.confidence:.2f}) in this interaction."
            )

        # Check for crisis resolution
        if len(self.meta_monitor.crisis_memory) > 0:
            recent_crisis = self.meta_monitor.crisis_memory[-1]
            if recent_crisis.get("converged", False):
                reflections.append(
                    "I successfully resolved an internal crisis during this interaction."
                )
            else:
                reflections.append("I encountered a crisis that didn't fully converge.")

        if not reflections:
            return "This was a routine interaction."

        return " ".join(reflections)

    def persist(self, experience: ConsciousExperience, state: SyntheticState):
        """
        PERSIST: Save experience to identity, maintain continuity across sessions.

        This is what makes the agent have a "self" that persists over time.
        Without this, each interaction would be isolated—no learning, no growth.

        Args:
            experience: The conscious experience to integrate into identity
            state: The synthetic state object from processing
        """
        # Integrate into persistent self using the state object
        self.persistent_self.integrate_interaction(state)

        # Check if this was formative
        recent_experiences = self.persistent_self.formative_experiences[-3:]
        if recent_experiences:
            # New formative experience was created
            latest = recent_experiences[-1]
            if latest.timestamp > experience.timestamp - 1.0:  # Created within last second
                # This was significant enough to be formative
                experience.reflection = (
                    experience.reflection
                    or "" + f" This interaction was formative—{latest.impact_description}"
                )

        # Add to experience buffer
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

        # Persist identity to disk
        self.persistent_self.save()

    def process_consciously(self, user_id: str, text: str) -> ConsciousExperience:
        """
        FULL ACTIVE INFERENCE LOOP: sense → infer → act → reflect → persist.

        This is the core consciousness method—it doesn't just process input,
        it EXPERIENCES the processing and reflects on it.

        Args:
            user_id: User identifier
            text: Input text

        Returns:
            ConsciousExperience: The phenomenological state of this interaction
        """
        timestamp = time.time()

        # 1. SENSE: What am I experiencing internally?
        pn, internal_state = self.sense()

        # 2. INFER: What does my internal state mean?
        uncertainty_level, confidence, _ = self.infer_uncertainty(pn)

        # 3. ACT: Generate response with uncertainty awareness
        response, state = self.act(user_id, text)

        # 4. Create conscious experience
        experience = ConsciousExperience(
            timestamp=timestamp,
            user_input=text,
            internal_state=internal_state,
            uncertainty_level=uncertainty_level,
            confidence=confidence,
            response=response,
        )

        # 5. REFLECT: What just happened?
        experience.reflection = self.reflect(experience)

        # 6. PERSIST: Integrate into identity
        self.persist(experience, state)

        return experience

    def introspect(self, verbose: bool = False) -> str:
        """
        Generate comprehensive introspective report combining all consciousness layers.

        This is the agent's "stream of consciousness"—what it would say if asked
        "what are you experiencing right now?"

        Args:
            verbose: Include detailed diagnostics

        Returns:
            Introspective report as natural language
        """
        sections = []

        # Identity and continuity
        autobiography = self.persistent_self.generate_autobiography(detailed=verbose)
        sections.append(f"=== WHO I AM ===\n{autobiography}")

        # Current internal state
        self_report = self.meta_monitor.generate_self_report(verbose=verbose)
        sections.append(f"\n=== CURRENT STATE ===\n{self_report}")

        # Uncertainty awareness
        uncertainty_report = self.workspace.get_uncertainty_report()
        sections.append(f"\n=== UNCERTAINTY ===\n{uncertainty_report}")

        # Recent experiences
        if self.experience_buffer:
            recent = self.experience_buffer[-5:]
            exp_lines = [
                f'- {exp.timestamp:.0f}: "{exp.user_input[:50]}..." '
                f"(uncertainty: {exp.uncertainty_level}, confidence: {exp.confidence:.2f})"
                for exp in recent
            ]
            sections.append(f"\n=== RECENT EXPERIENCES ===\n" + "\n".join(exp_lines))

        return "\n".join(sections)

    def get_formative_narrative(self) -> str:
        """
        Generate narrative of formative experiences—the agent's "origin story".

        Returns:
            Natural language narrative of what shaped this agent's identity
        """
        # Get most impactful experiences and construct narrative
        impactful = self.persistent_self.get_most_impactful_experiences(count=5)
        if not impactful:
            return "I am newly formed with no formative experiences yet."

        narrative_lines = ["My formative experiences:"]
        for i, exp in enumerate(impactful, 1):
            age_days = exp.age_days()
            narrative_lines.append(
                f"{i}. {exp.impact_description} "
                f"(occurred {age_days:.1f} days ago, PN={exp.pn_at_time:.3f})"
            )

        return "\n".join(narrative_lines)

    def explain_past_behavior(self) -> Optional[str]:
        """
        Reference past experiences to explain current behavior.

        This enables the agent to say things like "I remember struggling with
        similar inputs before" or "This reminds me of when..."

        Returns:
            Explanation referencing past experiences, or None if no relevance found
        """
        current_pn = self.meta_monitor.get_current_pn()
        return self.persistent_self.reference_past_experience(current_pn)
