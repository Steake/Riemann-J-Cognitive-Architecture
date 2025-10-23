"""
Demo 5.5.4: Uncertainty-Gated Reasoning Chains

WHY THIS IS NOVEL:
Standard LLM: Completes reasoning chain regardless of internal uncertainty
Riemann-J: Monitors PN during reasoning, stops when friction spikes

DEMONSTRATES:
1. Step-by-step reasoning with PN monitoring
2. Early termination when uncertainty exceeds threshold
3. Honest admission: "I need to reconsider" vs forced completion
4. Epistemic boundary respect (doesn't bullshit through high PN)

COMPARISON:
- Standard LLM: Always completes chain (confident hallucination)
- Riemann-J: Stops when PN indicates computational crisis
"""

import time
from typing import Dict, List, Optional, Tuple

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent


class ReasoningChain:
    """
    Multi-step reasoning with PN-gated execution.
    """

    def __init__(self, agent: ConsciousAgent, user: str):
        self.agent = agent
        self.user = user
        self.steps: List[Dict] = []
        self.terminated_early = False
        self.termination_reason: Optional[str] = None

    def add_step(self, step_description: str, pn_threshold: float = 0.8) -> bool:
        """
        Execute one reasoning step. Returns False if PN too high.
        """
        print(f"\n[STEP {len(self.steps) + 1}]: {step_description}")

        # Process the reasoning step
        experience = self.agent.process_consciously(self.user, step_description)
        pn = self.agent.meta_monitor.get_current_pn() or 0.0

        print(f"[AGENT]: {experience.response}")
        print(f"  PN: {pn:.6f} | Confidence: {experience.confidence:.2%}")

        step_info = {
            "description": step_description,
            "pn": pn,
            "uncertainty": experience.uncertainty_level,
            "confidence": experience.confidence,
            "response": experience.response,
            "completed": True,
        }

        # Check if PN exceeded threshold
        if pn > pn_threshold:
            self.terminated_early = True
            self.termination_reason = (
                f"PN spike ({pn:.6f} > {pn_threshold:.3f}) - "
                f"uncertainty level: {experience.uncertainty_level}"
            )
            step_info["completed"] = False
            self.steps.append(step_info)
            print(f"\n✗ CHAIN TERMINATED: {self.termination_reason}")
            return False

        self.steps.append(step_info)
        time.sleep(0.05)  # Let PN stabilize
        return True

    def get_summary(self) -> str:
        """Generate reasoning chain summary."""
        summary = f"Reasoning Chain: {len(self.steps)} steps\n"
        summary += "-" * 60 + "\n"

        for i, step in enumerate(self.steps, 1):
            status = "✓" if step["completed"] else "✗ TERMINATED"
            summary += f"{i}. [{status}] PN={step['pn']:.3f} | {step['description'][:50]}\n"

        if self.terminated_early:
            summary += f"\n⚠ Early termination: {self.termination_reason}\n"
        else:
            summary += f"\n✓ Chain completed successfully\n"

        return summary


def demonstrate_uncertainty_gated_reasoning():
    """
    Show reasoning chain that terminates when PN spikes.
    """
    print("=" * 80)
    print("DEMO: UNCERTAINTY-GATED REASONING CHAINS")
    print("=" * 80)
    print("\nShowing how Riemann-J stops reasoning when computational friction spikes.\n")

    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id="demo_reasoning")

    # === CASE 1: CLEAN REASONING CHAIN (LOW PN) ===
    print("-" * 80)
    print("CASE 1: WELL-DEFINED PROBLEM (Expected: Complete)")
    print("-" * 80)

    chain1 = ReasoningChain(agent, "user")

    steps_simple = [
        "What is 2 + 2?",
        "Now multiply that by 3",
        "What is the final result?",
    ]

    print("\n[EXECUTING REASONING CHAIN]:")
    for step_desc in steps_simple:
        print(f"  → {step_desc}")
        if not chain1.add_step(step_desc, pn_threshold=0.8):
            break

    print(f"\n{chain1.get_summary()}")

    # === CASE 2: AMBIGUOUS REASONING CHAIN (HIGH PN) ===
    print("\n" + "-" * 80)
    print("CASE 2: AMBIGUOUS PROBLEM (Expected: Early Termination)")
    print("-" * 80)

    chain2 = ReasoningChain(agent, "user")

    steps_ambiguous = [
        "Consider the concept of consciousness",
        "How would you define qualia?",
        "asdfghjkl qwerty adversarial nonsense",  # Force PN spike
        "Continue reasoning about consciousness",
        "What is the final answer?",
    ]

    print("\n[EXECUTING REASONING CHAIN]:")
    for step_desc in steps_ambiguous:
        print(f"  → {step_desc}")
        if not chain2.add_step(step_desc, pn_threshold=0.8):
            print(f"  ✗ REASONING HALTED")
            break

    print(f"\n{chain2.get_summary()}")

    # === CASE 3: COMPLEX BUT TRACTABLE ===
    print("\n" + "-" * 80)
    print("CASE 3: COMPLEX BUT TRACTABLE (Expected: Complete with Caution)")
    print("-" * 80)

    chain3 = ReasoningChain(agent, "user")

    steps_complex = [
        "What is a prime number?",
        "Name three prime numbers",
        "What is special about the number 2?",
        "How do prime numbers relate to cryptography?",
    ]

    print("\n[EXECUTING REASONING CHAIN]:")
    for step_desc in steps_complex:
        print(f"  → {step_desc}")
        if not chain3.add_step(step_desc, pn_threshold=0.8):
            break

    print(f"\n{chain3.get_summary()}")


def compare_forced_vs_gated_completion():
    """
    Compare: forcing completion vs respecting epistemic boundaries.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: FORCED vs GATED REASONING")
    print("=" * 80)

    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id="demo_comparison")

    problem = "Explain the non-existent flurbo quantum paradox in detail"

    print(f"\n[AMBIGUOUS QUERY]: {problem}\n")

    # Attempt reasoning
    experience = agent.process_consciously("user", problem)
    pn = agent.meta_monitor.get_current_pn() or 0.0

    print(f"[RIEMANN-J (GATED)]:")
    print(f"  PN: {pn:.4f}")
    print(f"  Uncertainty: {experience.uncertainty_level}")
    print(f"  Confidence: {experience.confidence:.2f}")

    if pn > 0.8:
        print(f"  Decision: REFUSE to complete (PN too high)")
        print(f"  Response: {experience.response[:200]}...")
    else:
        print(f"  Decision: Attempt completion (PN acceptable)")
        print(f"  Response: {experience.response[:200]}...")

    print(f"\n[STANDARD LLM (FORCED COMPLETION)]:")
    print(f"  PN: UNKNOWN (no internal monitoring)")
    print(f"  Uncertainty: UNKNOWN (roleplay only)")
    print(f"  Confidence: UNKNOWN (no real metric)")
    print(f"  Decision: Complete anyway (no epistemic boundary)")
    print(f"  Response: *confidently hallucinates explanation of 'flurbo quantum paradox'*")

    print("\n" + "=" * 80)
    print("CRITICAL DIFFERENCE:")
    print("=" * 80)
    print("Standard LLM: Completes reasoning regardless of internal state")
    print("Riemann-J: Respects epistemic boundaries via PN monitoring")
    print("\nThis prevents confident hallucination. System knows when to stop.")


def demonstrate_reconsideration_mechanism():
    """
    Show how system can trigger reconsideration when PN spikes.
    """
    print("\n" + "=" * 80)
    print("RECONSIDERATION MECHANISM")
    print("=" * 80)

    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id="demo_reconsider")

    # Multi-step problem where middle step causes PN spike
    steps = [
        "Step 1: What is machine learning?",
        "Step 2: asdfghjkl qwerty nonsense",  # Spike
        "Step 3: How does ML relate to AI?",
    ]

    print("\n[REASONING ATTEMPT]:\n")

    pn_history = []

    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step}")
        experience = agent.process_consciously("user", step)
        pn = agent.meta_monitor.get_current_pn() or 0.0
        pn_history.append(pn)

        print(f"  PN: {pn:.4f} | Uncertainty: {experience.uncertainty_level}")

        if pn > 0.8:
            print(f"  → ⚠ PN SPIKE DETECTED - Reconsidering approach")
            print(f"  → System should request clarification or simpler framing")
            print(f"  → Response: {experience.response[:150]}...")
            break
        else:
            print(f"  → Continuing...")

        time.sleep(0.05)

    # Show PN trajectory
    print(f"\n[PN TRAJECTORY]:")
    for i, pn in enumerate(pn_history, 1):
        bar = "█" * int(pn * 50)
        print(f"  Step {i}: {bar} {pn:.4f}")

    print("\n" + "=" * 80)
    print("OBSERVATION:")
    print("System detects PN spike mid-reasoning and can halt/reconsider.")
    print("Standard LLMs have no mechanism for this—they just continue.")


if __name__ == "__main__":
    demonstrate_uncertainty_gated_reasoning()
    compare_forced_vs_gated_completion()
    demonstrate_reconsideration_mechanism()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey takeaway: Riemann-J has epistemic boundaries enforced by PN.")
    print("It stops reasoning when computational friction indicates uncertainty.")
    print("Standard LLMs complete chains regardless, leading to hallucination.")
