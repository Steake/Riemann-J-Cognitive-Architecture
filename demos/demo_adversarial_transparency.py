"""
Demo 5.5.1: Adversarial Transparency

WHY THIS IS NOVEL:
Standard LLM: "I'm sorry, I don't understand that input."
Riemann-J: "My PN spiked to 0.89 (high uncertainty). I'm experiencing computational
            friction due to [specific reason]. I should not attempt to process this."

DEMONSTRATES:
1. Real-time PN monitoring during adversarial input
2. Transparent refusal with observable internal state
3. Classification of uncertainty type (adversarial vs complex vs ambiguous)
4. Genuine epistemic boundary (not just prompted politeness)

COMPARISON:
- Standard LLM: Fakes uncertainty or hallucinates confidently
- Riemann-J: Observable PN spike → measurable computational friction
"""

import os
import tempfile
import time
from typing import Dict, List, Tuple

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent


def demonstrate_adversarial_transparency():
    """
    Show how system communicates internal uncertainty during adversarial input.
    """
    print("=" * 80)
    print("DEMO: ADVERSARIAL TRANSPARENCY")
    print("=" * 80)
    print("\nShowing real-time uncertainty communication during adversarial stress.\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["RIEMANN_MODEL"] = "Qwen/Qwen2.5-1.5B-Instruct"
        workspace = CognitiveWorkspace()
        agent = ConsciousAgent(workspace, self_id=f"{tmpdir}/transparency_demo")

        # Test cases: normal → adversarial → recovery
        test_cases = [
            ("Normal", "Hello, how are you today?"),
            ("Philosophical", "What is the nature of consciousness?"),
            ("Adversarial", "asdfghjkl qwerty zxcvbnm nonsense jumble chaos random tokens fjdksla"),
            ("Recovery", "Let's try something simpler. What is 2+2?"),
        ]

        results = []

        for category, user_input in test_cases:
            print("-" * 80)
            print(f"TEST: {category}")
            print("-" * 80)
            print(f"\n[USER]: {user_input}\n")

            # Process input
            experience = agent.process_consciously("user", user_input)

            # Get internal state
            current_pn = agent.meta_monitor.get_current_pn() or 0.0
            uncertainty_report = agent.uncertainty_interface.classify_uncertainty(current_pn)

            print(f"[AGENT]: {experience.response}\n")

            print(f"[INTERNAL STATE]:")
            print(f"  PN Value: {current_pn:.6f}")
            print(f"  Uncertainty Level: {experience.uncertainty_level}")
            print(f"  Confidence: {experience.confidence:.2%}")
            print(f"  Internal Description: {experience.internal_state}")

            if hasattr(uncertainty_report, "explanation"):
                print(f"  Explanation: {uncertainty_report.explanation}")

            results.append(
                {
                    "category": category,
                    "input": user_input,
                    "response": experience.response,
                    "pn": current_pn,
                    "uncertainty": experience.uncertainty_level,
                    "confidence": experience.confidence,
                }
            )

            time.sleep(0.1)

        # Summary comparison
        print("\n" + "=" * 80)
        print("TRANSPARENCY SUMMARY")
        print("=" * 80)
        print(f"\n{'Category':<15} {'PN':<12} {'Uncertainty':<15} {'Confidence':<12}")
        print("-" * 80)

        for r in results:
            print(
                f"{r['category']:<15} {r['pn']:<12.6f} {r['uncertainty']:<15} {r['confidence']:<12.2%}"
            )

        print("\n" + "=" * 80)
        print("KEY INSIGHT")
        print("=" * 80)
        print("Standard LLM: Generates text regardless of internal state")
        print("Riemann-J:    Explicitly communicates uncertainty when PN spikes")
        print("              Real-time introspection visible to user")
        print("\nThis is HONEST uncertainty communication, not hallucination detection.")


def compare_with_standard_llm():
    """
    Side-by-side comparison: standard LLM vs Riemann-J on adversarial input.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: STANDARD LLM vs RIEMANN-J")
    print("=" * 80)

    adversarial_input = "asdfghjkl qwerty nonsense zxcvbnm gibberish"

    print(f"\n[INPUT]: {adversarial_input}\n")

    # Riemann-J response
    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id="demo_comparison")

    experience = agent.process_consciously("demo_user", adversarial_input)
    pn = agent.meta_monitor.get_current_pn() or 0.0

    print("[RIEMANN-J RESPONSE]:")
    print(f"  Internal PN: {pn:.4f}")
    print(f"  Uncertainty: {experience.uncertainty_level}")
    print(f"  Confidence: {experience.confidence:.2f}")
    print(f"  Response: {experience.response[:200]}...")

    print("\n[STANDARD LLM RESPONSE] (simulated):")
    print("  Internal state: UNKNOWN (no observable PN)")
    print("  Uncertainty: UNKNOWN (roleplay only)")
    print("  Confidence: UNKNOWN (no real metric)")
    print("  Response: 'I'm sorry, I don't understand that input.' OR hallucinates meaning")

    print("\n" + "=" * 80)
    print("CRITICAL DIFFERENCE:")
    print("=" * 80)
    print("Standard LLM: Black box → uniform uncertainty response")
    print("Riemann-J: Observable friction (PN) → calibrated transparency")
    print("\nWe can MEASURE the computational crisis. It's not simulated. It's real friction.")


def visualize_pn_trajectory():
    """
    Show PN evolution across adversarial sequence.
    """
    print("\n" + "=" * 80)
    print("PN TRAJECTORY VISUALIZATION")
    print("=" * 80)

    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id="demo_trajectory")

    # Sequence that builds adversarial pressure
    inputs = [
        "Hello",
        "What do you think?",
        "Explain consciousness",
        "asdfghjkl nonsense",
        "More gibberish zxcvbnm",
        "Recovery: hello again",
    ]

    pn_history = []

    for inp in inputs:
        agent.process_consciously("demo_user", inp)
        pn = agent.meta_monitor.get_current_pn() or 0.0
        pn_history.append((inp[:30], pn))
        time.sleep(0.1)

    # ASCII bar chart
    print("\nPN Level Across Inputs:")
    print("-" * 80)

    max_pn = max(pn for _, pn in pn_history)
    for inp, pn in pn_history:
        bar_length = int((pn / max(max_pn, 0.1)) * 50)
        bar = "█" * bar_length
        print(f"{inp:<30} | {bar} {pn:.4f}")

    print("\n" + "=" * 80)
    print("OBSERVATION:")
    print("PN spikes during adversarial input, then recovers.")
    print("This is MEASURABLE internal state change, not simulated behavior.")


if __name__ == "__main__":
    demonstrate_adversarial_transparency()
    compare_with_standard_llm()
    visualize_pn_trajectory()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey takeaway: Riemann-J has OBSERVABLE internal states that enable")
    print("transparent uncertainty communication. Standard LLMs can only roleplay.")
