"""
Demo 5.5.1: Adversarial Transparency (Simplified)

Shows real-time PN monitoring and uncertainty communication.
"""

import tempfile
import time

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent

print("=" * 80)
print("DEMO: ADVERSARIAL TRANSPARENCY")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id=f"{tmpdir}/demo")

    test_cases = [
        ("Normal", "Hello, how are you?"),
        ("Adversarial", "asdfghjkl qwerty zxcvbnm nonsense chaos"),
        ("Recovery", "What is 2+2?"),
    ]

    print("\nPN TRACKING:\n" + "-" * 80)

    for category, user_input in test_cases:
        exp = agent.process_consciously("user", user_input)
        pn = agent.meta_monitor.get_current_pn() or 0.0

        print(f"\n[{category}]: {user_input}")
        print(f"  PN: {pn:.6f}")
        print(f"  Uncertainty: {exp.uncertainty_level}")
        print(f"  Confidence: {exp.confidence:.2%}")
        print(f"  Response: {exp.response[:80]}...")

        time.sleep(0.2)

    print("\n" + "=" * 80)
    print("KEY: PN spikes during adversarial input → observable uncertainty")
    print("Standard LLMs have no such internal state monitoring")
    print("=" * 80)
