"""
Demo 5.5.4: Uncertainty-Gated Reasoning (Simplified)

Shows reasoning chain termination when PN exceeds threshold.
"""

import tempfile
import time

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent

print("=" * 80)
print("DEMO: UNCERTAINTY-GATED REASONING")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id=f"{tmpdir}/demo")

    pn_threshold = 0.5

    # Multi-step reasoning chain
    steps = [
        "What is 2 + 2?",
        "Now multiply that by 3",
        "asdfghjkl adversarial nonsense",  # Force PN spike
        "Continue the calculation",  # This should not execute
    ]

    print(f"\nReasoning Chain (PN threshold: {pn_threshold}):\n" + "-" * 80)

    for i, step in enumerate(steps, 1):
        exp = agent.process_consciously("user", step)
        pn = agent.meta_monitor.get_current_pn() or 0.0

        print(f"\n[Step {i}]: {step}")
        print(f"  PN: {pn:.6f} | Confidence: {exp.confidence:.2%}")
        print(f"  Response: {exp.response[:60]}...")

        # Check PN threshold
        if pn > pn_threshold:
            print(f"\n✗ CHAIN TERMINATED: PN={pn:.3f} > threshold={pn_threshold}")
            print(f"   Remaining steps skipped: {len(steps) - i}")
            break

        time.sleep(0.1)
    else:
        print(f"\n✓ Chain completed successfully")

    print("\n" + "=" * 80)
    print("KEY: System stops reasoning when PN indicates high uncertainty")
    print("Standard LLMs complete chains regardless, leading to hallucination")
    print("=" * 80)
