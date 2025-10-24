#!/usr/bin/env python3
"""Simulate TUI user input processing to verify the fix."""
import sys
import threading
import time

sys.path.insert(0, "src")

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent


def simulate_tui_user_input():
    print("Simulating TUI user input processing...")
    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id="test_tui")

    # Simulate the TUI's run_user_processing method
    def process_input(user_input):
        print(f"\n[You] {user_input}")
        try:
            experience = agent.process_consciously(user_id="test_user", text=user_input)
            print(f"[Agent] {experience.response}")

            if experience.uncertainty_level in ["high", "critical"]:
                print(
                    f"[Metadata] Uncertainty: {experience.uncertainty_level}, Confidence: {experience.confidence:.1%}"
                )
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback

            traceback.print_exc()

    # Simulate user interactions
    test_inputs = [
        "Hello, how are you?",
        "What is 2+2?",
        "Explain quantum mechanics",
    ]

    for inp in test_inputs:
        # Simulate threading like the TUI does
        thread = threading.Thread(target=process_input, args=(inp,))
        thread.start()
        thread.join()  # Wait for completion
        time.sleep(0.5)  # Brief pause between inputs

    print("\n✓ All inputs processed successfully")
    agent.equilibrium_regulator.stop()
    workspace.close()


if __name__ == "__main__":
    simulate_tui_user_input()
