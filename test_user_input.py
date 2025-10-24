#!/usr/bin/env python3
"""Test user input processing directly."""
import sys

sys.path.insert(0, "src")

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent


def test_user_input():
    print("Initializing workspace and agent...")
    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id="test_user")

    test_inputs = [
        "Hello, how are you?",
        "Explain why",
        "What is 2+2?",
    ]

    print("\nTesting user input processing:")
    for user_input in test_inputs:
        print(f"\n{'='*60}")
        print(f"Input: {user_input}")
        print(f"{'='*60}")

        try:
            experience = agent.process_consciously(user_id="test", text=user_input)
            print(f"Response: {experience.response}")
            print(f"Uncertainty: {experience.uncertainty_level}")
            print(f"Confidence: {experience.confidence:.2%}")
            if experience.reflection:
                print(f"Reflection: {experience.reflection}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()

    agent.equilibrium_regulator.stop()
    workspace.close()


if __name__ == "__main__":
    test_user_input()
