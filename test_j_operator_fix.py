#!/usr/bin/env python3
"""Test script to verify J-operator position embeddings fix."""
import sys

sys.path.insert(0, "src")

import time

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.pn_driver import PredictionErrorSignal


def test_j_operator():
    print("Initializing workspace...")
    workspace = CognitiveWorkspace()

    print("Creating high PN signal...")
    pn_signal = PredictionErrorSignal(
        timestamp=time.time(),
        source="Test",
        priority=10,
        p_n=0.95,
        uncompressed_data={"t": 25.0, "steps_since_zero": 500000},
    )

    print("Testing J-operator resolution...")
    try:
        state = workspace._j_operator_resolve(pn_signal)
        print(f"✓ SUCCESS: J-operator resolved")
        print(f"  Status: {state.status}")
        print(f'  Iterations: {state.analysis.get("iterations", "N/A")}')
        print(f'  Convergence: {state.analysis.get("convergence_type", "N/A")}')
        return True
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_j_operator()
    sys.exit(0 if success else 1)
