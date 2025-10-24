#!/usr/bin/env python3
"""Test script to verify multiple J-operator resolutions work correctly."""
import sys

sys.path.insert(0, "src")

import time

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.pn_driver import PredictionErrorSignal


def test_multiple_resolutions():
    print("Initializing workspace...")
    workspace = CognitiveWorkspace()

    # Test with different PN levels
    test_cases = [
        (0.85, 10000),
        (0.90, 50000),
        (0.95, 500000),
        (0.98, 800000),
        (0.99, 1000000),
    ]

    results = {"converged": 0, "failed": 0}

    for pn_value, steps in test_cases:
        print(f"\nTesting PN={pn_value:.2f}, steps={steps}...")
        pn_signal = PredictionErrorSignal(
            timestamp=time.time(),
            source="Test",
            priority=int((1 - pn_value) * 100),
            p_n=pn_value,
            uncompressed_data={"t": 25.0, "steps_since_zero": steps},
        )

        try:
            state = workspace._j_operator_resolve(pn_signal)
            if "CONVERGED" in state.status:
                results["converged"] += 1
                print(f"  ✓ {state.status} in {state.analysis['iterations']} iterations")
                print(
                    f"    Convergence: {state.analysis['convergence_type']}, distance={state.analysis['final_distance']:.4f}"
                )
            else:
                results["failed"] += 1
                print(f"  ✗ {state.status}")
        except Exception as e:
            results["failed"] += 1
            print(f"  ✗ EXCEPTION: {e}")

    print(f"\n{'='*60}")
    print(
        f"RESULTS: {results['converged']}/{len(test_cases)} converged, {results['failed']}/{len(test_cases)} failed"
    )
    print(f"Success rate: {results['converged']/len(test_cases)*100:.1f}%")
    print(f"{'='*60}")

    return results["converged"] == len(test_cases)


if __name__ == "__main__":
    success = test_multiple_resolutions()
    sys.exit(0 if success else 1)
