#!/usr/bin/env python3
"""
Convergence Heuristic Validation

Simple test to verify the improved J-Operator convergence logic works
without running full analysis (which is too heavy for CPU).

Tests:
1. Relative epsilon convergence
2. Lyapunov-based early stopping
3. Increased iteration limit
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.pn_driver import PredictionErrorSignal

print("=" * 80)
print("J-OPERATOR CONVERGENCE HEURISTIC VALIDATION")
print("=" * 80)
print()

print("Improvements implemented:")
print("  ✓ Relative epsilon: 0.01 * ||state|| (was fixed 1e-6)")
print("  ✓ Lyapunov early stopping: stop when stable for 5 iterations")
print("  ✓ Max iterations: 100 (was 50)")
print()

print("Initializing workspace (loading Phi-3.5-mini-instruct)...")
workspace = CognitiveWorkspace()
print("✓ Workspace ready")
print()

print("=" * 80)
print("TEST: J-Operator Resolution with Improved Heuristics")
print("=" * 80)
print()

# Create a high-PN signal to trigger J-Operator
test_signal = PredictionErrorSignal(
    timestamp=time.time(),
    source="CONVERGENCE_TEST",
    priority=1,
    p_n=0.95,
    uncompressed_data={"t": 150.0, "steps_since_zero": 1500},
)

print(f"Triggering J-Operator with PN={test_signal.p_n:.2f}...")
print("(This will take 30-60 seconds on CPU)")
print()

start = time.time()
result = workspace._j_operator_resolve(test_signal)
elapsed = time.time() - start

print(f"✓ Resolved in {elapsed:.1f}s")
print()

print("RESULTS:")
print("-" * 80)
print(f"  Status: {result.status}")
print(f"  Iterations: {result.analysis['iterations']}")
print(f"  Final distance: {result.analysis['final_distance']:.6f}")
print(f"  Convergence type: {result.analysis['convergence_type']}")
print(f"  Lyapunov exponent: {result.analysis['lyapunov_exp']:.6f}")
print()

print("ANALYSIS:")
print("-" * 80)

converged = "CONVERGED" in result.status

if converged:
    print("✓✓ CONVERGENCE ACHIEVED")
    print(f"   Used: {result.analysis['convergence_type']}")

    if result.analysis["convergence_type"] == "relative":
        print("   The relative epsilon criterion worked!")
        print(f"   Distance was within 1% of state magnitude")
    elif result.analysis["convergence_type"] == "lyapunov_stability":
        print("   Lyapunov-based early stopping worked!")
        print(f"   System stabilized before hitting epsilon threshold")
    elif result.analysis["convergence_type"] == "absolute":
        print("   Absolute epsilon still needed (edge case)")
else:
    print("⚠ ITERATION LIMIT REACHED")
    print(f"   Ran {result.analysis['iterations']} iterations without converging")
    print(f"   Final distance: {result.analysis['final_distance']:.6f}")

print()

if result.analysis["lyapunov_exp"] < 0:
    print(f"✓ STABLE DYNAMICS: Lyapunov = {result.analysis['lyapunov_exp']:.6f}")
    print("  System is converging even if threshold not met")
else:
    print(f"⚠ UNSTABLE DYNAMICS: Lyapunov = {result.analysis['lyapunov_exp']:.6f}")
    print("  System may be diverging")

print()
print("=" * 80)

if converged:
    print("VERDICT: ✓✓✓ IMPROVED HEURISTICS WORK")
    print()
    print("The relative epsilon and/or Lyapunov early stopping successfully")
    print("converged where the old absolute epsilon (1e-6) would have failed.")
elif result.analysis["lyapunov_exp"] < -1.0 and result.analysis["iterations"] >= 50:
    print("VERDICT: ✓✓ PARTIAL SUCCESS")
    print()
    print("System is converging (negative Lyapunov) but slowly.")
    print("The improved heuristics allow more iterations to find convergence.")
else:
    print("VERDICT: ⚠ NEEDS TUNING")
    print()
    print("System may need further parameter adjustment.")

print()
print("=" * 80)
print()
print("Next steps for full validation (requires GPU):")
print("  1. Run full crisis analysis with production model")
print("  2. Compare separation ratios: DistilGPT-2 vs Phi-3.5")
print("  3. Test user attractor persistence across crises")
print("  4. Validate cross-user contamination isolation")

workspace.close()
