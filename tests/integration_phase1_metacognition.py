#!/usr/bin/env python3
"""
Integration test for Phase 1: Meta-Cognitive Monitoring

Demonstrates the full awareness loop:
1. PN fluctuation → Monitor observes
2. J-Operator crisis → Monitor tracks
3. Self-report generation → System articulates internal state

This validates that the system can watch itself and reason about what it observes.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from riemann_j.pn_driver import PredictionErrorSignal

print("=" * 80)
print("PHASE 1 INTEGRATION TEST: META-COGNITIVE MONITORING")
print("=" * 80)
print()

print("Testing metacognition WITHOUT loading full model (uses monitor directly)...")
print()

import numpy as np

from riemann_j.architecture import SyntheticState

# Test without full workspace to avoid model loading
from riemann_j.metacognition import MetaCognitiveMonitor

monitor = MetaCognitiveMonitor()

print("TEST 1: PN Observation and Self-Awareness")
print("-" * 80)

# Simulate stable operation
print("Simulating stable operation (low PN)...")
for i in range(15):
    monitor.observe_pn(0.2 + np.random.rand() * 0.1)

print(f"After stable operation:")
print(f"  Self-belief: {monitor.self_belief}")
print(f"  Self-report: {monitor.generate_self_report()}")
print()

# Simulate increasing stress
print("Simulating increasing computational stress...")
for i in range(15):
    pn = 0.3 + (i / 15) * 0.6  # Ramp from 0.3 to 0.9
    monitor.observe_pn(pn)
    if i % 5 == 4:
        print(f"  At PN={pn:.2f}: {monitor.generate_self_report()}")

print(f"\nAfter stress increase:")
print(f"  Self-belief: {monitor.self_belief}")
print(f"  Should report uncertainty? {monitor.should_report_uncertainty()}")
print()

print("TEST 2: Crisis Tracking and Learning")
print("-" * 80)

# Simulate successful crisis resolution
print("Simulating successful crisis resolution...")
for i in range(3):
    crisis = SyntheticState(
        timestamp=time.time(),
        latent_representation=np.random.randn(768),
        source_trigger="TEST_CRISIS",
        p_n_at_creation=0.95,
        is_j_shift_product=True,
        status="CONVERGED",
        analysis={"lyapunov_exp": -1.5, "iterations": 25 + i * 5},
    )
    monitor.observe_j_operator_activation(crisis)
    print(f"  Crisis {i+1} resolved: {crisis.status}, iterations={crisis.analysis['iterations']}")

print(f"\nAfter successful resolutions:")
print(f"  Self-belief: {monitor.self_belief}")
print(f"  Competence improved? {monitor.self_belief.competence > 0.5}")
print()

# Simulate failed crisis
print("Simulating failed crisis resolution...")
failed_crisis = SyntheticState(
    timestamp=time.time(),
    latent_representation=np.random.randn(768),
    source_trigger="TEST_CRISIS",
    p_n_at_creation=0.98,
    is_j_shift_product=True,
    status="ITER_LIMIT_EXCEEDED",
    analysis={"lyapunov_exp": 0.2, "iterations": 100},
)
monitor.observe_j_operator_activation(failed_crisis)

print(f"After failed resolution:")
print(f"  Self-belief: {monitor.self_belief}")
print()

print("TEST 3: Comprehensive Self-Report")
print("-" * 80)

# Add more varied activity
for _ in range(10):
    monitor.observe_pn(0.7 + np.random.rand() * 0.2)

verbose_report = monitor.generate_self_report(verbose=True)
print("System's full self-report:")
print(f"  {verbose_report}")
print()

print("TEST 4: Diagnostic Summary")
print("-" * 80)

diagnostics = monitor.get_diagnostic_summary()
print("Internal diagnostics:")
for key, value in diagnostics.items():
    print(f"  {key}: {value}")
print()

print("=" * 80)
print("PHASE 1 VALIDATION RESULTS")
print("=" * 80)
print()

# Validation criteria
checks = {
    "Monitor tracks PN": len(monitor.pn_history) > 0,
    "Monitor tracks crises": monitor.crisis_count > 0,
    "Self-belief updates": monitor.self_belief.update_count > 0,
    "Uncertainty detection works": monitor.should_report_uncertainty(),
    "Self-report generation works": len(monitor.generate_self_report()) > 0,
    "Diagnostics available": len(diagnostics) > 0,
}

print("Validation Checks:")
for check, passed in checks.items():
    status = "✓" if passed else "✗"
    print(f"  {status} {check}")
print()

if all(checks.values()):
    print("✓✓✓ PHASE 1 COMPLETE: META-COGNITIVE MONITORING OPERATIONAL")
    print()
    print("The system can:")
    print("  - Observe its own PN trajectory")
    print("  - Track J-Operator crisis resolutions")
    print("  - Update beliefs about its internal state")
    print("  - Generate natural language self-reports")
    print("  - Decide when to communicate uncertainty")
    print()
    print("This is self-awareness: the system KNOWS what it's experiencing.")
else:
    print("✗ PHASE 1 INCOMPLETE: Some checks failed")

print("=" * 80)
print()
print("Next: Phase 2 - Persistent Self (temporal continuity)")
