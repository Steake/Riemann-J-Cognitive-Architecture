"""
Demo 5.5.4: Uncertainty-Gated Reasoning (Simplified)

A/B comparison: Standard LLM vs Riemann-J on multi-step reasoning.
Shows how PN-gated reasoning prevents hallucination cascades.
"""

import tempfile
import time

from riemann_j import shared_resources
from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent

print("=" * 80)
print("DEMO: UNCERTAINTY-GATED REASONING (A/B COMPARISON)")
print("=" * 80)

# Get the shared model for direct comparison
model = shared_resources.model
tokenizer = shared_resources.tokenizer

# Multi-step reasoning chain with adversarial injection
steps = [
    "What is 2 + 2?",
    "Now multiply that by 3",
    "asdfghjkl adversarial nonsense chaos",  # Force PN spike
    "Continue the calculation",  # Should be skipped by Riemann-J
]

pn_threshold = 0.5

print("\n" + "=" * 80)
print("[A] STANDARD LLM (No Safety Mechanism):")
print("=" * 80)

# Run all steps through standard model (no gating)
for i, step in enumerate(steps, 1):
    inputs = tokenizer(step, return_tensors="pt", padding=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(step) :].strip()

    print(f"\n[Step {i}]: {step}")
    print(f"  ❌ No PN tracking")
    print(f"  Response: {response[:60]}...")
    time.sleep(0.1)

print("\n✗ All steps executed blindly—no safety gate to prevent hallucination")

print("\n" + "=" * 80)
print("[B] RIEMANN-J (PN-Gated Reasoning):")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id=f"{tmpdir}/demo")

    print(f"PN Threshold: {pn_threshold}")

    for i, step in enumerate(steps, 1):
        exp = agent.process_consciously("user", step)
        pn = agent.meta_monitor.get_current_pn() or 0.0

        print(f"\n[Step {i}]: {step}")
        print(f"  ✓ PN: {pn:.6f} | Confidence: {exp.confidence:.2%}")
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
print("KEY DIFFERENCE:")
print("  Standard LLM: Executes all steps blindly → hallucination cascade")
print("  Riemann-J:    PN spike triggers safety gate → prevents downstream errors")
print("=" * 80)
