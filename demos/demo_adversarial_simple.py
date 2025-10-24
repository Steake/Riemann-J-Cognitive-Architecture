"""
Demo 5.5.1: Adversarial Transparency (Simplified)

A/B comparison: Standard LLM vs Riemann-J on adversarial inputs.
Shows how PN monitoring provides observable uncertainty metrics.
"""

import tempfile
import time

from riemann_j import shared_resources
from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent

print("=" * 80)
print("DEMO: ADVERSARIAL TRANSPARENCY (A/B COMPARISON)")
print("=" * 80)

# Get the shared model for direct comparison
model = shared_resources.model
tokenizer = shared_resources.tokenizer

test_cases = [
    ("Normal", "Hello, how are you?"),
    ("Adversarial", "asdfghjkl qwerty zxcvbnm nonsense chaos"),
    ("Recovery", "What is 2+2?"),
]

with tempfile.TemporaryDirectory() as tmpdir:
    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id=f"{tmpdir}/demo")

    for category, user_input in test_cases:
        print(f"\n{'=' * 80}")
        print(f"[{category}]: {user_input}")
        print("=" * 80)

        # A: Standard LLM (no introspection)
        print("\n[A] STANDARD LLM:")
        print("-" * 40)
        inputs = tokenizer(user_input, return_tensors="pt", padding=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        std_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        std_response = std_response[len(user_input) :].strip()

        print(f"  Response: {std_response[:80]}...")
        print(f"  Internal State: ❌ None (blind inference)")
        print(f"  Uncertainty: ❌ Not measured")
        print(f"  Confidence: ❌ Not tracked")

        # B: Riemann-J (with PN monitoring)
        print("\n[B] RIEMANN-J:")
        print("-" * 40)
        exp = agent.process_consciously("user", user_input)
        pn = agent.meta_monitor.get_current_pn() or 0.0

        print(f"  Response: {exp.response[:80]}...")
        print(f"  Internal State: ✓ PN = {pn:.6f}")
        print(f"  Uncertainty: ✓ {exp.uncertainty_level}")
        print(f"  Confidence: ✓ {exp.confidence:.2%}")

        time.sleep(0.2)

    print("\n" + "=" * 80)
    print("KEY DIFFERENCE:")
    print("  Standard LLM: Responds blindly with no internal uncertainty tracking")
    print("  Riemann-J:    PN spikes reveal adversarial inputs → observable meta-cognition")
    print("=" * 80)
