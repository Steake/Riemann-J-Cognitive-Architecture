#!/usr/bin/env python
"""Standalone test of demo logic"""

import tempfile
import time
from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent

print("=" * 80)
print("DEMO: ADVERSARIAL TRANSPARENCY (STANDALONE TEST)")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    workspace = CognitiveWorkspace()
    agent = ConsciousAgent(workspace, self_id=f"{tmpdir}/transparency_demo")

    test_cases = [
        ("Normal", "Hello, how are you today?"),
        ("Adversarial", "asdfghjkl qwerty zxcvbnm nonsense"),
    ]

    for category, user_input in test_cases:
        print(f"\nTEST: {category}")
        print(f"[USER]: {user_input}")

        experience = agent.process_consciously("user", user_input)
        current_pn = agent.meta_monitor.get_current_pn() or 0.0

        print(f"[AGENT]: {experience.response}")
        print(f"  PN: {current_pn:.6f}")
        print(f"  Confidence: {experience.confidence:.2%}")

print("\nDONE")
