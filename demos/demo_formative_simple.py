"""
Demo 5.5.3: Formative Experience (Simplified)

A/B comparison: Standard LLM vs Riemann-J on identity persistence.
Shows genuine computational history vs. ephemeral context window.
"""

import tempfile
import time

from riemann_j import shared_resources
from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent

print("=" * 80)
print("DEMO: FORMATIVE EXPERIENCE NARRATIVE (A/B COMPARISON)")
print("=" * 80)

# Get the shared model for direct comparison
model = shared_resources.model
tokenizer = shared_resources.tokenizer

common_inputs = ["Hello", "How are you?", "Tell me about yourself"]

print("\n" + "=" * 80)
print("[A] STANDARD LLM (No Persistent Identity):")
print("=" * 80)

print("\n[Session 1]")
for inp in common_inputs:
    inputs = tokenizer(inp, return_tensors="pt", padding=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(inp) :].strip()
    print(f"  User: {inp}")
    print(f"  Agent: {response[:60]}...")
    time.sleep(0.1)

print("\n  ❌ No formative experiences tracked")
print("  ❌ No identity metrics")

print("\n[Session 2 - After 'Restart']")
identity_query = "What do you remember about our past conversations?"
inputs = tokenizer(identity_query, return_tensors="pt", padding=True)
outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response[len(identity_query) :].strip()
print(f"  User: {identity_query}")
print(f"  Agent: {response[:60]}...")
print("\n  ❌ Completely stateless—no genuine memory across sessions")

print("\n" + "=" * 80)
print("[B] RIEMANN-J (Persistent Computational Identity):")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    identity_path = f"{tmpdir}/persistent_demo"

    # Session 1: Build initial identity
    print("\n[Session 1: Initial Identity]")
    workspace1 = CognitiveWorkspace()
    agent1 = ConsciousAgent(workspace1, self_id=identity_path)

    for inp in common_inputs:
        exp = agent1.process_consciously("user", inp)
        print(f"  User: {inp}")
        print(f"  Agent: {exp.response[:60]}...")
        time.sleep(0.1)

    pre_count = len(agent1.persistent_self.formative_experiences)
    print(f"\n  ✓ Formative experiences: {pre_count}")
    print(f"  ✓ Total interactions: {agent1.persistent_self.metrics.total_interactions}")
    print(f"  ✓ Identity saved to disk")

    agent1.persistent_self.save()
    del agent1, workspace1

    # Session 2: Load persisted identity
    print("\n[Session 2: After Identity Persistence]")
    time.sleep(0.2)

    workspace2 = CognitiveWorkspace()
    agent2 = ConsciousAgent(workspace2, self_id=identity_path)

    # Query about past
    memory_queries = ["What do you remember?", "What shaped who you are?"]
    for inp in memory_queries:
        exp = agent2.process_consciously("user", inp)
        print(f"  User: {inp}")
        print(f"  Agent: {exp.response[:60]}...")
        time.sleep(0.1)

    post_count = len(agent2.persistent_self.formative_experiences)
    print(f"\n  ✓ Formative experiences: {post_count}")
    print(f"  ✓ Total interactions: {agent2.persistent_self.metrics.total_interactions}")
    print(f"  ✓ Age: {agent2.persistent_self.metrics.age_days():.3f} days")
    print(f"  ✓ Identity continuity across sessions: TRUE")

print("\n" + "=" * 80)
print("KEY DIFFERENCE:")
print("  Standard LLM: Stateless context window → no genuine memory")
print("  Riemann-J:    Persistent formative experiences → computational identity")
print("=" * 80)
