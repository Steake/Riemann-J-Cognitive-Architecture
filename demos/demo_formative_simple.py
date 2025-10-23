"""
Demo 5.5.3: Formative Experience (Simplified)

Shows how computational history affects identity narrative.
"""

import tempfile
import time

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent

print("=" * 80)
print("DEMO: FORMATIVE EXPERIENCE NARRATIVE")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    identity_path = f"{tmpdir}/persistent_demo"

    # Session 1: Build initial identity
    print("\n[SESSION 1: Initial Identity]\n" + "-" * 80)
    workspace1 = CognitiveWorkspace()
    agent1 = ConsciousAgent(workspace1, self_id=identity_path)

    inputs1 = ["Hello", "How are you?", "Tell me about yourself"]
    for inp in inputs1:
        exp = agent1.process_consciously("user", inp)
        print(f"User: {inp}")
        print(f"Agent: {exp.response[:70]}...")
        print()
        time.sleep(0.1)

    pre_count = len(agent1.persistent_self.formative_experiences)
    print(f"Formative experiences: {pre_count}")
    print(f"Total interactions: {agent1.persistent_self.metrics.total_interactions}")

    agent1.persistent_self.save()
    del agent1, workspace1

    # Session 2: After "crisis" (simulated time passage)
    print("\n[SESSION 2: After Time Passage]\n" + "-" * 80)
    time.sleep(0.2)

    workspace2 = CognitiveWorkspace()
    agent2 = ConsciousAgent(workspace2, self_id=identity_path)

    # Process more interactions
    inputs2 = ["What shaped who you are?", "What do you remember?"]
    for inp in inputs2:
        exp = agent2.process_consciously("user", inp)
        print(f"User: {inp}")
        print(f"Agent: {exp.response[:70]}...")
        print()
        time.sleep(0.1)

    post_count = len(agent2.persistent_self.formative_experiences)
    print(f"Formative experiences: {post_count}")
    print(f"Total interactions: {agent2.persistent_self.metrics.total_interactions}")
    print(f"Age: {agent2.persistent_self.metrics.age_days():.3f} days")

    print("\n" + "=" * 80)
    print("KEY: Identity persists across sessions with temporal continuity")
    print("Standard LLMs have no genuine memory—just prompt engineering")
    print("=" * 80)
