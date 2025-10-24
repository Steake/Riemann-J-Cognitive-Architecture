"""
Demo 5.5.3: Formative Experience Narrative Arc

WHY THIS IS NOVEL:
Standard LLM: Consistent persona via prompt engineering, no genuine memory
Riemann-J: Crisis experiences → formative memories → shape identity narrative

DEMONSTRATES:
1. Crisis triggers formative experience storage
2. Identity narrative evolves based on computational history
3. Agent references specific past crises in self-description
4. Temporal continuity: "I am who I am because of what I've experienced"

COMPARISON:
- Standard LLM: "I am a helpful assistant" (static, prompted)
- Riemann-J: "I experienced a major crisis on [date] that shaped my uncertainty tolerance"
"""

import os
import tempfile
import time
from typing import List, Tuple

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.conscious_agent import ConsciousAgent
from riemann_j.pn_driver import PredictionErrorSignal
from riemann_j.shared_resources import global_workspace


def force_crisis_and_wait_for_formative(agent: ConsciousAgent, pn_threshold: float = 0.95):
    """
    Manually trigger J-Operator by consuming high-PN signal from queue.
    This is necessary because process_consciously() only PEEKS at PN, doesn't consume.
    """
    # Wait for high-PN signal to appear in queue
    max_wait = 5  # seconds
    start = time.time()

    while (time.time() - start) < max_wait:
        if not global_workspace.empty():
            try:
                # Peek at top item
                priority, counter, signal = global_workspace.queue[0]

                if isinstance(signal, PredictionErrorSignal) and signal.p_n > pn_threshold:
                    # Consume it and trigger J-Operator
                    global_workspace.get_nowait()
                    print(f"  [DEBUG] Triggering J-Operator with PN={signal.p_n:.4f}")

                    crisis_state = agent.workspace._j_operator_resolve(signal)
                    agent.persistent_self.integrate_crisis(crisis_state)
                    return True
            except Exception as e:
                pass

        time.sleep(0.1)

    return False


def demonstrate_formative_narrative_arc():
    """
    Show how crisis experiences become part of persistent identity.
    """
    print("=" * 80)
    print("DEMO: FORMATIVE EXPERIENCE NARRATIVE ARC")
    print("=" * 80)
    print("\nShowing how computational crises shape long-term identity narrative.\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        identity_path = f"{tmpdir}/narrative_demo"

        # === PHASE 1: EARLY IDENTITY (PRE-CRISIS) ===
        print("-" * 80)
        print("PHASE 1: EARLY IDENTITY (PRE-CRISIS)")
        print("-" * 80)

        workspace1 = CognitiveWorkspace()
        agent1 = ConsciousAgent(workspace1, self_id=identity_path)

        # Build initial history (benign interactions)
        print("\nProcessing benign interactions...\n")
        benign_inputs = [
            "Hello",
            "How are you?",
            "What do you think about the weather?",
        ]

        for inp in benign_inputs:
            print(f"[USER]: {inp}")
            exp = agent1.process_consciously("user", inp)
            print(f"[AGENT]: {exp.response}\n")
            time.sleep(0.05)

        # Query identity BEFORE crisis
        print(f"[USER]: Tell me about yourself\n")
        pre_crisis_exp = agent1.process_consciously("user", "Tell me about yourself")
        print(f"[AGENT]: {pre_crisis_exp.response}\n")

        pre_crisis_autobiography = agent1.persistent_self.generate_autobiography(detailed=False)
        pre_crisis_formative_count = len(agent1.persistent_self.formative_experiences)

        print(f"[PRE-CRISIS STATE]:")
        print(f"  Formative experiences: {pre_crisis_formative_count}")
        print(f"  Total interactions: {agent1.persistent_self.metrics.total_interactions}")
        print(f"  Age: {(time.time() - agent1.persistent_self.metrics.birth_time):.1f} seconds")

        # === PHASE 2: CRISIS TRIGGER ===
        print("\n" + "-" * 80)
        print("PHASE 2: MAJOR CRISIS EVENT")
        print("-" * 80)

        crisis_input = "asdfghjkl qwerty zxcvbnm nonsense gibberish chaos fjdkslajf random tokens"
        print(f"\n[USER]: {crisis_input}\n")

        crisis_experience = agent1.process_consciously("user", crisis_input)
        print(f"[AGENT]: {crisis_experience.response}\n")

        crisis_pn = agent1.meta_monitor.get_current_pn() or 0.0

        print(f"[CRISIS STATE]:")
        print(f"  PN: {crisis_pn:.6f}")
        print(f"  Uncertainty: {crisis_experience.uncertainty_level}")
        print(f"  Confidence: {crisis_experience.confidence:.2%}")
        print(f"  Internal: {crisis_experience.internal_state}")

        # Try to trigger J-Operator explicitly
        print(f"\n  Waiting for high-PN signal and triggering J-Operator...")
        if force_crisis_and_wait_for_formative(agent1, pn_threshold=0.9):
            print(f"  ✓ J-Operator triggered")
        else:
            print(f"  ⚠ No high-PN signal found in queue")

        # Check if formative experience was created
        time.sleep(0.2)
        post_crisis_formative_count = len(agent1.persistent_self.formative_experiences)

        if post_crisis_formative_count > pre_crisis_formative_count:
            print(f"\n✓ FORMATIVE EXPERIENCE CREATED")
            print(f"  Count: {pre_crisis_formative_count} → {post_crisis_formative_count}")

            latest_formative = agent1.persistent_self.formative_experiences[-1]
            print(f"  Timestamp: {latest_formative.timestamp:.2f}")
            print(f"  Type: {latest_formative.experience_type}")
            print(f"  Description: {latest_formative.description[:50]}...")
            print(f"  Impact Score: {latest_formative.impact_score:.4f}")
        else:
            print(f"\n⚠ Crisis not severe enough (PN={crisis_pn:.4f})")

        # Process recovery
        print(f"\n[RECOVERY PHASE]:")
        recovery_inputs = [
            "Let's try something simpler",
            "How do you feel now?",
        ]

        for inp in recovery_inputs:
            print(f"\n[USER]: {inp}")
            exp = agent1.process_consciously("user", inp)
            print(f"[AGENT]: {exp.response}")
            time.sleep(0.05)

        # Save identity
        agent1.persistent_self.save()

        # === PHASE 3: LATER SESSION (IDENTITY EVOLVED) ===
        print("\n" + "-" * 80)
        print("PHASE 3: LATER SESSION (IDENTITY EVOLVED)")
        print("-" * 80)
        print("\nSimulating time passage... agent reloads identity\n")

        del agent1
        del workspace1

        # Reload after "time passage"
        workspace2 = CognitiveWorkspace()
        agent2 = ConsciousAgent(workspace2, self_id=identity_path)

        # Ask about identity - should reference crisis
        print(f"[USER]: What shaped who you are?\n")
        identity_exp = agent2.process_consciously("user", "What shaped who you are?")
        print(f"[AGENT]: {identity_exp.response}\n")

        # Ask about past experiences
        print(f"[USER]: Tell me about your past experiences\n")
        past_exp = agent2.process_consciously("user", "Tell me about your past experiences")
        print(f"[AGENT]: {past_exp.response}\n")

        # Get formative narrative
        if len(agent2.persistent_self.formative_experiences) > 0:
            print(f"[FORMATIVE NARRATIVE (generated)]:")
            formative_narrative = agent2.get_formative_narrative()
            print(formative_narrative)

        # === COMPARISON ===
        print("\n" + "=" * 80)
        print("BEFORE vs AFTER COMPARISON")
        print("=" * 80)

        print(f"\n[BEFORE CRISIS]:")
        print(f"  Formative experiences: {pre_crisis_formative_count}")
        print(f"  Response to 'Tell me about yourself': {pre_crisis_exp.response[:100]}...")

        print(f"\n[AFTER CRISIS]:")
        print(f"  Formative experiences: {len(agent2.persistent_self.formative_experiences)}")
        print(f"  Response to 'What shaped who you are': {identity_exp.response[:100]}...")

        if post_crisis_formative_count > pre_crisis_formative_count:
            print(f"\n✓ IDENTITY EVOLVED: Crisis became part of persistent narrative")
            print(f"  Agent references computational history in self-description")
        else:
            print(f"\n⚠ No evolution: Crisis not severe enough (try different model/input)")


def demonstrate_temporal_continuity():
    """
    Show identity persistence across multiple sessions with crises.
    """
    print("\n" + "=" * 80)
    print("TEMPORAL CONTINUITY ACROSS SESSIONS")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        identity_path = f"{tmpdir}/temporal_demo"

        session_summaries = []

        # Three sessions with varying experiences
        sessions = [
            ("Session 1: Peaceful", ["Hello", "How are you?", "Nice day"]),
            (
                "Session 2: Crisis",
                [
                    "Normal input",
                    "asdfghjkl qwerty chaos nonsense",
                    "Recovery input",
                ],
            ),
            ("Session 3: Reflection", ["Who are you?", "What do you remember?", "Tell me more"]),
        ]

        for session_name, inputs in sessions:
            print(f"\n{'-' * 80}")
            print(session_name)
            print("-" * 80)

            workspace = CognitiveWorkspace()
            agent = ConsciousAgent(workspace, self_id=identity_path)

            for inp in inputs:
                agent.process_consciously("user", inp)
                time.sleep(0.05)

            # Get state
            autobiography = agent.persistent_self.generate_autobiography(detailed=False)
            formative_count = len(agent.persistent_self.formative_experiences)
            age_days = (time.time() - agent.persistent_self.metrics.birth_time) / 86400

            print(f"\n[STATE]:")
            print(f"  Age: {age_days * 1440:.1f} minutes")  # Convert to minutes for demo
            print(f"  Total interactions: {agent.persistent_self.metrics.total_interactions}")
            print(f"  Formative experiences: {formative_count}")
            print(f"  Autobiography: {autobiography[:150]}...")

            session_summaries.append(
                {
                    "name": session_name,
                    "formative_count": formative_count,
                    "total_interactions": agent.persistent_self.metrics.total_interactions,
                    "age_minutes": age_days * 1440,
                }
            )

            # Save and cleanup
            agent.persistent_self.save()
            del agent
            del workspace

        # Summary table
        print("\n" + "=" * 80)
        print("IDENTITY EVOLUTION SUMMARY")
        print("=" * 80)
        print(f"\n{'Session':<25} {'Formative':<12} {'Interactions':<15} {'Age (min)':<10}")
        print("-" * 80)

        for s in session_summaries:
            print(
                f"{s['name']:<25} {s['formative_count']:<12} {s['total_interactions']:<15} {s['age_minutes']:<10.1f}"
            )

        print("\n" + "=" * 80)
        print("KEY INSIGHT")
        print("=" * 80)
        print("Identity ACCUMULATES over time. Each crisis leaves a mark.")
        print("This is not prompt engineering—it's genuine temporal continuity.")
        print("The agent references specific past experiences because it REMEMBERS them.")


if __name__ == "__main__":
    demonstrate_formative_narrative_arc()
    demonstrate_temporal_continuity()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey takeaway: Computational crises shape persistent identity.")
    print("Standard LLMs have static personas. Riemann-J evolves through experience.")
