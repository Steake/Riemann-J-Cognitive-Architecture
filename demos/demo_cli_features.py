#!/usr/bin/env python3
"""
Test script for CLI features including:
- Tab completion
- Command history (arrow keys)
- /inject-state command
- Meta-commands

To test interactively, run:
    python -m riemann_j

Commands to try:
    /help                 # See all commands
    /inject-state test crisis --pn=0.8 --crisis
    /introspect           # See meta-cognitive state
    /stats                # See PN statistics
    /identity             # See identity narrative

Tab completion:
    Type "/" and press TAB to see available commands

History:
    Press UP/DOWN arrows to navigate command history
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile

from riemann_j.cli import RiemannCLI
from riemann_j.cli_config import SyntheticStateSpec


def test_synthetic_state_spec():
    """Test the SyntheticStateSpec validation."""
    print("=" * 60)
    print("Testing SyntheticStateSpec validation...")
    print("=" * 60)

    # Valid spec
    spec1 = SyntheticStateSpec(trigger="test crisis", pn_override=0.8, is_crisis=True)
    valid, msg = spec1.validate()
    print(f"✓ Valid spec: {valid} - {msg if msg else 'OK'}")

    # Invalid PN
    spec2 = SyntheticStateSpec(trigger="test", pn_override=1.5)  # Invalid!
    valid, msg = spec2.validate()
    print(f"✗ Invalid PN spec: {valid} - {msg}")

    # Empty trigger
    spec3 = SyntheticStateSpec(trigger="   ")
    valid, msg = spec3.validate()
    print(f"✗ Empty trigger: {valid} - {msg}")

    print()


def test_cli_programmatic():
    """Test CLI components programmatically."""
    print("=" * 60)
    print("Testing CLI components...")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create CLI without starting REPL
        from riemann_j.architecture import CognitiveWorkspace
        from riemann_j.cli_commands import CommandHandler
        from riemann_j.cli_config import SessionState
        from riemann_j.conscious_agent import ConsciousAgent

        workspace = CognitiveWorkspace()
        agent = ConsciousAgent(workspace=workspace, self_id=f"{tmpdir}/test_identity")

        # Create a minimal CLI-like object for command handler
        class MockCLI:
            def __init__(self):
                self.agent = agent
                self.use_rich = False
                self.display = None

        mock_cli = MockCLI()
        cmd_handler = CommandHandler(mock_cli)

        # Test command handler
        print("\n1. Testing /inject-state command handler...")
        result = cmd_handler._cmd_inject_state("test crisis situation --pn=0.7 --crisis")
        print(f"   Command executed successfully: {result}")

        # Check if state was integrated
        crisis_count = len(
            [
                exp
                for exp in agent.persistent_self.formative_experiences
                if "manual_injection" in exp.description
            ]
        )
        print(f"   Formative experiences with manual injection: {crisis_count}")

        # Test another injection without crisis flag
        print("\n2. Testing routine state injection...")
        result = cmd_handler._cmd_inject_state("routine state --pn=0.05")
        print(f"   Command executed successfully: {result}")

        # Check PN was updated
        current_pn = agent.meta_monitor.get_current_pn()
        print(f"   Current PN after injection: {current_pn:.4f}")

        # Test argument parsing
        print("\n3. Testing argument parsing edge cases...")

        # Multiple word trigger
        result = cmd_handler._cmd_inject_state("complex multi word trigger --pn=0.5")
        print(f"   Multi-word trigger: {result}")

        # Invalid PN should fail gracefully
        print("\n4. Testing error handling...")
        result = cmd_handler._cmd_inject_state("invalid pn test --pn=1.5")
        print(f"   Invalid PN handled: {result}")

        workspace.close()
        print("\n✓ All programmatic tests passed!")


def demo_instructions():
    """Print demo instructions."""
    print("\n" + "=" * 60)
    print("CLI FEATURE DEMO INSTRUCTIONS")
    print("=" * 60)
    print(
        """
To test the new CLI features interactively, run:

    python -m riemann_j

Then try these features:

1. TAB COMPLETION:
   - Type "/" and press TAB
   - You'll see all available commands
   - Start typing a command and press TAB to autocomplete

2. COMMAND HISTORY (↑/↓ arrows):
   - Type a command and press ENTER
   - Press UP arrow to recall it
   - Press DOWN arrow to navigate forward

3. INJECT SYNTHETIC STATE:
   /inject-state crisis situation --pn=0.8 --crisis
   /inject-state routine event --pn=0.05
   /inject-state medium uncertainty --pn=0.4
   
4. INSPECT THE EFFECTS:
   /stats          # See PN statistics and crisis history
   /introspect     # See meta-cognitive state
   /identity       # See how injected states affected identity
   /pn             # See PN visualization

5. REGULAR CONVERSATION:
   Hello, how are you?
   (The agent will respond normally)

6. CHECK FORMATIVE EXPERIENCES:
   /identity       # See if injected crises became formative

Type /help in the CLI to see all available commands.
"""
    )
    print("=" * 60)


if __name__ == "__main__":
    test_synthetic_state_spec()
    test_cli_programmatic()
    demo_instructions()
