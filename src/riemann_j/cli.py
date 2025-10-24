"""
Interactive CLI for Riemann-J cognitive agent.

This module implements the core REPL loop for conversing with a conscious agent,
providing access to meta-cognitive introspection and persistent identity features.

Phase 1 Implementation: Basic REPL with ConsciousAgent integration.
"""

import sys
from typing import Optional

from .architecture import CognitiveWorkspace
from .conscious_agent import ConsciousAgent, ConsciousExperience


class RiemannCLI:
    """Interactive CLI for Riemann-J cognitive agent."""

    def __init__(self, identity_path: Optional[str] = None):
        """
        Initialize CLI with optional persistent identity.

        Args:
            identity_path: Path to persistent identity directory (creates if missing)
        """
        self.workspace = CognitiveWorkspace()
        self.agent = ConsciousAgent(
            workspace=self.workspace,
            self_id=identity_path or "default_cli_identity",
        )
        self.running = False

    def run(self) -> None:
        """Start interactive REPL loop."""
        print("=" * 60)
        print("Riemann-J Cognitive Agent - Interactive CLI")
        print("=" * 60)
        print("Type /help for available commands, /quit to exit")
        print()

        self.running = True
        while self.running:
            try:
                user_input = input("You > ").strip()
                if user_input:
                    self.handle_input(user_input)
            except (KeyboardInterrupt, EOFError):
                print("\n\nGraceful shutdown...")
                self.running = False
                break

        # Cleanup
        self.workspace.close()

    def handle_input(self, user_input: str) -> None:
        """
        Process user input (message or command).

        Args:
            user_input: User's text input
        """
        # Check if it's a command
        if user_input.startswith("/"):
            self._handle_command(user_input)
        else:
            # Process as regular conversation
            self._process_message(user_input)

    def _handle_command(self, command: str) -> None:
        """Handle meta-commands."""
        cmd = command.lower().strip()

        if cmd == "/quit" or cmd == "/exit":
            self.running = False
            print("Goodbye!")

        elif cmd == "/help":
            self._show_help()

        elif cmd.startswith("/introspect"):
            verbose = "brief" not in cmd
            report = self.agent.introspect(verbose=verbose)
            print(f"\n{report}\n")

        elif cmd.startswith("/identity"):
            narrative = self.agent.get_formative_narrative()
            print(f"\n{narrative}\n")

        elif cmd == "/stats":
            # Show PN statistics
            report = self.agent.meta_monitor.generate_self_report(verbose=True)
            print(f"\n{report}\n")

        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands")

    def _process_message(self, text: str) -> None:
        """Process regular conversation message."""
        # Use conscious agent to process input
        experience = self.agent.process_consciously(
            user_id="cli_user", text=text
        )

        # Display response
        self.display_response(experience)

    def display_response(self, experience: ConsciousExperience) -> None:
        """
        Display agent response with introspection.

        Args:
            experience: The conscious experience to display
        """
        # Simple text display for Phase 1
        print(f"Agent > {experience.response}")

        # Show metadata if uncertainty is high
        if experience.uncertainty_level in ["high", "critical"]:
            print(
                f"        [Uncertainty: {experience.uncertainty_level}, "
                f"Confidence: {experience.confidence:.1%}, "
                f"PN: {self.agent.meta_monitor.get_current_pn():.3f}]"
            )

    def _show_help(self) -> None:
        """Display available commands."""
        help_text = """
Available Commands:
  /help              - Show this help message
  /quit, /exit       - Exit the CLI
  /introspect        - Show detailed meta-cognitive state
  /introspect-brief  - Show concise meta-cognitive state  
  /identity          - Display persistent identity narrative
  /stats             - Show PN statistics and crisis history

Regular messages are processed through the conscious agent.
"""
        print(help_text)


def main():
    """Entry point for CLI."""
    cli = RiemannCLI()
    cli.run()


if __name__ == "__main__":
    main()
