"""
Interactive CLI for Riemann-J cognitive agent.

This module implements the core REPL loop for conversing with a conscious agent,
providing access to meta-cognitive introspection and persistent identity features.

Phase 1 Implementation: Basic REPL with ConsciousAgent integration.
Phase 2 Implementation: Rich terminal UI with display manager.
"""

import sys
from typing import Optional

from .architecture import CognitiveWorkspace
from .conscious_agent import ConsciousAgent, ConsciousExperience
from .cli_display import DisplayManager


class RiemannCLI:
    """Interactive CLI for Riemann-J cognitive agent."""

    def __init__(self, identity_path: Optional[str] = None, use_rich: bool = True):
        """
        Initialize CLI with optional persistent identity.

        Args:
            identity_path: Path to persistent identity directory (creates if missing)
            use_rich: Whether to use rich formatting (default: True)
        """
        self.workspace = CognitiveWorkspace()
        self.agent = ConsciousAgent(
            workspace=self.workspace,
            self_id=identity_path or "default_cli_identity",
        )
        self.running = False
        self.use_rich = use_rich
        self.display = DisplayManager() if use_rich else None

    def run(self) -> None:
        """Start interactive REPL loop."""
        # Display welcome banner
        if self.use_rich:
            self.display.render_welcome(self.agent.persistent_self.self_id)
        else:
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
            if self.use_rich:
                self.display.print_info("Goodbye!")
            else:
                print("Goodbye!")

        elif cmd == "/help":
            self._show_help()

        elif cmd.startswith("/introspect"):
            verbose = "brief" not in cmd
            report = self.agent.introspect(verbose=verbose)
            if self.use_rich:
                self.display.render_meta_state(report)
            else:
                print(f"\n{report}\n")

        elif cmd.startswith("/identity"):
            narrative = self.agent.get_formative_narrative()
            if self.use_rich:
                self.display.render_identity(narrative)
            else:
                print(f"\n{narrative}\n")

        elif cmd == "/stats":
            # Show PN statistics
            report = self.agent.meta_monitor.generate_self_report(verbose=True)
            if self.use_rich:
                self.display.render_meta_state(report)
            else:
                print(f"\n{report}\n")

        elif cmd == "/pn":
            # Show PN monitor visualization
            pn_history = self.agent.meta_monitor.pn_history
            current_pn = self.agent.meta_monitor.get_current_pn() or 0.0
            if self.use_rich:
                self.display.render_pn_monitor(pn_history, current_pn)
            else:
                print(f"\nCurrent PN: {current_pn:.4f}")
                print(f"History: {pn_history[-10:]}\n")

        else:
            if self.use_rich:
                self.display.print_error(f"Unknown command: {command}")
                self.display.print_info("Type /help for available commands")
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
        if self.use_rich:
            # Rich formatted display
            show_metadata = experience.uncertainty_level in ["high", "critical", "moderate"]
            self.display.render_response(experience, show_metadata=show_metadata)
        else:
            # Simple text display
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
        if self.use_rich:
            self.display.render_help()
        else:
            help_text = """
Available Commands:
  /help              - Show this help message
  /quit, /exit       - Exit the CLI
  /introspect        - Show detailed meta-cognitive state
  /introspect-brief  - Show concise meta-cognitive state  
  /identity          - Display persistent identity narrative
  /stats             - Show PN statistics and crisis history
  /pn                - Show PN monitor visualization

Regular messages are processed through the conscious agent.
"""
            print(help_text)


def main():
    """Entry point for CLI."""
    cli = RiemannCLI()
    cli.run()


if __name__ == "__main__":
    main()
