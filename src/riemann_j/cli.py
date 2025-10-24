"""
Interactive CLI for Riemann-J cognitive agent.

This module implements the core REPL loop for conversing with a conscious agent,
providing access to meta-cognitive introspection and persistent identity features.

Phase 1 Implementation: Basic REPL with ConsciousAgent integration.
Phase 2 Implementation: Rich terminal UI with display manager.
Phase 3 Implementation: Advanced input handling and commands.
"""

import sys
from typing import Optional

from .architecture import CognitiveWorkspace
from .conscious_agent import ConsciousAgent, ConsciousExperience
from .cli_display import DisplayManager
from .cli_input import InputHandler, InputType
from .cli_commands import CommandHandler
from .cli_config import SessionState


class RiemannCLI:
    """Interactive CLI for Riemann-J cognitive agent."""

    def __init__(self, identity_path: Optional[str] = None, use_rich: bool = True, session_path: Optional[str] = None):
        """
        Initialize CLI with optional persistent identity.

        Args:
            identity_path: Path to persistent identity directory (creates if missing)
            use_rich: Whether to use rich formatting (default: True)
            session_path: Path to session file to load
        """
        self.workspace = CognitiveWorkspace()
        self.agent = ConsciousAgent(
            workspace=self.workspace,
            self_id=identity_path or "default_cli_identity",
        )
        self.running = False
        self.use_rich = use_rich
        self.display = DisplayManager() if use_rich else None
        self.input_handler = InputHandler()
        self.command_handler = CommandHandler(self)
        
        # Session management
        self.session: Optional[SessionState] = None
        if session_path:
            try:
                self.session = SessionState.load(session_path)
            except Exception as e:
                print(f"Warning: Could not load session from {session_path}: {e}")
                self.session = SessionState(identity_path=str(self.agent.persistent_self.identity_file))
        else:
            self.session = SessionState(identity_path=str(self.agent.persistent_self.identity_file))

    def run(self) -> None:
        """Start interactive REPL loop."""
        # Display welcome banner
        if self.use_rich:
            self.display.render_welcome(str(self.agent.persistent_self.identity_file))
        else:
            print("=" * 60)
            print("Riemann-J Cognitive Agent - Interactive CLI")
            print("=" * 60)
            print("Type /help for available commands, /quit to exit")
            print()

        self.running = True
        while self.running:
            try:
                # Show multiline prompt if in multiline mode
                in_multiline, buffered_lines = self.input_handler.get_multiline_status()
                if in_multiline:
                    prompt = f"... ({buffered_lines} lines) > "
                else:
                    prompt = "You > "
                
                user_input = input(prompt).strip()
                if user_input or in_multiline:
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
        # Parse input
        input_type, content = self.input_handler.parse(user_input)
        
        # Handle based on type
        if input_type == InputType.EMPTY:
            return  # Ignore empty input
        
        elif input_type == InputType.COMMAND:
            # Handle command
            should_continue = self.command_handler.handle(content)
            if not should_continue:
                self.running = False
        
        elif input_type in [InputType.MESSAGE, InputType.MULTILINE]:
            # Validate input
            valid, error_msg = self.input_handler.validate(content)
            if not valid:
                if self.use_rich:
                    self.display.print_error(error_msg)
                else:
                    print(f"Error: {error_msg}")
                return
            
            # Process as regular conversation
            self._process_message(content)

    def _handle_command(self, command: str) -> None:
        """
        Handle meta-commands.
        
        DEPRECATED: Use command_handler.handle() instead.
        This method is kept for backwards compatibility with tests.
        """
        should_continue = self.command_handler.handle(command)
        if not should_continue:
            self.running = False

    def _process_message(self, text: str) -> None:
        """Process regular conversation message."""
        # Record user message in session
        if self.session:
            self.session.add_turn("user", text)
        
        # Use conscious agent to process input
        experience = self.agent.process_consciously(
            user_id="cli_user", text=text
        )

        # Record agent response in session
        if self.session:
            current_pn = self.agent.meta_monitor.get_current_pn()
            self.session.add_turn(
                "agent",
                experience.response,
                pn=current_pn,
                confidence=experience.confidence,
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
