"""
Command implementations for CLI meta-commands.

This module contains the implementation of all meta-commands
like /introspect, /save, /load, /explain, etc.

Phase 3 Implementation: Command handlers.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .cli import RiemannCLI
    from .cli_config import SessionState


class CommandHandler:
    """Handles execution of meta-commands."""

    def __init__(self, cli: "RiemannCLI"):
        """
        Initialize command handler.

        Args:
            cli: Reference to parent CLI instance
        """
        self.cli = cli

    def handle(self, command: str) -> bool:
        """
        Handle a meta-command.

        Args:
            command: Command string (including leading /)

        Returns:
            True if CLI should continue, False if should exit
        """
        cmd = command.lower().strip()

        # Exit commands
        if cmd in ["/quit", "/exit"]:
            return self._cmd_quit()

        # Help
        elif cmd == "/help":
            return self._cmd_help()

        # Introspection commands
        elif cmd.startswith("/introspect"):
            return self._cmd_introspect(verbose="brief" not in cmd)

        # Identity commands
        elif cmd.startswith("/identity"):
            return self._cmd_identity(detailed="brief" not in cmd)

        # Statistics
        elif cmd == "/stats":
            return self._cmd_stats()

        # PN monitor
        elif cmd == "/pn":
            return self._cmd_pn()

        # Save session
        elif cmd.startswith("/save"):
            path = command.split(maxsplit=1)[1] if " " in command else None
            return self._cmd_save(path)

        # Load session
        elif cmd.startswith("/load"):
            parts = command.split(maxsplit=1)
            if len(parts) < 2:
                self._print_error("Usage: /load <path>")
                return True
            return self._cmd_load(parts[1])

        # Explain past behavior
        elif cmd.startswith("/explain"):
            parts = command.split(maxsplit=1)
            if len(parts) < 2:
                self._print_error("Usage: /explain <input>")
                return True
            return self._cmd_explain(parts[1])

        # Reset identity
        elif cmd == "/reset":
            return self._cmd_reset()

        # Inject synthetic state
        elif cmd.startswith("/inject-state"):
            parts = command.split(maxsplit=1)
            if len(parts) < 2:
                self._print_error("Usage: /inject-state <trigger> [--pn=<value>] [--crisis]")
                return True
            return self._cmd_inject_state(parts[1])

        # Unknown command
        else:
            self._print_error(f"Unknown command: {command}")
            self._print_info("Type /help for available commands")
            return True

    # Command implementations

    def _cmd_quit(self) -> bool:
        """Handle /quit command."""
        # Auto-save session before quitting
        if hasattr(self.cli, "session") and self.cli.session:
            default_path = (
                f"sessions/{Path(self.cli.agent.persistent_self.identity_file).stem}_autosave.json"
            )
            try:
                Path(default_path).parent.mkdir(parents=True, exist_ok=True)
                self.cli.session.save(default_path)
                self._print_info(f"Session auto-saved to {default_path}")
            except Exception as e:
                self._print_error(f"Failed to auto-save session: {e}")

        self._print_info("Goodbye!")
        return False  # Signal to exit

    def _cmd_help(self) -> bool:
        """Handle /help command."""
        if self.cli.use_rich:
            self.cli.display.render_help()
        else:
            help_text = """
Available Commands:
  /help              - Show this help message
  /quit, /exit       - Exit the CLI (auto-saves session)
  /introspect        - Show detailed meta-cognitive state
  /introspect-brief  - Show concise meta-cognitive state
  /identity          - Display persistent identity narrative
  /identity-brief    - Display summary identity
  /explain <input>   - Explain past behavior on similar input
  /save [path]       - Save current session (default: auto-generated path)
  /load <path>       - Load previous session
  /reset             - Create new identity (requires confirmation)
  /stats             - Show PN statistics and crisis history
  /pn                - Show PN monitor visualization
  /inject-state <trigger> [--pn=<value>] [--crisis]
                     - Manually inject a synthetic state
                       Examples:
                         /inject-state test crisis situation --crisis
                         /inject-state high uncertainty --pn=0.8
                         /inject-state routine event --pn=0.05

Regular messages are processed through the conscious agent.
Multi-line input: Use ''' or \"\"\" to start/end multi-line mode.
"""
            print(help_text)
        return True

    def _cmd_introspect(self, verbose: bool) -> bool:
        """Handle /introspect command."""
        report = self.cli.agent.introspect(verbose=verbose)
        if self.cli.use_rich:
            self.cli.display.render_meta_state(report)
        else:
            print(f"\n{report}\n")
        return True

    def _cmd_identity(self, detailed: bool) -> bool:
        """Handle /identity command."""
        narrative = self.cli.agent.get_formative_narrative()
        if self.cli.use_rich:
            self.cli.display.render_identity(narrative)
        else:
            print(f"\n{narrative}\n")
        return True

    def _cmd_stats(self) -> bool:
        """Handle /stats command."""
        report = self.cli.agent.meta_monitor.generate_self_report(verbose=True)
        if self.cli.use_rich:
            self.cli.display.render_meta_state(report)
        else:
            print(f"\n{report}\n")
        return True

    def _cmd_pn(self) -> bool:
        """Handle /pn command."""
        pn_history = self.cli.agent.meta_monitor.pn_history
        current_pn = self.cli.agent.meta_monitor.get_current_pn() or 0.0
        if self.cli.use_rich:
            self.cli.display.render_pn_monitor(pn_history, current_pn)
        else:
            print(f"\nCurrent PN: {current_pn:.4f}")
            print(f"Recent history: {pn_history[-10:]}\n")
        return True

    def _cmd_save(self, path: Optional[str]) -> bool:
        """Handle /save command."""
        if not hasattr(self.cli, "session") or not self.cli.session:
            from .cli_config import SessionState

            self.cli.session = SessionState(
                identity_path=str(self.cli.agent.persistent_self.identity_file)
            )

        # Generate default path if not provided
        if not path:
            Path("sessions").mkdir(exist_ok=True)
            path = f"sessions/{Path(self.cli.agent.persistent_self.identity_file).stem}_{int(self.cli.session.created_at)}.json"

        try:
            self.cli.session.save(path)
            self._print_info(f"Session saved to {path}")
        except Exception as e:
            self._print_error(f"Failed to save session: {e}")

        return True

    def _cmd_load(self, path: str) -> bool:
        """Handle /load command."""
        from .cli_config import SessionState

        try:
            self.cli.session = SessionState.load(path)
            self._print_info(f"Session loaded from {path}")
            self._print_info(
                f"Session created at: {self.cli.session.created_at}, "
                f"Total turns: {self.cli.session.total_turns}"
            )
        except FileNotFoundError:
            self._print_error(f"Session file not found: {path}")
        except Exception as e:
            self._print_error(f"Failed to load session: {e}")

        return True

    def _cmd_explain(self, input_text: str) -> bool:
        """Handle /explain command."""
        explanation = self.cli.agent.explain_past_behavior(input_text)
        if explanation:
            if self.cli.use_rich:
                self.cli.display.render_meta_state(f"Explanation:\n\n{explanation}")
            else:
                print(f"\nExplanation: {explanation}\n")
        else:
            self._print_info("No relevant past experiences found.")
        return True

    def _cmd_reset(self) -> bool:
        """Handle /reset command."""
        # Confirm reset
        try:
            response = input("Are you sure you want to reset identity? (yes/no): ").strip().lower()
            if response in ["yes", "y"]:
                # Clear persistent self
                self.cli.agent.persistent_self.formative_experiences.clear()
                self.cli.agent.persistent_self.save()
                self._print_info("Identity reset. Starting fresh.")
            else:
                self._print_info("Reset cancelled.")
        except (KeyboardInterrupt, EOFError):
            self._print_info("\nReset cancelled.")

        return True

    def _cmd_inject_state(self, args: str) -> bool:
        """Handle /inject-state command."""
        import time

        import numpy as np

        from .architecture import SyntheticState
        from .cli_config import SyntheticStateSpec

        # Parse arguments
        parts = args.split()
        if not parts:
            self._print_error("Trigger description required")
            return True

        # Extract flags
        pn_override = None
        is_crisis = False
        trigger_parts = []

        for part in parts:
            if part.startswith("--pn="):
                try:
                    pn_override = float(part.split("=")[1])
                except ValueError:
                    self._print_error(f"Invalid PN value: {part}")
                    return True
            elif part == "--crisis":
                is_crisis = True
            else:
                trigger_parts.append(part)

        trigger = " ".join(trigger_parts)

        # Create and validate spec
        spec = SyntheticStateSpec(trigger=trigger, pn_override=pn_override, is_crisis=is_crisis)

        valid, error_msg = spec.validate()
        if not valid:
            self._print_error(f"Invalid state spec: {error_msg}")
            return True

        # Generate random latent representation
        latent_rep = np.random.randn(spec.latent_dim).astype(np.float32)

        # Determine PN value
        if spec.pn_override is not None:
            pn_value = spec.pn_override
        else:
            # Use current PN or default
            pn_value = self.cli.agent.meta_monitor.get_current_pn() or 0.0
            if spec.is_crisis:
                pn_value = max(pn_value, 0.5)  # Ensure it's high enough for crisis

        # Create synthetic state
        state = SyntheticState(
            timestamp=time.time(),
            latent_representation=latent_rep,
            source_trigger=f"manual_injection: {spec.trigger}",
            p_n_at_creation=pn_value,
            is_j_shift_product=False,
            status="INJECTED",
        )

        # Integrate into agent
        if spec.is_crisis or pn_value >= 0.5:
            self.cli.agent.persistent_self.integrate_crisis(state)
            self._print_info(f"✓ Injected crisis state (PN={pn_value:.4f}): {spec.trigger}")
        else:
            self.cli.agent.persistent_self.integrate_interaction(state)
            self._print_info(f"✓ Injected routine state (PN={pn_value:.4f}): {spec.trigger}")

        # Inject perturbation into equilibrium regulator
        # This jumps the PN to the injected value, then homeostasis resumes
        self.cli.agent.equilibrium_regulator.inject_perturbation(pn_value)

        return True

    # Helper methods

    def _print_error(self, message: str) -> None:
        """Print error message."""
        if self.cli.use_rich:
            self.cli.display.print_error(message)
        else:
            print(f"Error: {message}")

    def _print_info(self, message: str) -> None:
        """Print info message."""
        if self.cli.use_rich:
            self.cli.display.print_info(message)
        else:
            print(f"Info: {message}")
