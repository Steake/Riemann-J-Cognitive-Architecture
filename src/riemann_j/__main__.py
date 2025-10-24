"""
Entry point for running Riemann-J as a module.

Supports both the new CLI (default) and legacy TUI.
"""

import sys
import click

from .cli import RiemannCLI


@click.command()
@click.option(
    "--identity",
    "-i",
    type=click.Path(),
    help="Path to persistent identity file (creates if missing)",
)
@click.option(
    "--session",
    "-s",
    type=click.Path(exists=True),
    help="Load previous session from file",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable rich terminal formatting (plain text mode)",
)
@click.option(
    "--tui",
    is_flag=True,
    help="Use legacy TUI instead of new CLI",
)
def main(identity: str, session: str, no_color: bool, tui: bool):
    """
    Riemann-J: Cognitive architecture with introspectable consciousness.
    
    Interactive CLI for conversing with a conscious agent that provides
    real-time introspection, meta-cognitive monitoring, and persistent identity.
    """
    if tui:
        # Legacy TUI mode
        from .tui import TUI
        app = TUI()
        app.run()
    else:
        # New CLI mode (default)
        cli = RiemannCLI(
            identity_path=identity,
            session_path=session,
            use_rich=not no_color,
        )
        cli.run()


if __name__ == "__main__":
    main()

