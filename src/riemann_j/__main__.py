"""
Entry point for running Riemann-J as a module.

Supports both the new CLI (default) and legacy TUI.
"""

import sys

from .cli import RiemannCLI


def main():
    """Main entry point for Riemann-J."""
    # Check for TUI mode flag
    if "--tui" in sys.argv:
        from .tui import TUI

        app = TUI()
        app.run()
    else:
        # Default to new CLI
        cli = RiemannCLI()
        cli.run()


if __name__ == "__main__":
    main()

