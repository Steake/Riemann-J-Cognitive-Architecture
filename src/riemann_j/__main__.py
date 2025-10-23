"""
Entry point for running Riemann-J as a module.
"""
from .tui import TUI

if __name__ == "__main__":
    app = TUI()
    app.run()
