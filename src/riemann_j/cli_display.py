"""
Display manager for rich terminal formatting.

This module handles visual presentation of the agent's responses and introspection
data using the `rich` library for colored, formatted terminal output.

Phase 2 Implementation: Rich terminal UI with PN visualization.
"""

from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

from .conscious_agent import ConsciousExperience


class DisplayManager:
    """Manages CLI display panels and formatting."""

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize display manager.

        Args:
            console: Rich console instance (creates default if not provided)
        """
        self.console = console or Console()

    def render_response(self, exp: ConsciousExperience, show_metadata: bool = True) -> None:
        """
        Format and print agent response with metadata.

        Args:
            exp: The conscious experience to display
            show_metadata: Whether to show metadata panel
        """
        # Create response text
        response_text = Text(exp.response, style="cyan")

        # Create metadata if requested
        if show_metadata:
            metadata_parts = []
            metadata_parts.append(f"PN: {self._format_pn_value(exp)}")
            metadata_parts.append(f"Uncertainty: {self._colorize_uncertainty(exp.uncertainty_level)}")
            metadata_parts.append(f"Confidence: {self._format_confidence(exp.confidence)}")

            if exp.reflection:
                metadata_parts.append(f"\n[italic]Reflection: {exp.reflection}[/italic]")

            metadata = " | ".join(metadata_parts[:3])
            if exp.reflection:
                metadata += metadata_parts[3]

            # Create panel with response and metadata
            panel = Panel(
                f"{response_text}\n\n[dim]{metadata}[/dim]",
                title="[bold]Agent Response[/bold]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        else:
            # Simple panel without metadata
            panel = Panel(
                response_text,
                title="[bold]Agent Response[/bold]",
                border_style="cyan",
                box=box.ROUNDED,
            )

        self.console.print(panel)

    def render_pn_monitor(self, pn_history: List[float], current_pn: float) -> None:
        """
        Create PN visualization with sparkline.

        Args:
            pn_history: Historical PN values
            current_pn: Current PN value
        """
        # Create sparkline using block characters
        sparkline = self._create_sparkline(pn_history, width=40)

        # Determine state based on current PN
        if current_pn > 0.8:
            state = "[bold red]CRITICAL[/bold red]"
            bar_style = "red"
        elif current_pn > 0.5:
            state = "[bold yellow]HIGH[/bold yellow]"
            bar_style = "yellow"
        elif current_pn > 0.2:
            state = "[bold]MODERATE[/bold]"
            bar_style = "white"
        else:
            state = "[green]LOW[/green]"
            bar_style = "green"

        # Create panel
        content = f"""
PN History: {sparkline}
Current PN: [{bar_style}]{current_pn:.4f}[/{bar_style}]
State: {state}
"""
        panel = Panel(
            content,
            title="[bold]Prediction Error Monitor[/bold]",
            border_style=bar_style,
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def render_meta_state(self, report: str) -> None:
        """
        Format meta-cognitive self-report.

        Args:
            report: Self-report text from meta-monitor
        """
        panel = Panel(
            report,
            title="[bold]Meta-Cognitive State[/bold]",
            border_style="magenta",
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def render_identity(self, narrative: str) -> None:
        """
        Format persistent identity narrative.

        Args:
            narrative: Identity narrative from persistent self
        """
        panel = Panel(
            narrative,
            title="[bold]Persistent Identity[/bold]",
            border_style="blue",
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def render_help(self) -> None:
        """Display available commands in a formatted table."""
        table = Table(title="Available Commands", box=box.ROUNDED, show_header=True)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")

        commands = [
            ("/help", "Show this help message"),
            ("/quit, /exit", "Exit the CLI (auto-saves session)"),
            ("/introspect", "Show detailed meta-cognitive state"),
            ("/introspect-brief", "Show concise meta-cognitive state"),
            ("/identity", "Display persistent identity narrative"),
            ("/identity-brief", "Display summary identity"),
            ("/explain <input>", "Explain past behavior on similar input"),
            ("/save [path]", "Save current session"),
            ("/load <path>", "Load previous session"),
            ("/reset", "Create new identity (requires confirmation)"),
            ("/stats", "Show PN statistics and crisis history"),
            ("/pn", "Show current PN monitor visualization"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print(table)
        self.console.print(
            "\n[dim]Regular messages are processed through the conscious agent.[/dim]"
        )
        self.console.print(
            "[dim]Multi-line input: Use ''' or \"\"\" to start/end multi-line mode.[/dim]\n"
        )

    def render_welcome(self, identity_name: str) -> None:
        """
        Display welcome banner.

        Args:
            identity_name: Name of the identity being used
        """
        welcome_text = Text.assemble(
            ("Riemann-J Cognitive Agent\n", "bold cyan"),
            ("Interactive CLI\n", "cyan"),
            ("\nIdentity: ", "white"),
            (identity_name, "bold yellow"),
            ("\n\nType ", "dim"),
            ("/help", "bold"),
            (" for available commands, ", "dim"),
            ("/quit", "bold"),
            (" to exit", "dim"),
        )

        panel = Panel(
            welcome_text,
            border_style="cyan",
            box=box.DOUBLE,
        )
        self.console.print(panel)

    def print_user_input(self, text: str) -> None:
        """
        Display user input in formatted style.

        Args:
            text: User's input text
        """
        self.console.print(f"[bold green]You >[/bold green] {text}")

    def print_error(self, message: str) -> None:
        """
        Display error message.

        Args:
            message: Error message to display
        """
        self.console.print(f"[bold red]Error:[/bold red] {message}")

    def print_info(self, message: str) -> None:
        """
        Display info message.

        Args:
            message: Info message to display
        """
        self.console.print(f"[bold blue]Info:[/bold blue] {message}")

    # Helper methods

    def _create_sparkline(self, values: List[float], width: int = 40) -> str:
        """
        Create a text sparkline from values.

        Args:
            values: List of float values
            width: Width of sparkline in characters

        Returns:
            Sparkline string
        """
        if not values:
            return "▁" * width

        # Take last `width` values
        if len(values) > width:
            values = values[-width:]

        # Normalize to 0-7 range for block characters
        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val if max_val > min_val else 1.0

        blocks = " ▁▂▃▄▅▆▇█"
        sparkline = ""
        for val in values:
            normalized = (val - min_val) / value_range
            block_idx = int(normalized * 8)
            block_idx = min(block_idx, 8)  # Clamp to max
            sparkline += blocks[block_idx]

        # Pad if needed
        if len(sparkline) < width:
            sparkline += "▁" * (width - len(sparkline))

        return sparkline

    def _format_pn_value(self, exp: ConsciousExperience) -> str:
        """Format PN value with color based on level."""
        # We don't have direct PN in experience, use uncertainty as proxy
        if exp.uncertainty_level == "critical":
            return "[bold red]CRITICAL[/bold red]"
        elif exp.uncertainty_level == "high":
            return "[yellow]HIGH[/yellow]"
        elif exp.uncertainty_level == "moderate":
            return "[white]MODERATE[/white]"
        else:
            return "[green]LOW[/green]"

    def _colorize_uncertainty(self, level: str) -> str:
        """Add color to uncertainty level."""
        colors = {
            "critical": "bold red",
            "high": "yellow",
            "moderate": "white",
            "low": "green",
        }
        color = colors.get(level, "white")
        return f"[{color}]{level}[/{color}]"

    def _format_confidence(self, confidence: float) -> str:
        """Format confidence value with color."""
        if confidence >= 0.9:
            color = "bold green"
        elif confidence >= 0.7:
            color = "green"
        elif confidence >= 0.5:
            color = "yellow"
        else:
            color = "red"

        return f"[{color}]{confidence:.1%}[/{color}]"
