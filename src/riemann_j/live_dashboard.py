"""
Live auto-refreshing dashboard for CLI.

This module implements a background thread that continuously updates a status bar
showing real-time system state: PN trajectory through critical strip, latent manifold
visualization, equilibrium regulator dynamics, and system metrics.

WHY: Provides real-time visibility into the cognitive architecture's internal dynamics
without requiring manual commands. The user sees the system's "vital signs" continuously.
"""

import math
import threading
import time
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class LiveDashboard(threading.Thread):
    """
    Background thread that maintains a live-updating dashboard.

    Visualizes:
    - PN trajectory through Riemann critical strip
    - Equilibrium regulator dynamics
    - Latent manifold state clustering
    - System metrics (crisis count, formative experiences, etc.)
    """

    def __init__(
        self,
        agent,
        pn_driver,
        console: Optional[Console] = None,
        update_interval: float = 0.5,
    ):
        """
        Initialize live dashboard.

        Args:
            agent: ConsciousAgent instance
            pn_driver: PNDriverRiemannZeta instance
            console: Rich console for rendering
            update_interval: Update frequency in seconds
        """
        super().__init__(daemon=True)
        self.agent = agent
        self.pn_driver = pn_driver
        self.console = console or Console()
        self.update_interval = update_interval
        self.is_running = True
        self.enabled = False  # Disabled by default to not interfere with prompt
        self.live_display = None

    def run(self):
        """Main update loop - prints compact status line periodically."""
        while self.is_running:
            if self.enabled:
                # Print compact one-line status
                status = self._render_compact_status()
                # Use \r to overwrite same line (non-invasive)
                print(f"\r{status}", end="", flush=True)
            time.sleep(self.update_interval)

    def _render_compact_status(self) -> str:
        """Render compact one-line status."""
        current_pn = self.agent.equilibrium_regulator.get_current_pn()
        target_pn = self.agent.equilibrium_regulator.target_pn
        driver_t = self.pn_driver.current_t
        steps = self.pn_driver.steps_since_last_zero

        # Create mini sparkline (10 chars)
        pn_history = self.agent.meta_monitor.pn_history
        sparkline = self._create_sparkline(pn_history, width=10)

        # Compact status
        status = (
            f"[PN:{current_pn:.3f}→{target_pn:.1f} {sparkline} "
            f"t:{driver_t:.1f} steps:{steps:3d} "
            f"crises:{self.agent.persistent_self.metrics.total_crises}]"
        )

        return status

    def _render_pn_trajectory(self) -> Panel:
        """
        Render PN trajectory through the critical strip.

        Visualizes:
        - Current position in critical strip (Re(s) = 0.5 + Im(s))
        - PN driver's t parameter progression
        - Equilibrium regulator's target seeking
        """
        # Get current state
        current_pn = self.agent.equilibrium_regulator.get_current_pn()
        target_pn = self.agent.equilibrium_regulator.target_pn
        driver_t = self.pn_driver.current_t
        steps_since_zero = self.pn_driver.steps_since_last_zero

        # Create critical strip visualization (ASCII art)
        strip_visual = self._draw_critical_strip(current_pn, target_pn)

        # Create trajectory sparkline
        pn_history = self.agent.meta_monitor.pn_history
        sparkline = self._create_sparkline(pn_history, width=60)

        # Build content
        content = Text()
        content.append("━━━ RIEMANN CRITICAL STRIP ━━━\n", style="bold cyan")
        content.append(strip_visual)
        content.append(f"\n\n")
        content.append(f"PN Trajectory: {sparkline}\n", style="yellow")
        content.append(f"Current PN: {current_pn:.4f}  ", style="bold green")
        content.append(f"Target: {target_pn:.4f}  ", style="dim")
        content.append(f"Δ = {abs(current_pn - target_pn):.4f}\n")
        content.append(
            f"\nDriver State: t = {driver_t:.4f}  Steps: {steps_since_zero}/{self.pn_driver._calculate_pn.__code__.co_consts[1] if hasattr(self.pn_driver, '_calculate_pn') else 'N/A'}"
        )

        return Panel(content, title="[bold]PN Dynamics[/bold]", border_style="cyan")

    def _draw_critical_strip(self, current_pn: float, target_pn: float) -> str:
        """
        Draw ASCII visualization of position in critical strip.

        The critical strip is Re(s) = 0.5, we visualize PN as position along Im(s).
        """
        width = 60
        height = 10

        # Calculate positions
        current_pos = int(current_pn * (height - 1))
        target_pos = int(target_pn * (height - 1))

        lines = []
        for i in range(height - 1, -1, -1):
            line = "│"

            # Draw the strip
            for j in range(width):
                if i == current_pos and j == width // 2:
                    line += "●"  # Current PN
                elif i == target_pos and j == width // 2:
                    line += "◎"  # Target
                elif j == width // 2:
                    line += "│"  # Critical line
                elif j % 10 == 0:
                    line += "·"  # Grid
                else:
                    line += " "

            line += "│"

            # Add PN value labels
            pn_value = i / (height - 1)
            line += f"  {pn_value:.1f}"

            lines.append(line)

        # Add bottom border
        lines.append("└" + "─" * width + "┘")
        lines.append(f"  Re(s) = 0.5 (Critical Strip)")

        return "\n".join(lines)

    def _render_system_metrics(self) -> Panel:
        """Render system metrics and state."""
        metrics = self.agent.persistent_self.metrics

        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="green")

        # Metrics
        table.add_row("Interactions:", f"{metrics.total_interactions}")
        table.add_row("Crises:", f"{metrics.total_crises}")
        table.add_row("Resolved:", f"{metrics.successful_resolutions}")
        table.add_row("Failed:", f"{metrics.failed_resolutions}")
        table.add_row("Formative:", f"{metrics.formative_experiences}")

        # Regulator state
        tau = self.agent.equilibrium_regulator.tau
        table.add_row("", "")
        table.add_row("Eq. τ:", f"{tau:.1f}s")

        # Self-belief
        belief = self.agent.meta_monitor.self_belief
        table.add_row("", "")
        table.add_row("Stability:", f"{belief.stability:.2f}")
        table.add_row("Competence:", f"{belief.competence:.2f}")
        table.add_row("Uncertainty:", f"{belief.uncertainty:.2f}")

        return Panel(table, title="[bold]System State[/bold]", border_style="green")

    def _create_sparkline(self, pn_history, width: int) -> str:
        """Create sparkline from PN history."""
        if not pn_history:
            return "▁" * width

        # Extract numeric values
        values_list = list(pn_history)[-width:]
        numeric_values = []
        for v in values_list:
            if isinstance(v, dict):
                numeric_values.append(v.get("value", 0.0))
            else:
                numeric_values.append(float(v))

        if not numeric_values:
            return "▁" * width

        # Normalize
        min_val = min(numeric_values)
        max_val = max(numeric_values)
        value_range = max_val - min_val if max_val > min_val else 1.0

        blocks = " ▁▂▃▄▅▆▇█"
        sparkline = ""
        for val in numeric_values:
            normalized = (val - min_val) / value_range
            block_idx = int(normalized * 8)
            block_idx = min(block_idx, 8)
            sparkline += blocks[block_idx]

        # Pad if needed
        if len(sparkline) < width:
            sparkline += "▁" * (width - len(sparkline))

        return sparkline[:width]

    def toggle(self):
        """Toggle dashboard on/off."""
        self.enabled = not self.enabled

    def stop(self):
        """Stop the dashboard thread."""
        self.is_running = False
        if self.live_display:
            self.live_display.stop()
