# tui.py
"""
The interactive Textual User Interface for the Riemann-J architecture.
WHY: A TUI provides a much richer, more dynamic, and professional user experience
than a simple CLI. It allows for real-time, graphical visualization of the system's
internal state (like the PN sparkline) alongside the conversation log, making
the architecture's behavior tangible and intuitive.

FEATURES:
- Live PN trajectory through critical strip visualization
- Real-time equilibrium regulator dynamics
- Latent manifold clustering (via formative experiences)
- Auto-refreshing status dashboard
- Conversation log with rich formatting
"""
import queue
import threading
from collections import deque

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Label, RichLog, Sparkline, Static

from .architecture import CognitiveWorkspace
from .config import *
from .conscious_agent import ConsciousAgent
from .pn_driver import PNDriverRiemannZeta, PredictionErrorSignal
from .shared_resources import global_workspace


class TUI(App):
    TITLE = "Riemann-J Cognitive Architecture v4.0 - Live Dashboard"
    CSS_PATH = "tui.css"

    # Reactive variables will automatically update the UI when changed
    current_user = reactive("default_user")
    current_pn = reactive(0.0)
    target_pn = reactive(0.5)
    regulated_pn = reactive(0.5)
    attractor_size = reactive(0)
    workspace_q_size = reactive(0)
    system_status = reactive("NOMINAL")
    status_style = reactive("green")
    pn_history = deque([0.0] * 50, maxlen=50)

    # Critical strip visualization
    driver_t = reactive(14.1347)
    steps_since_zero = reactive(0)

    # System metrics
    total_interactions = reactive(0)
    total_crises = reactive(0)
    formative_count = reactive(0)
    last_crisis_pn = reactive(0.0)
    last_crisis_iters = reactive(0)
    last_crisis_type = reactive("none")

    # Self-belief
    stability = reactive(0.5)
    competence = reactive(0.5)
    uncertainty = reactive(0.5)

    def __init__(self):
        super().__init__()
        self.workspace = CognitiveWorkspace()
        self.agent = ConsciousAgent(workspace=self.workspace, self_id="tui_agent")
        self.pn_driver = self.workspace.pn_driver

    def compose(self) -> ComposeResult:
        """Create child widgets for the app with comprehensive live dashboard."""
        yield Header()
        with Container(id="main_container"):
            # Top: Live Dashboard
            with Horizontal(id="live_dashboard"):
                # Left column: PN trajectory and critical strip
                with Vertical(id="pn_column"):
                    yield Label("━━━ PN TRAJECTORY ━━━", id="pn_title")
                    yield Sparkline(
                        self.pn_history,
                        id="pn_sparkline",
                        summary_function=max,  # Use max value for summary
                    )
                    yield Static(id="pn_details")
                    yield Static(id="critical_strip")

                # Middle column: System metrics
                with Vertical(id="metrics_column"):
                    yield Label("━━━ SYSTEM STATE ━━━", id="metrics_title")
                    yield Static(id="metrics_display")
                    yield Static(id="belief_display")

                # Right column: Equilibrium regulator
                with Vertical(id="regulator_column"):
                    yield Label("━━━ EQUILIBRIUM ━━━", id="eq_title")
                    yield Static(id="regulator_display")

            # Middle: Conversation log
            yield RichLog(id="conversation_log", auto_scroll=True, highlight=True, wrap=True)

            # Bottom: Input
            yield Input(placeholder="Type your message or /help for commands...", id="user_input")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.query_one("#user_input").focus()
        # Set intervals to update UI and check for system messages
        self.set_interval(1 / 4, self.update_dashboard)
        self.set_interval(1 / 10, self.check_workspace_queue)

    def update_dashboard(self) -> None:
        """Update the comprehensive live dashboard with all system metrics."""
        # Get regulated PN from equilibrium regulator
        self.regulated_pn = self.agent.equilibrium_regulator.get_current_pn()
        self.target_pn = self.agent.equilibrium_regulator.target_pn

        # Update PN history
        self.pn_history.append(self.regulated_pn)
        self.query_one(Sparkline).data = list(self.pn_history)

        # Get PN driver state
        self.driver_t = self.pn_driver.current_t
        self.steps_since_zero = self.pn_driver.steps_since_last_zero

        # Get metrics
        metrics = self.agent.persistent_self.metrics
        self.total_interactions = metrics.total_interactions
        self.total_crises = metrics.total_crises
        self.formative_count = metrics.formative_experiences

        # Get self-belief
        belief = self.agent.meta_monitor.self_belief
        self.stability = belief.stability
        self.competence = belief.competence
        self.uncertainty = belief.uncertainty

        # Get user attractor
        user_attractor = self.workspace.get_or_create_user(self.current_user)
        self.attractor_size = len(user_attractor.state_history)
        self.workspace_q_size = global_workspace.qsize()

        # Determine system status
        if self.regulated_pn > PN_THRESHOLD:
            self.system_status = "J-SHIFT IMMINENT!"
            self.status_style = "bold red"
        elif self.regulated_pn > 0.5:
            self.system_status = "HIGH PRESSURE"
            self.status_style = "yellow"
        else:
            self.system_status = "NOMINAL"
            self.status_style = "green"

        # Update PN details
        pn_delta = abs(self.regulated_pn - self.target_pn)
        pn_details = (
            f"Regulated: {self.regulated_pn:.4f}\n"
            f"Target:    {self.target_pn:.4f}\n"
            f"Δ:         {pn_delta:.4f}\n"
            f"Status: [{self.status_style}]{self.system_status}[/{self.status_style}]"
        )
        self.query_one("#pn_details").update(pn_details)

        # Update critical strip visualization
        strip_visual = self._render_critical_strip()
        self.query_one("#critical_strip").update(strip_visual)

        # Update metrics display
        metrics_text = (
            f"Interactions: {self.total_interactions}\n"
            f"Crises:       {self.total_crises}\n"
            f"Resolved:     {metrics.successful_resolutions}\n"
            f"Failed:       {metrics.failed_resolutions}\n"
            f"Formative:    {self.formative_count}\n"
            f"\n"
            f"Last Crisis: PN={self.last_crisis_pn:.4f}\n"
            f"  → {self.last_crisis_iters} iters ({self.last_crisis_type})\n"
            f"\n"
            f"User Attractor: {self.attractor_size} states\n"
            f"Queue Size:     {self.workspace_q_size}"
        )
        self.query_one("#metrics_display").update(metrics_text)

        # Update belief display
        belief_text = (
            f"[bold cyan]Self-Belief State[/bold cyan]\n"
            f"Stability:    {self._render_bar(self.stability)}\n"
            f"Competence:   {self._render_bar(self.competence)}\n"
            f"Uncertainty:  {self._render_bar(self.uncertainty)}"
        )
        self.query_one("#belief_display").update(belief_text)

        # Update regulator display
        tau = self.agent.equilibrium_regulator.tau
        regulator_text = (
            f"Time Constant: {tau:.1f}s\n"
            f"\n"
            f"[bold yellow]Driver State[/bold yellow]\n"
            f"t = {self.driver_t:.4f}\n"
            f"Steps: {self.steps_since_zero}\n"
            f"\n"
            f"Target seeking:\n"
            f"{self._render_convergence_arrow()}"
        )
        self.query_one("#regulator_display").update(regulator_text)

    def _render_critical_strip(self) -> str:
        """Render ASCII visualization of position in critical strip."""
        # Mini version for TUI (10 chars wide, 6 tall)
        width = 20
        height = 6

        current_pos = int(self.regulated_pn * (height - 1))
        target_pos = int(self.target_pn * (height - 1))

        lines = ["[bold cyan]Critical Strip[/bold cyan]"]
        for i in range(height - 1, -1, -1):
            line = "│"
            for j in range(width):
                if i == current_pos and j == width // 2:
                    line += "●"  # Current PN
                elif i == target_pos and j == width // 2:
                    line += "◎"  # Target
                elif j == width // 2:
                    line += "│"  # Critical line
                else:
                    line += " "
            line += "│"
            pn_val = i / (height - 1)
            line += f" {pn_val:.1f}"
            lines.append(line)

        lines.append("└" + "─" * width + "┘")
        return "\n".join(lines)

    def _render_bar(self, value: float) -> str:
        """Render a progress bar for a 0-1 value."""
        bar_length = 10
        filled = int(value * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        return f"{bar} {value:.2f}"

    def _render_convergence_arrow(self) -> str:
        """Render arrow showing convergence direction."""
        if abs(self.regulated_pn - self.target_pn) < 0.01:
            return "  ═══ ⊙ ═══  [green]EQUILIBRIUM[/green]"
        elif self.regulated_pn > self.target_pn:
            return "  ↓↓↓ ⇣ ↓↓↓  [yellow]DECAYING[/yellow]"
        else:
            return "  ↑↑↑ ⇡ ↑↑↑  [yellow]RISING[/yellow]"

    def check_workspace_queue(self) -> None:
        """Check for and process high-priority J-Shift messages."""
        try:
            priority, counter, message = global_workspace.get_nowait()
            if isinstance(message, PredictionErrorSignal) and message.p_n > PN_THRESHOLD:
                state_obj = self.workspace._j_operator_resolve(message)
                self.workspace.log_state(state_obj)

                # HOOK: Let persistent_self track this crisis for metrics
                self.agent.persistent_self.integrate_crisis(state_obj)

                # Update dashboard crisis info (NO LOG SPAM!)
                self.last_crisis_pn = message.p_n
                self.last_crisis_iters = state_obj.analysis.get("iterations", 0)
                self.last_crisis_type = state_obj.analysis.get("convergence_type", "unknown")
        except queue.Empty:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input with command support."""
        user_input = event.value
        log = self.query_one(RichLog)
        log.write(f"[bold]You >[/bold] {user_input}")
        event.input.clear()

        # Handle commands
        if user_input.lower() in ["/exit", "/quit"]:
            self.agent.equilibrium_regulator.stop()
            self.workspace.close()
            self.exit()
            return

        if user_input.startswith("/switch "):
            self.current_user = user_input.split(" ", 1)[1]
            log.write(f"[bold green]Switched to user: {self.current_user}[/bold green]")
            return

        if user_input == "/help":
            help_text = """[bold cyan]Commands:[/bold cyan]
/help - Show this help
/exit, /quit - Exit TUI
/switch <user> - Switch user context
/blend <alpha> - Set projection blending (0.0-1.0, default 0.0)
/inject-state <trigger> [--pn=N] [--crisis] - Inject synthetic state
/stats - Show system statistics
/introspect - Show meta-cognitive state

Regular messages are processed through the conscious agent."""
            log.write(help_text)
            return

        if user_input.startswith("/blend "):
            try:
                alpha = float(user_input.split(" ", 1)[1])
                if 0.0 <= alpha <= 1.0:
                    import riemann_j.config as cfg

                    cfg.PROJECTION_BLEND_ALPHA = alpha
                    log.write(f"[bold green]✓ Projection blend set to {alpha:.2f}[/bold green]")
                    if alpha == 0.0:
                        log.write("[dim]Pure prompt-based generation (default)[/dim]")
                    elif alpha < 0.3:
                        log.write("[dim]Subtle state influence (experimental)[/dim]")
                    else:
                        log.write("[dim yellow]⚠️  High blend - may degrade quality[/dim yellow]")
                else:
                    log.write("[red]Error: Alpha must be between 0.0 and 1.0[/red]")
            except (ValueError, IndexError):
                log.write("[red]Usage: /blend <alpha>  (e.g., /blend 0.1)[/red]")
            return

        if user_input == "/stats":
            metrics = self.agent.persistent_self.metrics
            stats_text = f"""[bold cyan]System Statistics:[/bold cyan]
Total Interactions: {metrics.total_interactions}
Total Crises: {metrics.total_crises}
Successful Resolutions: {metrics.successful_resolutions}
Failed Resolutions: {metrics.failed_resolutions}
Formative Experiences: {metrics.formative_experiences}
Current PN: {self.regulated_pn:.4f}
Stability: {self.stability:.2f}
Competence: {self.competence:.2f}"""
            log.write(stats_text)
            return

        if user_input == "/introspect":
            report = self.agent.introspect(verbose=True)
            log.write(f"[bold cyan]Meta-Cognitive State:[/bold cyan]\n{report}")
            return

        if user_input.startswith("/inject-state"):
            self._handle_inject_state(user_input)
            return

        if user_input:
            # Run the conscious agent processing in a separate thread to not block the UI
            threading.Thread(target=self.run_user_processing, args=(user_input,)).start()

    def _handle_inject_state(self, command: str):
        """Handle /inject-state command."""
        import time

        import numpy as np

        from .architecture import SyntheticState

        parts = command.split()[1:]  # Skip "/inject-state"
        if not parts:
            self.query_one(RichLog).write("[red]Error: Trigger required[/red]")
            return

        # Parse flags
        pn_override = None
        is_crisis = False
        trigger_parts = []

        for part in parts:
            if part.startswith("--pn="):
                try:
                    pn_override = float(part.split("=")[1])
                except ValueError:
                    self.query_one(RichLog).write(f"[red]Error: Invalid PN value[/red]")
                    return
            elif part == "--crisis":
                is_crisis = True
            else:
                trigger_parts.append(part)

        trigger = " ".join(trigger_parts)
        pn_value = pn_override if pn_override is not None else self.regulated_pn

        # Create state
        state = SyntheticState(
            timestamp=time.time(),
            latent_representation=np.random.randn(768).astype(np.float32),
            source_trigger=f"manual_injection: {trigger}",
            p_n_at_creation=pn_value,
            is_j_shift_product=False,
            status="INJECTED",
        )

        # Integrate
        if is_crisis or pn_value >= 0.5:
            self.agent.persistent_self.integrate_crisis(state)
            self.query_one(RichLog).write(
                f"[bold yellow]✓ Injected crisis state (PN={pn_value:.4f}): {trigger}[/bold yellow]"
            )
        else:
            self.agent.persistent_self.integrate_interaction(state)
            self.query_one(RichLog).write(
                f"[bold green]✓ Injected routine state (PN={pn_value:.4f}): {trigger}[/bold green]"
            )

        # Inject perturbation
        self.agent.equilibrium_regulator.inject_perturbation(pn_value)

    def run_user_processing(self, user_input: str):
        """Process user input through conscious agent (runs in background thread)."""
        try:
            # Use conscious agent for full experience
            experience = self.agent.process_consciously(user_id=self.current_user, text=user_input)

            # Update UI from thread - FIXED: use write() not write_line()
            log = self.query_one(RichLog)
            self.call_from_thread(
                log.write,
                f"[bold cyan]Agent:[/bold cyan] {experience.response}",
            )

            # Show metadata if high uncertainty
            if experience.uncertainty_level in ["high", "critical"]:
                metadata = (
                    f"[dim]Uncertainty: {experience.uncertainty_level} | "
                    f"Confidence: {experience.confidence:.1%} | "
                    f"PN: {self.regulated_pn:.3f}[/dim]"
                )
                self.call_from_thread(log.write, metadata)
        except Exception as e:
            # Log error to UI so user can see what went wrong
            import traceback

            error_msg = f"[bold red]Error processing input:[/bold red] {str(e)}"
            self.call_from_thread(self.query_one(RichLog).write, error_msg)
            # Also print full traceback to console for debugging
            traceback.print_exc()


def main():
    """Main entry point for the Riemann-J TUI application."""
    app = TUI()
    app.run()


if __name__ == "__main__":
    main()
