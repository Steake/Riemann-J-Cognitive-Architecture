# tui.py
"""
The interactive Textual User Interface for the Riemann-J architecture.
WHY: A TUI provides a much richer, more dynamic, and professional user experience
than a simple CLI. It allows for real-time, graphical visualization of the system's
internal state (like the PN sparkline) alongside the conversation log, making
the architecture's behavior tangible and intuitive.
"""
import queue
import threading
from collections import deque

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Log, Sparkline, Static

from .architecture import CognitiveWorkspace
from .config import *
from .pn_driver import PNDriverRiemannZeta, PredictionErrorSignal
from .shared_resources import global_workspace


class TUI(App):
    TITLE = "Riemann-J Cognitive Architecture v4.0"
    CSS_PATH = "tui.css"

    # Reactive variables will automatically update the UI when changed
    current_user = reactive("default_user")
    current_pn = reactive(0.0)
    attractor_size = reactive(0)
    workspace_q_size = reactive(0)
    system_status = reactive("NOMINAL")
    status_style = reactive("green")
    pn_history = deque([0.0] * 50, maxlen=50)

    def __init__(self):
        super().__init__()
        self.workspace = CognitiveWorkspace()
        self.pn_driver = PNDriverRiemannZeta()
        self.pn_driver.start()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container(id="main_container"):
            with Horizontal(id="dashboard"):
                yield Sparkline(
                    self.pn_history,
                    id="pn_sparkline",
                    summary_function=lambda data: f"PN: {data[-1]:.3f}",
                )
                yield Static(id="status_display")
            yield Log(id="conversation_log", auto_scroll=True)
            yield Input(placeholder="Type your message...", id="user_input")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.query_one("#user_input").focus()
        # Set intervals to update UI and check for system messages
        self.set_interval(1 / 4, self.update_dashboard)
        self.set_interval(1 / 10, self.check_workspace_queue)

    def update_dashboard(self) -> None:
        """Update the reactive variables for the dashboard."""
        if global_workspace.queue:
            # Peek at the highest priority item without removing it
            _priority, message = global_workspace.queue[0]
            if isinstance(message, PredictionErrorSignal):
                self.current_pn = message.p_n
                self.pn_history.append(self.current_pn)
                # Update sparkline data
                self.query_one(Sparkline).data = list(self.pn_history)

        user_attractor = self.workspace.get_or_create_user(self.current_user)
        self.attractor_size = len(user_attractor.state_history)
        self.workspace_q_size = global_workspace.qsize()

        if self.current_pn > PN_THRESHOLD:
            self.system_status = "J-SHIFT IMMINENT!"
            self.status_style = "bold red"
        elif self.current_pn > 0.5:
            self.system_status = "HIGH PRESSURE"
            self.status_style = "yellow"
        else:
            self.system_status = "NOMINAL"
            self.status_style = "green"

        status_text = (
            f"User: [bold]{self.current_user}[/bold]\n"
            f"Status: [{self.status_style}]{self.system_status}[/{self.status_style}]\n"
            f"Attractor Size: {self.attractor_size}\n"
            f"Queue Size: {self.workspace_q_size}"
        )
        self.query_one("#status_display").update(status_text)

    def check_workspace_queue(self) -> None:
        """Check for and process high-priority J-Shift messages."""
        try:
            priority, counter, message = global_workspace.get_nowait()
            if isinstance(message, PredictionErrorSignal) and message.p_n > PN_THRESHOLD:
                log = self.query_one(Log)
                log.write_line(
                    f"[bold red blink]>> J-SHIFT TRIGGERED! PN = {message.p_n:.4f} <<[/bold red blink]"
                )
                state_obj = self.workspace._j_operator_resolve(message)
                self.workspace.log_state(state_obj)
                response = self.workspace.symbolic_interface.decoder(
                    state_obj.latent_representation
                )
                log.write_line(
                    f"[bold yellow]System Internal Response ({state_obj.status}):[/bold yellow] {response}"
                )
        except queue.Empty:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input."""
        user_input = event.value
        log = self.query_one(Log)
        log.write_line(f"[bold]You >[/bold] {user_input}")
        event.input.clear()

        if user_input.lower() == "/exit":
            self.pn_driver.is_running = False
            self.workspace.close()
            self.exit()
            return

        if user_input.startswith("/switch "):
            self.current_user = user_input.split(" ", 1)[1]
            log.write_line(f"[bold green]Switched to user: {self.current_user}[/bold green]")
            return

        if user_input:
            # Run the workspace processing in a separate thread to not block the UI
            threading.Thread(target=self.run_user_processing, args=(user_input,)).start()

    def run_user_processing(self, user_input: str):
        response, state_obj = self.workspace.process_user_input(self.current_user, user_input)
        self.workspace.log_state(state_obj)
        # Use call_from_thread to safely update UI from another thread
        self.call_from_thread(
            self.query_one(Log).write_line, f"[bold cyan]Agent:[/bold cyan] {response}"
        )


def main():
    """Main entry point for the Riemann-J TUI application."""
    app = TUI()
    app.run()


if __name__ == "__main__":
    main()
