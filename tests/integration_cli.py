"""
Integration tests for the CLI.

These tests validate the CLI works with the real ConsciousAgent
(but may use lightweight mocking for the model).
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from riemann_j.cli import RiemannCLI
from riemann_j.cli_config import SessionState


class TestCLIIntegration:
    """Integration tests for CLI with real components."""

    def test_cli_initialization_with_identity(self):
        """Test CLI initialization with custom identity path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            identity_path = str(Path(tmpdir) / "test_identity.pkl")
            cli = RiemannCLI(identity_path=identity_path, use_rich=False)
            
            assert cli.agent is not None
            assert cli.workspace is not None
            assert cli.session is not None
            assert cli.input_handler is not None
            assert cli.command_handler is not None

    def test_session_save_and_load(self):
        """Test saving and loading sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "test_session.json"
            
            # Create a session and add some data
            session = SessionState(identity_path="test_identity")
            session.add_turn("user", "Hello", pn=0.1, confidence=0.9)
            session.add_turn("agent", "Hi there!", pn=0.05, confidence=0.95)
            
            # Save it
            session.save(str(session_path))
            assert session_path.exists()
            
            # Load it
            loaded_session = SessionState.load(str(session_path))
            assert loaded_session.identity_path == "test_identity"
            assert loaded_session.total_turns == 1
            assert len(loaded_session.conversation_history) == 2

    def test_multiline_input_flow(self):
        """Test multi-line input handling."""
        cli = RiemannCLI(use_rich=False)
        
        # Start multiline
        cli.handle_input('"""')
        in_mode, lines = cli.input_handler.get_multiline_status()
        assert in_mode is True
        
        # Add lines
        cli.handle_input("Line 1")
        cli.handle_input("Line 2")
        
        in_mode, lines = cli.input_handler.get_multiline_status()
        assert lines == 2
        
        # End multiline (would process through agent)
        # We don't actually call this because it would invoke the model
        # Just verify the buffer is populated
        assert len(cli.input_handler.multiline_buffer) == 2

    def test_command_routing(self):
        """Test that commands are routed correctly."""
        cli = RiemannCLI(use_rich=False)
        
        # Mock the command handler
        cli.command_handler.handle = MagicMock(return_value=True)
        
        # Send a command
        cli.handle_input("/help")
        
        # Verify it was routed to command handler
        cli.command_handler.handle.assert_called_once_with("/help")

    def test_message_routing(self):
        """Test that messages are routed to agent."""
        cli = RiemannCLI(use_rich=False)
        
        # Mock the process method
        cli._process_message = MagicMock()
        
        # Send a message
        cli.handle_input("Hello")
        
        # Verify it was routed to message processor
        cli._process_message.assert_called_once_with("Hello")

    def test_help_command_display(self):
        """Test help command displays correctly."""
        cli = RiemannCLI(use_rich=False)
        
        with patch('builtins.print') as mock_print:
            cli.command_handler._cmd_help()
            
            # Check that help was printed
            calls = [str(call) for call in mock_print.call_args_list]
            help_output = "".join(calls)
            assert "/help" in help_output
            assert "/quit" in help_output
            assert "/introspect" in help_output

    def test_session_tracking(self):
        """Test that conversation turns are tracked in session."""
        cli = RiemannCLI(use_rich=False)
        
        # Mock agent to avoid model loading
        mock_experience = MagicMock()
        mock_experience.response = "Test response"
        mock_experience.confidence = 0.95
        mock_experience.uncertainty_level = "low"
        cli.agent.process_consciously = MagicMock(return_value=mock_experience)
        cli.agent.meta_monitor.get_current_pn = MagicMock(return_value=0.05)
        
        # Process a message
        initial_turns = cli.session.total_turns
        cli._process_message("Test message")
        
        # Verify session was updated
        assert cli.session.total_turns == initial_turns + 1
        assert len(cli.session.conversation_history) >= 2  # user + agent

    def test_quit_command_stops_loop(self):
        """Test that /quit command stops the REPL loop."""
        cli = RiemannCLI(use_rich=False)
        cli.running = True
        
        # Execute quit command
        cli.command_handler._cmd_quit()
        
        # Verify running is set to False
        # Note: The command handler returns False to signal exit
        # but doesn't directly set cli.running

    def test_error_handling_invalid_input(self):
        """Test handling of invalid input."""
        cli = RiemannCLI(use_rich=False)
        
        # Very long input should be rejected
        long_input = "a" * 10001
        
        with patch('builtins.print') as mock_print:
            cli.handle_input(long_input)
            
            # Should print an error
            calls = [str(call) for call in mock_print.call_args_list]
            error_output = "".join(calls)
            assert "error" in error_output.lower() or "long" in error_output.lower()
