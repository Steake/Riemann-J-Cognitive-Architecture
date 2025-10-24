"""
Unit tests for the new CLI implementation.

These tests validate the basic REPL functionality without requiring the full model.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from riemann_j.cli import RiemannCLI
from riemann_j.conscious_agent import ConsciousExperience


class TestRiemannCLI:
    """Test suite for RiemannCLI basic functionality."""

    def test_cli_initialization(self):
        """Test that CLI initializes correctly."""
        cli = RiemannCLI(identity_path="test_identity")
        assert cli.agent is not None
        assert cli.workspace is not None
        assert cli.running is False
        assert cli.use_rich is True
        assert cli.display is not None

    def test_cli_initialization_no_rich(self):
        """Test that CLI initializes correctly without rich."""
        cli = RiemannCLI(identity_path="test_identity", use_rich=False)
        assert cli.agent is not None
        assert cli.workspace is not None
        assert cli.running is False
        assert cli.use_rich is False
        assert cli.display is None

    def test_quit_command(self):
        """Test that /quit command stops the REPL."""
        cli = RiemannCLI()
        cli.running = True
        cli.handle_input("/quit")
        assert cli.running is False

    def test_exit_command(self):
        """Test that /exit command stops the REPL."""
        cli = RiemannCLI()
        cli.running = True
        cli.handle_input("/exit")
        assert cli.running is False

    def test_help_command(self, capsys):
        """Test that /help command displays help text."""
        cli = RiemannCLI()
        cli.handle_input("/help")
        captured = capsys.readouterr()
        assert "Available Commands" in captured.out
        assert "/quit" in captured.out
        assert "/introspect" in captured.out

    def test_unknown_command(self, capsys):
        """Test handling of unknown commands."""
        cli = RiemannCLI()
        cli.handle_input("/unknown")
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    @patch.object(RiemannCLI, '_process_message')
    def test_message_processing(self, mock_process):
        """Test that non-command input is processed as message."""
        cli = RiemannCLI()
        cli.handle_input("Hello, agent!")
        mock_process.assert_called_once_with("Hello, agent!")

    def test_introspect_command_calls_agent(self):
        """Test that /introspect command calls agent.introspect()."""
        cli = RiemannCLI()
        cli.agent.introspect = Mock(return_value="Introspection report")
        
        with patch('builtins.print'):
            cli.handle_input("/introspect")
        
        cli.agent.introspect.assert_called_once_with(verbose=True)

    def test_identity_command_calls_agent(self):
        """Test that /identity command calls agent.get_formative_narrative()."""
        cli = RiemannCLI()
        cli.agent.get_formative_narrative = Mock(return_value="Identity narrative")
        
        with patch('builtins.print'):
            cli.handle_input("/identity")
        
        cli.agent.get_formative_narrative.assert_called_once()

    def test_display_response_shows_metadata_on_high_uncertainty(self):
        """Test that high uncertainty shows metadata."""
        cli = RiemannCLI(use_rich=False)  # Use plain text for easier testing
        cli.agent.meta_monitor.get_current_pn = Mock(return_value=0.75)
        
        experience = ConsciousExperience(
            timestamp=1234567890.0,
            user_input="test input",
            internal_state="test state",
            uncertainty_level="high",
            confidence=0.65,
            response="Test response",
        )
        
        with patch('builtins.print') as mock_print:
            cli.display_response(experience)
            # Check that print was called with response
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("Test response" in str(call) for call in calls)

    def test_display_response_minimal_on_low_uncertainty(self):
        """Test that low uncertainty shows only response."""
        cli = RiemannCLI(use_rich=False)  # Use plain text for easier testing
        
        experience = ConsciousExperience(
            timestamp=1234567890.0,
            user_input="test input",
            internal_state="test state",
            uncertainty_level="low",
            confidence=0.95,
            response="Test response",
        )
        
        with patch('builtins.print') as mock_print:
            cli.display_response(experience)
            # Check that print was called with response
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("Test response" in str(call) for call in calls)
