"""
Unit tests for CLI display manager.

Tests the rich formatting and visualization capabilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from riemann_j.cli_display import DisplayManager
from riemann_j.conscious_agent import ConsciousExperience
from rich.console import Console
from io import StringIO


class TestDisplayManager:
    """Test suite for DisplayManager."""

    def test_initialization(self):
        """Test that DisplayManager initializes correctly."""
        display = DisplayManager()
        assert display.console is not None

    def test_custom_console(self):
        """Test initialization with custom console."""
        custom_console = Console()
        display = DisplayManager(console=custom_console)
        assert display.console is custom_console

    def test_create_sparkline_empty(self):
        """Test sparkline creation with empty values."""
        display = DisplayManager()
        sparkline = display._create_sparkline([], width=10)
        assert len(sparkline) == 10
        assert all(c == "▁" for c in sparkline)

    def test_create_sparkline_normal(self):
        """Test sparkline creation with normal values."""
        display = DisplayManager()
        values = [0.1, 0.5, 0.9, 0.3, 0.7]
        sparkline = display._create_sparkline(values, width=5)
        assert len(sparkline) == 5
        # Higher values should use higher blocks
        assert sparkline != "▁" * 5

    def test_format_pn_value_critical(self):
        """Test PN value formatting for critical level."""
        display = DisplayManager()
        exp = ConsciousExperience(
            timestamp=123.0,
            user_input="test",
            internal_state="test",
            uncertainty_level="critical",
            confidence=0.5,
            response="test",
        )
        result = display._format_pn_value(exp)
        assert "CRITICAL" in result
        assert "red" in result

    def test_format_pn_value_low(self):
        """Test PN value formatting for low level."""
        display = DisplayManager()
        exp = ConsciousExperience(
            timestamp=123.0,
            user_input="test",
            internal_state="test",
            uncertainty_level="low",
            confidence=0.95,
            response="test",
        )
        result = display._format_pn_value(exp)
        assert "LOW" in result
        assert "green" in result

    def test_colorize_uncertainty(self):
        """Test uncertainty colorization."""
        display = DisplayManager()
        
        result_critical = display._colorize_uncertainty("critical")
        assert "red" in result_critical
        
        result_high = display._colorize_uncertainty("high")
        assert "yellow" in result_high
        
        result_low = display._colorize_uncertainty("low")
        assert "green" in result_low

    def test_format_confidence_high(self):
        """Test confidence formatting for high values."""
        display = DisplayManager()
        result = display._format_confidence(0.95)
        assert "95.0%" in result
        assert "green" in result

    def test_format_confidence_low(self):
        """Test confidence formatting for low values."""
        display = DisplayManager()
        result = display._format_confidence(0.35)
        assert "35.0%" in result
        assert "red" in result

    def test_render_response_with_metadata(self):
        """Test rendering response with metadata."""
        console = Console(file=StringIO(), force_terminal=True)
        display = DisplayManager(console=console)
        
        exp = ConsciousExperience(
            timestamp=123.0,
            user_input="test input",
            internal_state="test state",
            uncertainty_level="moderate",
            confidence=0.75,
            response="Test response",
        )
        
        # Should not raise exception
        display.render_response(exp, show_metadata=True)

    def test_render_response_without_metadata(self):
        """Test rendering response without metadata."""
        console = Console(file=StringIO(), force_terminal=True)
        display = DisplayManager(console=console)
        
        exp = ConsciousExperience(
            timestamp=123.0,
            user_input="test input",
            internal_state="test state",
            uncertainty_level="low",
            confidence=0.95,
            response="Test response",
        )
        
        # Should not raise exception
        display.render_response(exp, show_metadata=False)

    def test_render_pn_monitor(self):
        """Test PN monitor rendering."""
        console = Console(file=StringIO(), force_terminal=True)
        display = DisplayManager(console=console)
        
        pn_history = [0.1, 0.2, 0.3, 0.4, 0.5]
        current_pn = 0.5
        
        # Should not raise exception
        display.render_pn_monitor(pn_history, current_pn)

    def test_render_help(self):
        """Test help rendering."""
        console = Console(file=StringIO(), force_terminal=True)
        display = DisplayManager(console=console)
        
        # Should not raise exception
        display.render_help()

    def test_render_welcome(self):
        """Test welcome banner rendering."""
        console = Console(file=StringIO(), force_terminal=True)
        display = DisplayManager(console=console)
        
        # Should not raise exception
        display.render_welcome("test_identity")

    def test_print_user_input(self):
        """Test user input printing."""
        console = Console(file=StringIO(), force_terminal=True)
        display = DisplayManager(console=console)
        
        # Should not raise exception
        display.print_user_input("test input")

    def test_print_error(self):
        """Test error message printing."""
        console = Console(file=StringIO(), force_terminal=True)
        display = DisplayManager(console=console)
        
        # Should not raise exception
        display.print_error("test error")

    def test_print_info(self):
        """Test info message printing."""
        console = Console(file=StringIO(), force_terminal=True)
        display = DisplayManager(console=console)
        
        # Should not raise exception
        display.print_info("test info")
