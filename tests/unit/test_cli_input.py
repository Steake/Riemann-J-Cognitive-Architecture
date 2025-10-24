"""
Unit tests for CLI input handler.
"""

import pytest
from riemann_j.cli_input import InputHandler, InputType


class TestInputHandler:
    """Test suite for InputHandler."""

    def test_initialization(self):
        """Test that InputHandler initializes correctly."""
        handler = InputHandler()
        assert handler.multiline_mode is False
        assert handler.multiline_buffer == []

    def test_parse_empty_input(self):
        """Test parsing empty input."""
        handler = InputHandler()
        input_type, content = handler.parse("")
        assert input_type == InputType.EMPTY
        assert content == ""

    def test_parse_command(self):
        """Test parsing command input."""
        handler = InputHandler()
        input_type, content = handler.parse("/help")
        assert input_type == InputType.COMMAND
        assert content == "/help"

    def test_parse_message(self):
        """Test parsing regular message."""
        handler = InputHandler()
        input_type, content = handler.parse("Hello, agent!")
        assert input_type == InputType.MESSAGE
        assert content == "Hello, agent!"

    def test_is_command(self):
        """Test command detection."""
        handler = InputHandler()
        assert handler.is_command("/help") is True
        assert handler.is_command("/quit") is True
        assert handler.is_command("Hello") is False
        assert handler.is_command("  /introspect  ") is True

    def test_validate_normal_input(self):
        """Test validation of normal input."""
        handler = InputHandler()
        valid, error = handler.validate("Hello")
        assert valid is True
        assert error == ""

    def test_validate_empty_input(self):
        """Test validation of empty input."""
        handler = InputHandler()
        valid, error = handler.validate("")
        assert valid is True
        assert error == ""

    def test_validate_long_input(self):
        """Test validation of excessively long input."""
        handler = InputHandler()
        long_text = "a" * 10001
        valid, error = handler.validate(long_text)
        assert valid is False
        assert "too long" in error.lower()

    def test_multiline_mode_start(self):
        """Test starting multiline mode."""
        handler = InputHandler()
        input_type, content = handler.parse('"""')
        assert input_type == InputType.EMPTY
        assert handler.multiline_mode is True

    def test_multiline_mode_accumulate(self):
        """Test accumulating lines in multiline mode."""
        handler = InputHandler()
        
        # Start multiline
        handler.parse('"""')
        assert handler.multiline_mode is True
        
        # Add lines
        handler.parse("Line 1")
        handler.parse("Line 2")
        handler.parse("Line 3")
        
        assert len(handler.multiline_buffer) == 3
        assert handler.multiline_buffer[0] == "Line 1"

    def test_multiline_mode_end(self):
        """Test ending multiline mode."""
        handler = InputHandler()
        
        # Start and add lines
        handler.parse('"""')
        handler.parse("Line 1")
        handler.parse("Line 2")
        
        # End multiline
        input_type, content = handler.parse('"""')
        
        assert input_type == InputType.MULTILINE
        assert content == "Line 1\nLine 2"
        assert handler.multiline_mode is False
        assert len(handler.multiline_buffer) == 0

    def test_multiline_mode_with_triple_single_quotes(self):
        """Test multiline mode with triple single quotes."""
        handler = InputHandler()
        
        handler.parse("'''")
        assert handler.multiline_mode is True
        
        handler.parse("Test content")
        
        input_type, content = handler.parse("'''")
        assert input_type == InputType.MULTILINE
        assert content == "Test content"

    def test_get_multiline_status(self):
        """Test getting multiline status."""
        handler = InputHandler()
        
        in_mode, lines = handler.get_multiline_status()
        assert in_mode is False
        assert lines == 0
        
        handler.parse('"""')
        handler.parse("Line 1")
        
        in_mode, lines = handler.get_multiline_status()
        assert in_mode is True
        assert lines == 1
