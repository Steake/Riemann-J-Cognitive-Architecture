"""
Input handler for CLI with advanced features.

This module provides input parsing, validation, and command detection
with support for multi-line input and command history.

Phase 3 Implementation: Advanced input handling.
"""

from enum import Enum
from typing import Tuple, Optional


class InputType(Enum):
    """Types of user input."""

    MESSAGE = "message"  # Regular conversation
    COMMAND = "command"  # Meta-command
    MULTILINE = "multiline"  # Multi-line message
    EMPTY = "empty"  # Empty input (ignore)


class InputHandler:
    """Handles user input parsing and validation."""

    def __init__(self):
        """Initialize input handler."""
        self.multiline_mode = False
        self.multiline_buffer = []

    def parse(self, raw_input: str) -> Tuple[InputType, str]:
        """
        Parse input into (type, content).

        Args:
            raw_input: Raw user input string

        Returns:
            (InputType, content): Tuple of input type and cleaned content
        """
        # Handle empty input
        if not raw_input or not raw_input.strip():
            return (InputType.EMPTY, "")

        # Check for command
        if self.is_command(raw_input):
            return (InputType.COMMAND, raw_input.strip())

        # Check for multi-line mode markers
        if raw_input.strip() == '"""' or raw_input.strip() == "'''":
            if not self.multiline_mode:
                # Start multi-line mode
                self.multiline_mode = True
                self.multiline_buffer = []
                return (InputType.EMPTY, "")
            else:
                # End multi-line mode
                self.multiline_mode = False
                content = "\n".join(self.multiline_buffer)
                self.multiline_buffer = []
                return (InputType.MULTILINE, content)

        # If in multi-line mode, accumulate
        if self.multiline_mode:
            self.multiline_buffer.append(raw_input)
            return (InputType.EMPTY, "")

        # Regular message
        return (InputType.MESSAGE, raw_input.strip())

    def is_command(self, text: str) -> bool:
        """
        Check if input is a meta-command.

        Args:
            text: Input text to check

        Returns:
            True if text is a command
        """
        return text.strip().startswith("/")

    def validate(self, text: str) -> Tuple[bool, str]:
        """
        Validate input.

        Args:
            text: Input text to validate

        Returns:
            (valid, error_msg): Tuple of validation result and error message
        """
        # Empty input is valid but will be ignored
        if not text or not text.strip():
            return (True, "")

        # Check for excessively long input
        if len(text) > 10000:
            return (False, "Input too long (max 10000 characters)")

        # All validation passed
        return (True, "")

    def get_multiline_status(self) -> Tuple[bool, int]:
        """
        Get current multi-line mode status.

        Returns:
            (in_multiline_mode, lines_buffered): Status tuple
        """
        return (self.multiline_mode, len(self.multiline_buffer))
