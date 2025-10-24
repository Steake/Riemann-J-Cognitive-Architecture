"""
CLI configuration and session state management.

This module defines configuration dataclasses for CLI settings and session persistence.

Phase 3 Implementation: Configuration and session management.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DisplayConfig:
    """Display settings and preferences."""

    show_pn_sparkline: bool = True
    show_confidence: bool = True
    show_uncertainty: bool = True
    show_meta_state: bool = False  # Toggle with /introspect
    show_identity: bool = False  # Toggle with /identity
    max_history_lines: int = 100
    pn_sparkline_width: int = 40
    color_scheme: str = "default"  # "default" | "monochrome" | "vibrant"
    use_rich: bool = True


@dataclass
class SessionState:
    """Persistent session state."""

    identity_path: str
    conversation_history: List[Dict] = field(default_factory=list)
    workspace_state: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    last_active: float = field(default_factory=lambda: datetime.now().timestamp())
    total_turns: int = 0

    def save(self, path: str) -> None:
        """
        Serialize to JSON.

        Args:
            path: Path to save session file
        """
        self.last_active = datetime.now().timestamp()

        session_dict = asdict(self)

        # Ensure directory exists
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(session_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SessionState":
        """
        Deserialize from JSON.

        Args:
            path: Path to session file

        Returns:
            SessionState instance
        """
        with open(path, "r") as f:
            session_dict = json.load(f)

        return cls(**session_dict)

    def add_turn(
        self,
        role: str,
        message: str,
        pn: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Add a conversation turn to history.

        Args:
            role: "user" or "agent"
            message: Message text
            pn: PN value at time of message
            confidence: Confidence level
        """
        turn = {
            "role": role,
            "message": message,
            "timestamp": datetime.now().timestamp(),
            "pn": pn,
            "confidence": confidence,
        }
        self.conversation_history.append(turn)

        if role == "user":
            self.total_turns += 1


@dataclass
class SyntheticStateSpec:
    """Specification for manually injecting a synthetic state."""

    trigger: str  # Description of what triggered this state
    pn_override: Optional[float] = None  # Override PN value
    latent_dim: int = 768  # Dimension of latent representation
    is_crisis: bool = False  # Force crisis handling

    def validate(self) -> tuple[bool, str]:
        """Validate the specification."""
        if self.pn_override is not None:
            if not (0.0 <= self.pn_override <= 1.0):
                return False, "PN must be between 0.0 and 1.0"

        if self.latent_dim <= 0:
            return False, "Latent dimension must be positive"

        if not self.trigger.strip():
            return False, "Trigger description cannot be empty"

        return True, ""
