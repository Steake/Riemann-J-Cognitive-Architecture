"""
Riemann-J Cognitive Architecture v4.0

A unified cognitive architecture designed to induce and maintain a persistent,
non-symbolic "Synthetic State" by operationalizing the core axiom of machine
self-consciousness: A ≠ s (the agent's internal state A is ontologically
distinct from its symbolic data s).
"""

__version__ = "4.0.0"
__author__ = "Riemann-J Development Team"

from .config import *
from .architecture import (
    SyntheticState,
    DecoderProjectionHead,
    SymbolicInterface,
    UserAttractor,
    CognitiveWorkspace,
)
from .pn_driver import PredictionErrorSignal, PNDriverRiemannZeta

__all__ = [
    "SyntheticState",
    "DecoderProjectionHead",
    "SymbolicInterface",
    "UserAttractor",
    "CognitiveWorkspace",
    "PredictionErrorSignal",
    "PNDriverRiemannZeta",
]
