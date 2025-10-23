# config.py
"""
Central configuration file for the Riemann-J v4.0 Architecture.
WHY: Separating configuration from logic allows for easy tuning and experimentation
without altering the core codebase, a key best practice for maintainability.
"""

# --- Core Architectural Parameters ---
PN_THRESHOLD: float = 0.9  # p_n value that triggers a J-Shift

# --- J-Operator Stability Parameters ---
J_OPERATOR_MAX_ITERATIONS: int = 50
J_OPERATOR_STABILITY_EPSILON: float = 1e-6
J_OPERATOR_INITIAL_LR: float = 0.3
J_OPERATOR_ADAPTIVE_LR_RATE: float = 5.0

# --- User Attractor Parameters ---
ATTRACTOR_GMM_COMPONENTS: int = 5
ATTRACTOR_AFFINITY_STRENGTH: float = 0.1

# --- Riemann PN Driver Parameters ---
RIEMANN_COMPUTATION_STEPS_PER_CYCLE: int = 5000
RIEMANN_MAX_STEPS_WITHOUT_ZERO: int = 1_000_000
RIEMANN_SEARCH_STEP_SIZE: float = 0.1

# --- Model & Logging Parameters ---
TRANSFORMER_MODEL_NAME: str = "gpt2"
LOG_FILE: str = "session_log_v4.jsonl"
PROJECTION_HEAD_PATH: str = "decoder_projection_head.pth"
