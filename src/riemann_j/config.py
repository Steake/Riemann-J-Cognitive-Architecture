# config.py
"""
Central configuration file for the Riemann-J v4.0 Architecture.
WHY: Separating configuration from logic allows for easy tuning and experimentation
without altering the core codebase, a key best practice for maintainability.
"""

# --- Core Architectural Parameters ---
PN_THRESHOLD: float = 0.9  # p_n value that triggers a J-Shift

# --- J-Operator Stability Parameters ---
J_OPERATOR_MAX_ITERATIONS: int = 100
J_OPERATOR_STABILITY_EPSILON: float = 1e-6  # Absolute threshold (backup)
J_OPERATOR_RELATIVE_EPSILON: float = 0.01  # Relative threshold: 1% of state magnitude
J_OPERATOR_INITIAL_LR: float = 0.3
J_OPERATOR_ADAPTIVE_LR_RATE: float = 5.0
J_OPERATOR_LYAPUNOV_WINDOW: int = 10  # Check stability over last N iterations
J_OPERATOR_LYAPUNOV_THRESHOLD: float = -1.0  # Negative = converging

# --- User Attractor Parameters ---
ATTRACTOR_GMM_COMPONENTS: int = 5
ATTRACTOR_AFFINITY_STRENGTH: float = 0.1

# --- Riemann PN Driver Parameters ---
RIEMANN_COMPUTATION_STEPS_PER_CYCLE: int = 15000  # Moderate accumulation for demo
RIEMANN_MAX_STEPS_WITHOUT_ZERO: int = 1_000_000
RIEMANN_SEARCH_STEP_SIZE: float = 0.1

# --- Model & Logging Parameters ---
# Default: Qwen3Guard-Gen-0.6B (600M, Jan 2025, latest from Qwen)
# Alternative: "HuggingFaceTB/SmolLM2-135M-Instruct" (135M), "gpt2" (124M)
# Override with RIEMANN_MODEL environment variable for testing
TRANSFORMER_MODEL_NAME: str = "Qwen/Qwen3Guard-Gen-0.6B"
LOG_FILE: str = "session_log_v4.jsonl"
PROJECTION_HEAD_PATH: str = "decoder_projection_head.pth"
