# config.py
"""
Central configuration file for the Riemann-J v4.0 Architecture.
WHY: Separating configuration from logic allows for easy tuning and experimentation
without altering the core codebase, a key best practice for maintainability.
"""

# --- Core Architectural Parameters ---
PN_THRESHOLD: float = 0.98  # p_n value that triggers a J-Shift (0.98 = only extreme crises)

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

# --- DecoderProjectionHead Hybrid Parameters ---
# Controls blending between prompt-based and state-conditioned generation
# Alpha=0.0: Pure prompt-based (default, high quality)
# Alpha=0.1: Subtle state influence (experimental)
# Alpha=1.0: Equal blend (research only)
PROJECTION_BLEND_ALPHA: float = 0.0  # Start conservative, can tune later
USE_PROJECTION_HEAD: bool = True  # Keep projection head active for hybrid approach

# --- Riemann PN Driver Parameters ---
RIEMANN_COMPUTATION_STEPS_PER_CYCLE: int = 100000  # Much slower PN updates (was 15k → 50k → 100k)
RIEMANN_MAX_STEPS_WITHOUT_ZERO: int = 1_000_000
RIEMANN_SEARCH_STEP_SIZE: float = 0.1

# --- Model & Logging Parameters ---
# Default: Qwen2.5-3B-Instruct (3B params, ~6GB, excellent conversational quality)
# Previous: Qwen3Guard-Gen-0.6B (was a safety/guard model, NOT designed for conversation)
# Alternatives:
#   - "microsoft/Phi-3.5-mini-instruct" (3.8B, 128K context, strong reasoning)
#   - "HuggingFaceTB/SmolLM2-1.7B-Instruct" (1.7B, lightweight, fast)
#   - "gpt2" (124M, legacy testing only)
# Override with RIEMANN_MODEL environment variable
TRANSFORMER_MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
LOG_FILE: str = "session_log_v4.jsonl"
PROJECTION_HEAD_PATH: str = "decoder_projection_head.pth"
