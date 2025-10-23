# API Reference

## Configuration Module (`config.py`)

### Core Architectural Parameters

#### `PN_THRESHOLD: float = 0.9`
The prediction error value that triggers a J-Shift event.
- **Type**: float
- **Range**: 0.0 to 1.0
- **Default**: 0.9
- **Purpose**: Controls sensitivity of J-Operator activation

### J-Operator Stability Parameters

#### `J_OPERATOR_MAX_ITERATIONS: int = 50`
Maximum number of iterations for J-Operator convergence.
- **Type**: int
- **Default**: 50
- **Purpose**: Prevents infinite loops in convergence

#### `J_OPERATOR_STABILITY_EPSILON: float = 1e-6`
Convergence threshold for state distance.
- **Type**: float
- **Default**: 0.000001
- **Purpose**: Defines when convergence is achieved

#### `J_OPERATOR_INITIAL_LR: float = 0.3`
Initial learning rate for adaptive convergence.
- **Type**: float
- **Default**: 0.3
- **Purpose**: Starting point for learning rate schedule

#### `J_OPERATOR_ADAPTIVE_LR_RATE: float = 5.0`
Adaptation rate for learning rate adjustment.
- **Type**: float
- **Default**: 5.0
- **Purpose**: Controls how quickly LR decreases with distance

### User Attractor Parameters

#### `ATTRACTOR_GMM_COMPONENTS: int = 5`
Number of Gaussian components in user GMM.
- **Type**: int
- **Default**: 5
- **Purpose**: Determines multi-modality of user model

#### `ATTRACTOR_AFFINITY_STRENGTH: float = 0.1`
Strength of affinity pull toward user centroids.
- **Type**: float
- **Default**: 0.1
- **Range**: 0.0 (no effect) to 1.0 (full pull)

### Riemann PN Driver Parameters

#### `RIEMANN_COMPUTATION_STEPS_PER_CYCLE: int = 5000`
Number of computational steps added per PN cycle.
- **Type**: int
- **Default**: 5000
- **Purpose**: Controls PN accumulation rate

#### `RIEMANN_MAX_STEPS_WITHOUT_ZERO: int = 1_000_000`
Maximum steps before PN saturates.
- **Type**: int
- **Default**: 1,000,000
- **Purpose**: Sets upper bound for sigmoid calculation

#### `RIEMANN_SEARCH_STEP_SIZE: float = 0.1`
Step size for t-value increment in Zeta search.
- **Type**: float
- **Default**: 0.1
- **Purpose**: Simulates progression along critical line

### Model & Logging Parameters

#### `TRANSFORMER_MODEL_NAME: str = "gpt2"`
Hugging Face model identifier.
- **Type**: str
- **Default**: "gpt2"
- **Options**: Any compatible causal LM

#### `LOG_FILE: str = "session_log_v4.jsonl"`
Path to session log file.
- **Type**: str
- **Default**: "session_log_v4.jsonl"
- **Format**: JSON Lines

#### `PROJECTION_HEAD_PATH: str = "decoder_projection_head.pth"`
Path to save/load projection head weights.
- **Type**: str
- **Default**: "decoder_projection_head.pth"
- **Format**: PyTorch state dict

---

## PN Driver Module (`pn_driver.py`)

### `PredictionErrorSignal`

**Dataclass** representing a PN signal.

#### Fields
- `timestamp: float` - Unix timestamp of creation
- `source: str` - Source identifier (e.g., "PNDriver_RiemannZeta")
- `priority: int` - Queue priority (lower = higher priority)
- `p_n: float` - Prediction error value (0.0 to 1.0)
- `uncompressed_data: dict` - Additional metadata

#### Example
```python
signal = PredictionErrorSignal(
    timestamp=time.time(),
    source="PNDriver_RiemannZeta",
    priority=10,
    p_n=0.95,
    uncompressed_data={'t': 14.5, 'steps_since_zero': 950000}
)
```

### `PNDriverRiemannZeta`

**Thread class** that generates continuous prediction error.

#### Constructor
```python
def __init__(self):
```
- Initializes with `current_t = 14.1347` (first non-trivial zero)
- Sets `steps_since_last_zero = 0`
- Configured as daemon thread

#### Methods

##### `_calculate_pn() -> float`
Calculates current prediction error using sigmoid function.

**Returns**: float in range [0.0, 1.0]

**Algorithm**:
```python
x = (self.steps_since_last_zero / RIEMANN_MAX_STEPS_WITHOUT_ZERO) * 12 - 6
p_n = 1 / (1 + math.exp(-x))
```

##### `run() -> None`
Main thread loop. Runs perpetually until `is_running = False`.

**Behavior**:
1. Increments `current_t` by `RIEMANN_SEARCH_STEP_SIZE`
2. Adds `RIEMANN_COMPUTATION_STEPS_PER_CYCLE` to `steps_since_last_zero`
3. Simulates zero discovery (5% probability)
4. Calculates PN and creates signal
5. Adds signal to `global_workspace` (if queue size < 100)
6. Sleeps for 0.1 seconds

---

## Architecture Module (`architecture.py`)

### `SyntheticState`

**Dataclass** representing system's internal state.

#### Fields
- `timestamp: float` - Unix timestamp
- `latent_representation: np.ndarray` - 768-dim state vector
- `source_trigger: str` - "USER_INPUT" or "RIEMANN_PN"
- `p_n_at_creation: float` - PN value at creation time
- `is_j_shift_product: bool` - True if created by J-Operator
- `status: str` - "OK", "CONVERGED", or "ITER_LIMIT_EXCEEDED"
- `analysis: dict` - Metadata (Lyapunov exp, iterations, etc.)

#### Example
```python
state = SyntheticState(
    timestamp=time.time(),
    latent_representation=np.random.randn(768),
    source_trigger="USER_INPUT",
    p_n_at_creation=0.3,
    is_j_shift_product=False,
    status="OK",
    analysis={}
)
```

### `DecoderProjectionHead`

**Neural network module** for state-to-logits projection.

#### Constructor
```python
def __init__(self, hidden_size: int, output_size: int):
```
- `hidden_size`: Dimension of input state (typically 768)
- `output_size`: Vocabulary size (typically 50,257 for GPT-2)

#### Methods

##### `forward(hidden_state: torch.Tensor) -> torch.Tensor`
Forward pass through projection network.

**Args**:
- `hidden_state`: Tensor of shape (hidden_size,)

**Returns**: Tensor of shape (output_size,) representing logit bias

### `SymbolicInterface`

**Class** managing encoding and decoding between text and state.

#### Constructor
```python
def __init__(self):
```
Initializes projection head and sets to eval mode.

#### Methods

##### `encoder(text: str) -> np.ndarray`
**Static method** encoding text to latent state vector.

**Args**:
- `text`: Input text string

**Returns**: numpy array of shape (768,) - last hidden state

**Process**:
1. Tokenize text
2. Pass through language model
3. Extract final layer, final token hidden state
4. Convert to numpy and return

##### `decoder(state_vector: np.ndarray) -> str`
Decodes state vector to text using biased generation.

**Args**:
- `state_vector`: numpy array of shape (768,)

**Returns**: Generated text string

**Process**:
1. Convert state to tensor
2. Generate logit bias via projection head
3. Create custom LogitsProcessor
4. Generate with bias (max 50 tokens)
5. Decode and return

### `UserAttractor`

**Class** managing per-user state history and affinity.

#### Constructor
```python
def __init__(self, user_id: str):
```
- `user_id`: Unique identifier for user
- Initializes empty `state_history`
- Creates GMM with 5 components

#### Methods

##### `add_state(state: np.ndarray) -> None`
Adds state to history and updates GMM.

**Args**:
- `state`: numpy array of shape (768,)

**Behavior**:
- Appends to `state_history`
- Updates GMM every 10 states

##### `apply_affinity(state: np.ndarray) -> np.ndarray`
Applies user-specific affinity to input state.

**Args**:
- `state`: numpy array of shape (768,)

**Returns**: Modified state pulled toward user centroids

**Algorithm**:
```python
cluster_index = gmm.predict(state)
centroid = gmm.means_[cluster_index]
influence = (centroid - state) * ATTRACTOR_AFFINITY_STRENGTH
return state + influence
```

### `CognitiveWorkspace`

**Main class** orchestrating cognitive operations.

#### Constructor
```python
def __init__(self):
```
- Creates `SymbolicInterface`
- Initializes empty `user_attractors` dict
- Opens log file for appending

#### Methods

##### `get_or_create_user(user_id: str) -> UserAttractor`
Retrieves or creates UserAttractor for given ID.

**Args**:
- `user_id`: User identifier

**Returns**: UserAttractor instance

##### `process_user_input(user_id: str, text: str) -> tuple[str, SyntheticState]`
Processes user input through full pipeline.

**Args**:
- `user_id`: User identifier
- `text`: Input text

**Returns**: Tuple of (response_text, state_object)

**Process**:
1. Get/create user attractor
2. Encode text to state
3. Apply affinity transformation
4. Add to user history
5. Generate response via decoder
6. Create SyntheticState object
7. Return response and state

##### `log_state(state: SyntheticState) -> None`
Logs state to JSONL file.

**Args**:
- `state`: SyntheticState to log

**Format**:
```json
{
  "timestamp": 1234567890.123,
  "latent_representation": {"mean": 0.05, "std": 0.95},
  "source_trigger": "USER_INPUT",
  "p_n_at_creation": 0.3,
  "is_j_shift_product": false,
  "status": "OK",
  "analysis": {}
}
```

##### `close() -> None`
Closes log file. Call on shutdown.

---

## TUI Module (`tui.py`)

### `TUI`

**Textual App class** providing interactive interface.

#### Class Attributes
- `TITLE = "Riemann-J Cognitive Architecture v4.0"`
- `CSS_PATH = "tui.css"`

#### Reactive Variables
- `current_user: str` - Active user ID
- `current_pn: float` - Latest PN value
- `attractor_size: int` - Current user's history size
- `workspace_q_size: int` - Global workspace queue size
- `system_status: str` - Status text
- `status_style: str` - Status color

#### Constructor
```python
def __init__(self):
```
- Creates `CognitiveWorkspace`
- Starts `PNDriverRiemannZeta` thread

#### Methods

##### `compose() -> ComposeResult`
Defines widget layout.

**Widgets**:
- Header
- Dashboard container
  - Sparkline (PN history)
  - Status display
- Conversation log
- Input field
- Footer

##### `on_mount() -> None`
Called when app starts.

**Setup**:
- Focuses input field
- Sets update interval (4 Hz)
- Sets queue check interval (10 Hz)

##### `update_dashboard() -> None`
Periodic update of dashboard metrics.

**Updates**:
- Peeks at workspace queue for latest PN
- Updates sparkline data
- Calculates system status
- Updates status display

##### `check_workspace_queue() -> None`
Checks for high-priority J-Shift signals.

**Behavior**:
- Dequeues messages from workspace
- Triggers J-Shift if PN > threshold
- Displays J-Shift responses

##### `on_input_submitted(event: Input.Submitted) -> None`
Handles user input submission.

**Commands**:
- `/exit` - Shutdown application
- `/switch <user>` - Switch active user
- Text input - Process as user message

##### `run_user_processing(user_input: str) -> None`
Processes user input in separate thread.

**Process**:
1. Call `workspace.process_user_input()`
2. Log resulting state
3. Update conversation log with response

##### `main() -> None`
**Entry point function** for command-line execution.

```python
def main():
    app = TUI()
    app.run()
```

---

## Shared Resources Module (`shared_resources.py`)

### Global Variables

#### `global_workspace: queue.PriorityQueue`
Thread-safe priority queue for inter-thread communication.

**Priority**: Lower number = higher priority
**Usage**: 
```python
priority = int((1 - p_n) * 100)
global_workspace.put((priority, signal))
```

#### `tokenizer: AutoTokenizer`
Hugging Face tokenizer for GPT-2.
- Loaded from `TRANSFORMER_MODEL_NAME`
- Pad token set to EOS token if undefined

#### `model: AutoModelForCausalLM`
Hugging Face GPT-2 model.
- Set to eval() mode
- Moved to appropriate device (CUDA/CPU)

#### `device: str`
Execution device: "cuda" or "cpu"
- Automatically detected based on availability

---

## Usage Examples

### Basic Initialization
```python
from riemann_j.tui import TUI

app = TUI()
app.run()
```

### Programmatic Access
```python
from riemann_j import CognitiveWorkspace

workspace = CognitiveWorkspace()
response, state = workspace.process_user_input("user1", "Hello!")
print(f"Response: {response}")
print(f"State: {state.status}")
workspace.close()
```

### Custom Configuration
```python
from riemann_j import config

# Modify before importing other modules
config.PN_THRESHOLD = 0.85
config.ATTRACTOR_GMM_COMPONENTS = 3

from riemann_j import CognitiveWorkspace
# Will use modified config
```

### Running Tests
```bash
# Unit tests
pytest tests/unit/

# BDD tests
pytest tests/bdd/

# All tests with coverage
pytest tests/ --cov=src/riemann_j --cov-report=html
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-23  
**Status**: Production
