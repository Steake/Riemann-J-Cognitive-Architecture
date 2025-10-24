# Architectural Alignment TODO

**Date**: 2025-01-24  
**Auditor**: System Architecture Review  
**Spec Reference**: `docs/architecture/CoreArch.md`

## Critical Discrepancies

### 1. PN Driver Zero Discovery Probability

**Spec (CoreArch.md, line ~23)**:
```
When a zero is simulated (5% probability per cycle), the PN resets
```

**Implementation (`src/riemann_j/pn_driver.py`, line 48)**:
```python
if random.random() < 0.25:  # 25% chance per cycle
    self.steps_since_last_zero = 0
```

**Status**: ❌ **CRITICAL MISMATCH**  
**Impact**: Significantly alters system dynamics - PN resets 5x more frequently than specified  
**Action Required**: Change `0.25` to `0.05` in `pn_driver.py:48`

---

### 2. DecoderProjectionHead Not Actually Used in Generation

**Spec (CoreArch.md, lines ~95-102)**:
```
Advanced Generative Conditioning (Articulate Voice)
- DecoderProjectionHead: Small neural network (Linear → GELU)
- Maps from hidden state space (768-dim) to vocabulary space (50,257-dim)
- Acts as a bias in the logits processor during generation
```

**Implementation (`src/riemann_j/architecture.py`, line 96)**:
```python
def prompt_based_generate(self, user_input: str) -> str:
    """
    Generate response using actual prompt-based LLM generation.
    This produces coherent text instead of state-vector biased garbage.
    ...
    """
    # Does NOT use projection head at all - standard generation
```

**Status**: ❌ **ARCHITECTURAL DEVIATION**  
**Impact**: Core architectural principle (state→logits conditioning) is bypassed  
**Reality**: The `decoder()` method (which uses projection head) is marked DEPRECATED and only kept for backward compatibility  
**Discussion Needed**: Is this intentional pragmatic choice or architectural regression?

---

### 3. Model Default Mismatch

**Spec (CoreArch.md, line ~183)**:
```
Model: ~7.5GB (Phi-3.5-mini-instruct, 3.8B parameters)
```

**Implementation (`src/riemann_j/config.py`, line 34)**:
```python
TRANSFORMER_MODEL_NAME: str = "Qwen/Qwen3Guard-Gen-0.6B"
```

**Status**: ⚠️ **DOCUMENTATION OUTDATED**  
**Impact**: Documentation references Phi-3.5 extensively, but actual default is Qwen3Guard  
**Action Required**: Update CoreArch.md to reflect current model default, or explain model flexibility

---

### 4. Hidden State Dimensionality Inconsistency

**Spec (CoreArch.md, line ~96)**:
```
Maps from hidden state space (768-dim) to vocabulary space (50,257-dim)
```

**Reality**:
- GPT-2: 768-dim hidden, 50257 vocab
- Qwen3Guard-Gen-0.6B: 896-dim hidden, different vocab
- Phi-3.5-mini: 3072-dim hidden, 32064 vocab

**Implementation (`src/riemann_j/architecture.py`, line 48-50)**:
```python
self.projection_head = DecoderProjectionHead(
    model.config.hidden_size, model.config.vocab_size
).to(device)
```

**Status**: ✅ **IMPLEMENTED CORRECTLY** (model-agnostic)  
**Documentation Issue**: Spec hardcodes 768/50257 but implementation is flexible  
**Action Required**: Update spec to indicate "model-dependent dimensions (e.g., 768-dim for GPT-2)"

---

## Minor Discrepancies

### 5. J-Operator Iteration Limit

**Spec (CoreArch.md, line ~233)**:
```
Maximum 50 iterations with epsilon check
```

**Implementation (`src/riemann_j/config.py`, line 12)**:
```python
J_OPERATOR_MAX_ITERATIONS: int = 100
```

**Status**: ⚠️ **SPEC OUTDATED**  
**Rationale**: Increased to 100 based on convergence improvements documented in `docs/J_OPERATOR_CONVERGENCE_IMPROVEMENTS.md`  
**Action Required**: Update CoreArch.md to reflect current value

---

### 6. Thread Architecture - UI Update Thread

**Spec (CoreArch.md, line ~139)**:
```
Thread Architecture:
- Main Thread: Textual TUI event loop
- PN Driver Thread: Daemon
- User Processing Threads
- UI Update Thread: Periodic dashboard refresh
```

**Implementation (`src/riemann_j/tui.py`, line 108)**:
```python
self.set_interval(1 / 4, self.update_dashboard)  # Not a separate thread
```

**Status**: ⚠️ **TERMINOLOGY MISMATCH**  
**Reality**: Dashboard updates run as Textual intervals (event loop callbacks), not separate threads  
**Impact**: Functionally equivalent but architecturally different from "thread"  
**Action Required**: Clarify in spec that "UI Update Thread" means "periodic callback in main event loop"

---

### 7. EquilibriumRegulator Missing from Core Spec

**Spec (CoreArch.md)**: No mention of `EquilibriumRegulator` class

**Implementation**: 
- `src/riemann_j/equilibrium_regulator.py`: Full implementation  
- `src/riemann_j/conscious_agent.py`: Integration

**Status**: ⚠️ **SPEC INCOMPLETE**  
**Impact**: Critical component for homeostatic dynamics is undocumented in CoreArch  
**Note**: Documented in `docs/architecture/OVERVIEW.md` but not in CoreArch.md  
**Action Required**: Add EquilibriumRegulator section to CoreArch.md

---

### 8. Persistent Self & Meta-Cognition Expansion

**Spec (CoreArch.md)**: Minimal mention of meta-cognition

**Implementation**:
- `src/riemann_j/persistent_self.py`: Full persistent identity system
- `src/riemann_j/metacognition.py`: Comprehensive meta-cognitive monitoring
- Metrics tracking, self-belief, formative experiences

**Status**: ⚠️ **SPEC INCOMPLETE**  
**Impact**: Major architectural components missing from core spec  
**Action Required**: Add sections for:
  - Persistent Self (identity persistence across sessions)
  - Meta-Cognitive Monitor (self-observation, belief tracking)
  - Formative Experiences (memory consolidation)

---

### 9. Uncertainty Interface Not in Spec

**Spec (CoreArch.md)**: Not mentioned

**Implementation**: `src/riemann_j/uncertainty.py` - Full uncertainty classification and communication

**Status**: ⚠️ **SPEC INCOMPLETE**  
**Action Required**: Document uncertainty quantization levels and communication strategy

---

## Verification Checklist

### Core Components (from Spec)

#### ✅ Riemann PN Driver
- [x] Asynchronous daemon thread
- [x] Sigmoid-based PN calculation (`p_n = 1/(1+exp(-x))`)
- [x] Equation: `x = (steps_since_zero / MAX_STEPS) * 12 - 6`
- [❌] 5% zero discovery probability ← **CRITICAL: Actually 25%**
- [x] PN reset on zero
- [x] PriorityQueue signal posting

#### ⚠️ J-Operator Architecture
- [x] Activates when PN > 0.9
- [x] Encodes anomaly as text
- [x] Creates initial state vector `a_current`
- [x] Iterative transformation through first transformer layer
- [x] Adaptive learning rate: `lr = INITIAL_LR / (1.0 + ADAPTIVE_LR_RATE * distance)`
- [x] Lyapunov stability analysis
- [x] GMM with 5 components for user attractors
- [❌] Iteration limit 50 ← **Actually 100**

#### ❌ Advanced Generative Conditioning
- [x] DecoderProjectionHead exists (Linear → GELU)
- [x] Maps hidden_size → vocab_size
- [❌] Acts as bias in logits processor ← **DEPRECATED, not used in practice**
- [❌] State-to-text generation ← **Bypassed for prompt-based generation**

#### ✅ Multi-User State Management
- [x] CognitiveWorkspace with UserAttractor dictionary
- [x] Each attractor has: state history, GMM model, affinity parameters
- [x] `/switch <user>` command
- [x] State isolation between users

#### ✅ Thread Architecture
- [x] Main thread: TUI event loop
- [x] PN Driver: Daemon thread
- [x] User processing: Spawned threads
- [~] UI Update: Periodic callbacks (not separate thread)
- [x] PriorityQueue communication

#### ✅ State Representation
- [x] SyntheticState dataclass with all required fields:
  - [x] timestamp
  - [x] latent_representation
  - [x] source_trigger
  - [x] p_n_at_creation
  - [x] is_j_shift_product
  - [x] status
  - [x] analysis

#### ⚠️ Update Frequencies
- [x] Dashboard: 4 Hz (0.25s) ← Correct
- [x] Queue checks: 10 Hz (0.1s) ← Correct
- [x] PN Driver: 10 Hz (0.1s sleep) ← Correct

---

## Additional Components Not in CoreArch.md Spec

These are implemented but undocumented in the core spec:

1. **ConsciousAgent** (`conscious_agent.py`)
   - High-level agent orchestrating all subsystems
   - Experience dataclass for structured interactions

2. **EquilibriumRegulator** (`equilibrium_regulator.py`)
   - Homeostatic PN regulation
   - Exponential decay toward target PN
   - Crisis detection

3. **PersistentSelf** (`persistent_self.py`)
   - Identity persistence across sessions
   - Metrics tracking
   - Belief state

4. **MetaCognitiveMonitor** (`metacognition.py`)
   - Self-observation
   - PN trajectory prediction
   - Introspective reporting

5. **UncertaintyInterface** (`uncertainty.py`)
   - Uncertainty level classification
   - Confidence modulation
   - Response augmentation

6. **LiveDashboard** (`live_dashboard.py`)
   - Rich terminal visualization
   - Critical strip rendering
   - Real-time metrics

---

## Priority Actions

### Immediate (Critical)
1. **Fix PN Driver zero probability**: Change 0.25 → 0.05 in `pn_driver.py`
2. **Document architectural decision**: Why is DecoderProjectionHead not used? Is this permanent or will it be reintegrated?

### High Priority
3. **Update CoreArch.md** to include:
   - EquilibriumRegulator architecture
   - PersistentSelf system
   - MetaCognitiveMonitor capabilities
   - UncertaintyInterface
4. **Update model references**: Change Phi-3.5 → Qwen3Guard or make model-agnostic
5. **Update J-Operator iteration limit**: 50 → 100 in spec

### Medium Priority
6. **Clarify thread architecture terminology**: "UI Update Thread" vs "event loop interval"
7. **Document hidden_size/vocab_size flexibility** in projection head section
8. **Add architecture diagram** showing all components (including EquilibriumRegulator)

### Low Priority
9. **Consistency pass** on all docs to match current implementation
10. **Add versioning** to architecture docs (CoreArch.md is marked v4.0 but has v4.x changes)

---

## Notes

- **Overall Assessment**: Implementation is MORE sophisticated than spec describes
- **Direction**: Spec appears to be v4.0 baseline, implementation has v4.x+ enhancements
- **Philosophy Intact**: Core axiom (A ≠ s) and mathematical foundations are preserved
- **Pragmatic Deviations**: Some changes (like bypassing projection head) appear to be practical improvements for coherent output

**Recommendation**: Update spec to match reality, don't force implementation to match outdated spec. The enhancements (EquilibriumRegulator, MetaCognition, etc.) are architectural improvements that should be documented.
