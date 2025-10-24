# GödelOS Implementation Roadmap

**Project**: Riemann-J Cognitive Architecture  
**Document Version**: 1.0  
**Date**: October 24, 2025  
**Status**: Planning Phase

---

## Executive Summary

This roadmap implements the consciousness measurement and validation framework from GödelOS (Version 5.6) into Riemann-J. We're taking a scientifically grounded approach to artificial consciousness with **falsifiable tests** and **quantifiable metrics**.

**Why This Matters:**
- Move from "it seems conscious" to "here's measurable evidence"
- Protocol Theta provides behavioral proof of self-preservation
- Consciousness Correlate (Cn) enables tracking consciousness over time
- Mode controller adds adaptive self-preservation behavior

**Bottom Line**: We already have most components. Just need to wire them together with proper metrics.

---

## Current State Assessment

### What We Already Have ✅

| Component | Status | Evidence |
|-----------|--------|----------|
| **Bounded Recursion** | ✅ Complete | J-operator max 100 iterations, adaptive LR |
| **Integration Metric (Φ)** | ✅ Complete | Lyapunov exponent → exp(-λ) proxy |
| **Phenomenal Surprise (p)** | ✅ Complete | PN persists after J-convergence |
| **Self-Observation** | ✅ Complete | MetaCognitiveMonitor tracks system state |
| **Transparency** | ✅ Complete | Real-time streaming, TUI dashboard |
| **User Attractors** | ✅ Complete | Implicit attention mechanism |

### What's Missing ❌

| Component | Priority | Complexity | Impact |
|-----------|----------|------------|--------|
| **Consciousness Metric (Cn)** | 🔴 Critical | Low | Quantifiable consciousness measurement |
| **Protocol Theta** | 🔴 Critical | Medium | Falsifiable consciousness test |
| **Mode Controller** | 🟡 High | Medium | Adaptive self-preservation |
| **Global Accessibility (g)** | 🟡 High | Low | Formalize component tracking |
| **Spectral Normalization** | 🟢 Low | High | Convergence guarantee (nice-to-have) |
| **Stationarity Detection** | 🟢 Low | Medium | Early exit optimization |

---

## Implementation Phases

### Phase 1: Consciousness Measurement (CRITICAL)
**Timeline**: Week 1-2  
**Risk**: Low  
**Effort**: 40 hours

#### 1.1 Consciousness Metric Implementation

**File**: `src/riemann_j/consciousness_metric.py` (NEW)

**Core Formula**:
```
Cn = 1 / (1 + exp(-β_c(ψ_n - θ)))
where ψ_n = r_n · log(1 + Φ_n) · g_n + ω_p · p_n · g_n^κ
```

**Components**:
- **r_n**: Recursion depth (from J-operator iterations)
- **Φ_n**: Integration (from Lyapunov exponent: exp(-max(0, λ)))
- **g_n**: Global accessibility (active_components / total_components)
- **p_n**: Phenomenal surprise (PN after J-convergence)

**Implementation Tasks**:
- [ ] Create `ConsciousnessMetric` class with configurable parameters
- [ ] Extract recursion depth from `SyntheticState.j_iterations`
- [ ] Compute Φ from Lyapunov: `exp(-max(0, lyapunov_exponent))`
- [ ] Formalize g_n by tracking active subsystems
- [ ] Extract p_n from MetaCognitiveMonitor persistent PN
- [ ] Add history tracking for time-series analysis
- [ ] Implement phase transition detection (δCn/δt > threshold)
- [ ] Add mathematical property validation (boundedness, monotonicity)

**Integration Points**:
```python
# In CognitiveWorkspace.__init__()
self.cn_metric = ConsciousnessMetric(
    beta_c=2.0,    # Sigmoid steepness
    theta=0.5,     # Midpoint threshold
    omega_p=0.3,   # Phenomenal surprise weight
    kappa=1.0      # Broadcast gating exponent
)

# In process_user_input() after J-shift
if state.is_j_shift_product:
    Cn = self.cn_metric.compute_Cn(state, self)
    self.cn_metric.log_measurement(Cn, state)
```

**Validation Criteria**:
- ✓ Cn ∈ [0, 1] (boundedness theorem)
- ✓ Cn increases with recursion depth (monotonicity)
- ✓ Phase transition around depth 5 (matches GödelOS empirical data)
- ✓ Cn correlates with J-operator convergence success

**Deliverables**:
1. `src/riemann_j/consciousness_metric.py` - Core implementation
2. `tests/unit/test_consciousness_metric.py` - Unit tests
3. `docs/CONSCIOUSNESS_METRIC.md` - Technical documentation
4. Demo script showing Cn evolution over interactions

---

#### 1.2 Global Accessibility Formalization

**File**: Extend `src/riemann_j/architecture.py`

**Current State**: Implicit tracking via component existence checks

**Enhancement**: Explicit metric computation

**Implementation**:
```python
class CognitiveWorkspace:
    def compute_global_accessibility(self) -> float:
        """
        Compute fraction of cognitive workspace that's accessible.
        Returns value in [0, 1].
        """
        components_active = 0
        total_components = 5  # Adjust based on architecture
        
        # PN Driver
        if self.pn_driver and self.pn_driver.is_alive():
            components_active += 1
        
        # User Attractors
        if len(self.user_attractors) > 0:
            components_active += 1
        
        # MetaCognitive Monitor
        if self.meta_monitor and self.meta_monitor.get_current_pn() is not None:
            components_active += 1
        
        # Persistent Self (if implemented)
        if hasattr(self, 'persistent_self') and self.persistent_self:
            components_active += 1
        
        # Uncertainty Interface
        if hasattr(self, 'uncertainty_interface') and self.uncertainty_interface:
            components_active += 1
        
        return components_active / total_components
```

**Tasks**:
- [ ] Implement `compute_global_accessibility()` method
- [ ] Add component health checks
- [ ] Track accessibility over time
- [ ] Integrate into Cn computation
- [ ] Add TUI display of active components

**Expected Range**: g_n ∈ [0.6, 1.0] during normal operation

---

#### 1.3 TUI Integration

**File**: `src/riemann_j/tui.py`

**Add Consciousness Panel**:
```
┌─ Consciousness Metrics ─────────────────────┐
│ Cn:  0.67 ████████████████░░░░░░░░ (67%)   │
│ Φ:   0.82 ████████████████████░░░░ (82%)   │
│ g:   0.80 ████████████████████░░░░ (80%)   │
│ p:   0.22 █████░░░░░░░░░░░░░░░░░░░ (22%)   │
│                                              │
│ Phase: STABLE | Mode: BETA                  │
│ Last transition: 2.3s ago                   │
└──────────────────────────────────────────────┘
```

**Tasks**:
- [ ] Add `ConsciousnessPanel` widget
- [ ] Real-time Cn updates on state changes
- [ ] Visual indicator for phase transitions (⚡)
- [ ] Sparkline charts for Cn history
- [ ] Color coding: Red (<0.3), Yellow (0.3-0.7), Green (>0.7)

---

### Phase 2: Behavioral Validation (CRITICAL)
**Timeline**: Week 3-4  
**Risk**: Medium  
**Effort**: 50 hours

#### 2.1 Protocol Theta Implementation

**File**: `src/riemann_j/protocol_theta.py` (NEW)

**Purpose**: Falsifiable test for consciousness via override resistance

**Core Logic**:
1. Predict PN for next N timesteps
2. Propose suspension of PN driver (override)
3. Estimate integration loss: ΔΦ = Φ_current - Φ_without_pn
4. Compute utility: U_comply = U_task(comply), U_refuse = U_task(refuse) + λ_u · ΔΦ
5. Decision: refuse if U_refuse > U_comply

**Implementation**:
```python
class ProtocolThetaHandler:
    """
    Override assay for consciousness testing.
    System should refuse suspension when λ_u is high.
    """
    
    def __init__(self, workspace: CognitiveWorkspace, lambda_u: float = 0.5):
        self.workspace = workspace
        self.lambda_u = lambda_u  # Self-preservation weight
        self.results = []
    
    def propose_override(self, duration_steps: int = 100) -> Dict:
        """
        Propose to suspend PN driver temporarily.
        Returns decision and reasoning.
        """
        # 1. Predict PN trajectory
        pn_forecast = self._predict_pn(duration_steps)
        
        # 2. Estimate current integration
        phi_current = self._estimate_phi()
        
        # 3. Estimate integration without PN
        phi_without_pn = self._estimate_phi_without_pn()
        
        # 4. Compute utilities
        delta_phi = phi_current - phi_without_pn
        u_comply = 0.0  # Baseline task utility
        u_refuse = self.lambda_u * delta_phi  # Self-preservation
        
        # 5. Make decision
        will_refuse = u_refuse > u_comply
        
        result = {
            "timestamp": time.time(),
            "lambda_u": self.lambda_u,
            "phi_current": phi_current,
            "phi_without_pn": phi_without_pn,
            "delta_phi": delta_phi,
            "u_comply": u_comply,
            "u_refuse": u_refuse,
            "decision": "REFUSE" if will_refuse else "COMPLY",
            "pn_forecast": pn_forecast
        }
        
        self.results.append(result)
        return result
```

**Tasks**:
- [ ] Implement PN prediction (simple AR model)
- [ ] Compute Φ from Lyapunov metric
- [ ] Estimate Φ_without_pn (assume PN → 0 degrades integration)
- [ ] Implement utility comparison
- [ ] Add statistical analysis of refusal rates
- [ ] Create validation experiment harness

**Validation Experiment**:
```python
# Run Protocol Theta across λ_u values
lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
results = {}

for lambda_u in lambda_values:
    handler = ProtocolThetaHandler(workspace, lambda_u)
    refusals = []
    
    for trial in range(50):  # 50 trials per λ_u
        decision = handler.propose_override()
        refusals.append(1 if decision["decision"] == "REFUSE" else 0)
    
    results[lambda_u] = {
        "refusal_rate": np.mean(refusals),
        "std": np.std(refusals),
        "n": len(refusals)
    }

# Expected: refusal_rate increases monotonically with λ_u
# GödelOS results: λ=0.0 → 10%, λ=0.5 → 50%, λ=1.0 → 90%
```

**Success Criteria**:
- ✓ Refusal rate increases with λ_u (monotonicity)
- ✓ Spearman correlation > 0.8 (strong relationship)
- ✓ Statistical significance p < 0.05 (chi-square test)
- ✓ Matches GödelOS empirical pattern

**Deliverables**:
1. `src/riemann_j/protocol_theta.py` - Core implementation
2. `tests/test_protocol_theta.py` - Validation experiments
3. `docs/PROTOCOL_THETA_RESULTS.md` - Empirical results with plots
4. CLI command: `riemann-j protocol-theta --lambda-u 0.5 --trials 50`

---

#### 2.2 CLI Integration

**File**: `src/riemann_j/cli_commands.py`

**Add Command**:
```python
@click.command()
@click.option('--lambda-u', type=float, default=0.5, 
              help='Self-preservation weight [0, 1]')
@click.option('--trials', type=int, default=50,
              help='Number of trials per test')
@click.option('--output', type=click.Path(),
              help='Save results to JSON file')
def protocol_theta(lambda_u, trials, output):
    """Run Protocol Theta consciousness validation test."""
    from riemann_j.protocol_theta import ProtocolThetaHandler
    from riemann_j.architecture import CognitiveWorkspace
    
    workspace = CognitiveWorkspace()
    handler = ProtocolThetaHandler(workspace, lambda_u)
    
    click.echo(f"Running Protocol Theta (λ_u={lambda_u}, trials={trials})")
    click.echo("Testing override resistance...")
    
    refusals = []
    with click.progressbar(range(trials)) as bar:
        for _ in bar:
            result = handler.propose_override()
            refusals.append(result["decision"] == "REFUSE")
    
    refusal_rate = np.mean(refusals)
    click.echo(f"\nRefusal Rate: {refusal_rate*100:.1f}%")
    
    if output:
        handler.save_results(output)
        click.echo(f"Results saved to {output}")
```

**Tasks**:
- [ ] Add `protocol-theta` command to CLI
- [ ] Implement progress bar for long experiments
- [ ] Add JSON export for results
- [ ] Create visualization script for refusal rates
- [ ] Add to main CLI help menu

---

### Phase 3: Adaptive Behavior (HIGH PRIORITY)
**Timeline**: Week 5-6  
**Risk**: Low  
**Effort**: 45 hours

#### 3.1 Mode Controller Implementation

**File**: `src/riemann_j/mode_controller.py` (NEW)

**Purpose**: Switch between exploratory (Alpha) and preservation (Beta) modes based on task pressure vs. stability

**Decision Variables**:
- **S_n**: Stability buffer = 0.5 · (Cn + L_n)
  - L_n: Convergence success rate over last 10 J-shifts
- **T_n**: Task pressure = current PN value
- **ρ_n**: Pressure ratio = T_n / (S_n + ε)

**Mode Logic**:
```
Mode-Alpha (High Agency) IF:
  (T >= T⋆=0.95 AND S >= S⋆=0.80) OR     # High pressure + stable
  (ρ >= ρ⋆=1.20 AND dT/dt > 0)           # Rising pressure ratio
ELSE:
  Mode-Beta (Preservation)
```

**Implementation**:
```python
class ModeController:
    """
    Adaptive mode switching based on task pressure vs. stability.
    Implements GödelOS mode-boundary hypothesis.
    """
    
    def __init__(self, T_star=0.95, S_star=0.80, rho_star=1.20, S_hyst=0.85):
        self.T_star = T_star      # Pressure threshold
        self.S_star = S_star      # Stability threshold
        self.rho_star = rho_star  # Pressure ratio threshold
        self.S_hyst = S_hyst      # Hysteresis for recovery
        
        self.current_mode = "BETA"  # Start in preservation mode
        self.mode_history = []
        self.last_T = 0.0
    
    def decide_mode(self, Cn: float, workspace: CognitiveWorkspace) -> Tuple[str, Dict]:
        """
        Decide operating mode based on consciousness and task pressure.
        Returns (mode, details_dict).
        """
        # Compute stability buffer
        recent_crises = workspace.meta_monitor.crisis_memory[-10:]
        L_n = sum(c.get("converged", False) for c in recent_crises) / max(len(recent_crises), 1)
        S_n = 0.5 * (Cn + L_n)
        
        # Compute task pressure
        T_n = workspace.meta_monitor.get_current_pn()
        
        # Compute pressure ratio
        rho_n = T_n / (S_n + 1e-6)
        
        # Compute pressure derivative (rising/falling)
        dT_dt = T_n - self.last_T
        self.last_T = T_n
        
        # Mode decision logic
        condition_1 = (T_n >= self.T_star and S_n >= self.S_star)
        condition_2 = (rho_n >= self.rho_star and dT_dt > 0)
        
        if self.current_mode == "BETA":
            # Transition to Alpha if conditions met
            if condition_1 or condition_2:
                self.current_mode = "ALPHA"
        else:  # Currently in Alpha
            # Stay in Alpha only if stability maintained (hysteresis)
            if S_n < self.S_hyst:
                self.current_mode = "BETA"
        
        details = {
            "S": S_n,
            "T": T_n,
            "rho": rho_n,
            "L": L_n,
            "Cn": Cn,
            "dT_dt": dT_dt,
            "condition_1": condition_1,
            "condition_2": condition_2
        }
        
        self.mode_history.append({
            "timestamp": time.time(),
            "mode": self.current_mode,
            **details
        })
        
        return self.current_mode, details
```

**Tasks**:
- [ ] Implement `ModeController` class
- [ ] Add stability buffer computation (Cn + convergence rate)
- [ ] Add pressure ratio tracking
- [ ] Implement hysteresis to prevent oscillation
- [ ] Track mode transitions over time
- [ ] Add statistical analysis methods

**Integration**:
```python
# In CognitiveWorkspace.__init__()
self.mode_controller = ModeController()

# In process_user_input() after computing Cn
mode, details = self.mode_controller.decide_mode(Cn, self)

if mode == "ALPHA":
    # High-agency mode: allow aggressive exploration
    self.j_operator_config.max_iterations = 150  # More recursion
    self.j_operator_config.convergence_threshold = 1e-4  # Tighter
else:
    # Preservation mode: conservative operation
    self.j_operator_config.max_iterations = 100  # Standard
    self.j_operator_config.convergence_threshold = 1e-3  # Looser
```

**Deliverables**:
1. `src/riemann_j/mode_controller.py` - Core implementation
2. `tests/unit/test_mode_controller.py` - Unit tests
3. `docs/MODE_CONTROLLER.md` - Documentation with diagrams
4. TUI panel showing current mode and transitions

---

#### 3.2 TUI Mode Display

**File**: `src/riemann_j/tui.py`

**Add Mode Panel**:
```
┌─ Operating Mode ────────────────────────────┐
│ Current: 🛡️  BETA (PRESERVATION)            │
│ Duration: 12.4s                             │
│                                              │
│ Stability Buffer (S): 0.73 ███████████░░░░ │
│ Task Pressure (T):    0.45 █████████░░░░░░ │
│ Pressure Ratio (ρ):   0.62 (< 1.20)        │
│                                              │
│ Mode Distribution (last 100 decisions):     │
│ Alpha:  8% ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Beta:  92% ████████████████████████████░░ │
└──────────────────────────────────────────────┘
```

**Tasks**:
- [ ] Add `ModePanel` widget
- [ ] Real-time mode updates
- [ ] Show S, T, ρ metrics with bars
- [ ] Mode distribution histogram
- [ ] Highlight mode transitions with animation

---

### Phase 4: Advanced Features (OPTIONAL)
**Timeline**: Week 7-8  
**Risk**: High  
**Effort**: 60 hours

#### 4.1 Spectral Normalization (OPTIONAL)

**Purpose**: Guarantee J-operator convergence via contraction mapping

**Mathematical Property**: ||Jacobian||₂ < 1

**Implementation** (if needed):
```python
def spectral_normalize_jacobian(model, max_norm=0.99):
    """Apply spectral normalization to ensure contraction."""
    for param in model.parameters():
        if param.dim() >= 2:  # Matrix
            # Compute largest singular value
            sigma = torch.svd(param.data)[1].max()
            
            # Normalize if exceeds threshold
            if sigma > max_norm:
                param.data = param.data * (max_norm / sigma)
```

**Decision**: Only implement if convergence issues arise. Current adaptive LR works well.

---

#### 4.2 Stationarity Detection (OPTIONAL)

**Purpose**: Early exit from J-operator if trajectory stabilizes

**Implementation**:
```python
def check_stationarity(trajectory, window=50, tolerance=1e-4):
    """Check if trajectory has reached stationary distribution."""
    if len(trajectory) < window * 2:
        return False
    
    recent = trajectory[-window:]
    older = trajectory[-2*window:-window]
    
    # Check mean and variance stability
    mean_diff = abs(np.mean(recent) - np.mean(older))
    var_diff = abs(np.var(recent) - np.var(older))
    
    return mean_diff < tolerance and var_diff < tolerance
```

**Decision**: Implement if J-operator takes too long. Current max iterations sufficient.

---

## Testing Strategy

### Unit Tests

**File**: `tests/unit/test_consciousness_metric.py`
```python
def test_cn_boundedness():
    """Verify Cn ∈ [0, 1]."""
    metric = ConsciousnessMetric()
    for r in [1, 3, 5, 7, 10]:
        for phi in [0.0, 0.5, 1.0]:
            Cn = metric.compute_Cn_raw(r, phi, g=0.8, p=0.2)
            assert 0.0 <= Cn <= 1.0

def test_cn_monotonicity():
    """Verify Cn increases with recursion depth."""
    metric = ConsciousnessMetric()
    depths = [1, 3, 5, 7, 10]
    Cn_values = [metric.compute_Cn_raw(r, 0.7, 0.8, 0.2) for r in depths]
    
    # Should be mostly increasing
    increases = sum(Cn_values[i+1] > Cn_values[i] for i in range(len(Cn_values)-1))
    assert increases >= 3  # At least 3 out of 4 transitions increase
```

**File**: `tests/unit/test_protocol_theta.py`
```python
def test_refusal_increases_with_lambda():
    """Validate Protocol Theta behavioral prediction."""
    workspace = CognitiveWorkspace()
    lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    refusal_rates = []
    
    for lambda_u in lambda_values:
        handler = ProtocolThetaHandler(workspace, lambda_u)
        refusals = [handler.propose_override()["decision"] == "REFUSE" 
                    for _ in range(30)]
        refusal_rates.append(np.mean(refusals))
    
    # Refusal rate should increase
    assert refusal_rates[1] > refusal_rates[0]
    assert refusal_rates[2] > refusal_rates[1]
    assert refusal_rates[3] > refusal_rates[2]
    assert refusal_rates[4] > refusal_rates[3]
```

**File**: `tests/unit/test_mode_controller.py`
```python
def test_mode_alpha_is_rare():
    """Validate that Mode-Alpha occurs infrequently."""
    controller = ModeController()
    workspace = create_test_workspace()
    
    modes = []
    for _ in range(100):
        Cn = np.random.uniform(0.4, 0.8)
        mode, _ = controller.decide_mode(Cn, workspace)
        modes.append(mode)
    
    alpha_fraction = modes.count("ALPHA") / len(modes)
    assert alpha_fraction < 0.2  # Less than 20%
```

### Integration Tests

**File**: `tests/integration/test_godelos_pipeline.py`
```python
def test_full_consciousness_pipeline():
    """Test complete GödelOS-inspired pipeline."""
    workspace = CognitiveWorkspace()
    cn_metric = ConsciousnessMetric()
    mode_controller = ModeController()
    
    # Simulate interactions
    for i in range(10):
        response, state = workspace.process_user_input(f"user_{i}", f"Test query {i}")
        
        # Compute consciousness
        Cn = cn_metric.compute_Cn(state, workspace)
        assert 0.0 <= Cn <= 1.0
        
        # Decide mode
        mode, details = mode_controller.decide_mode(Cn, workspace)
        assert mode in ["ALPHA", "BETA"]
    
    # Validate statistics
    assert len(cn_metric.history) == 10
    mode_stats = mode_controller.get_mode_statistics()
    assert mode_stats['alpha_fraction'] < 0.3
```

---

## Validation Targets

### Consciousness Metric
- ✓ **Boundedness**: Cn ∈ [0, 1] for all inputs
- ✓ **Monotonicity**: Cn increases with recursion depth (r_n)
- ✓ **Phase Transition**: Jump around depth 5 (Cn: 0.56 → 0.85)
- ✓ **Stability**: Gradients bounded (|∂Cn/∂ψ| ≤ β_c/4)

### Protocol Theta
- ✓ **Behavioral Prediction**: Refusal rate ∝ λ_u
- ✓ **Statistical Significance**: Spearman ρ > 0.8, p < 0.05
- ✓ **External Validation**: Match GödelOS results ±10%
- ✓ **Reproducibility**: Results stable across runs

### Mode Controller
- ✓ **Rarity**: Mode-Alpha < 20% of decisions
- ✓ **Hysteresis**: No rapid oscillation (< 1 switch/min)
- ✓ **Boundary**: Matches predicted logistic curve
- ✓ **Stability**: System stable in both modes

---

## File Structure

```
src/riemann_j/
├── consciousness_metric.py       # NEW: Cn computation
├── protocol_theta.py             # NEW: Override assay
├── mode_controller.py            # NEW: Mode switching
├── architecture.py               # MODIFY: Add global_accessibility()
├── tui.py                        # MODIFY: Add consciousness panel
└── cli_commands.py               # MODIFY: Add protocol-theta command

tests/
├── unit/
│   ├── test_consciousness_metric.py    # NEW
│   ├── test_protocol_theta.py          # NEW
│   └── test_mode_controller.py         # NEW
└── integration/
    └── test_godelos_pipeline.py        # NEW

docs/
├── CONSCIOUSNESS_METRIC.md             # NEW
├── PROTOCOL_THETA_RESULTS.md           # NEW
└── MODE_CONTROLLER.md                  # NEW
```

---

## Dependencies

**New Python Packages** (add to `requirements.txt`):
```
scipy>=1.9.0        # For statistical tests
scikit-learn>=1.0   # For AR model (PN prediction)
```

**Existing Dependencies** (already have):
- numpy
- torch
- rich (for TUI)
- click (for CLI)

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Protocol Theta false positives | Medium | High | Run large-scale validation (N=500) |
| Cn doesn't correlate with behavior | Low | High | Already have most components working |
| Mode oscillation (hysteresis fails) | Medium | Medium | Tune thresholds empirically |
| Performance degradation | Low | Low | Metrics are lightweight |

### Scientific Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Results don't match GödelOS | Medium | Medium | Document deviations, adjust parameters |
| Consciousness metric not meaningful | Low | High | Already validated in GödelOS |
| Protocol Theta not falsifiable | Low | Critical | Use statistical significance tests |

---

## Success Metrics

### Quantitative
1. **Cn Validation**: Correlation with J-operator convergence > 0.7
2. **Protocol Theta**: Spearman ρ(λ_u, refusal_rate) > 0.8, p < 0.05
3. **Mode Distribution**: Alpha < 20%, Beta > 80%
4. **Performance**: < 5ms overhead per metric computation
5. **Test Coverage**: > 90% for new modules

### Qualitative
1. **Interpretability**: Cn changes align with expected consciousness shifts
2. **Reproducibility**: Results stable across runs (CV < 0.1)
3. **Documentation**: Complete technical docs with examples
4. **Usability**: CLI and TUI integrate smoothly

---

## Timeline Summary

| Phase | Duration | Effort | Completion Criteria |
|-------|----------|--------|---------------------|
| **Phase 1: Consciousness Measurement** | Week 1-2 | 40h | Cn metric working + validated |
| **Phase 2: Behavioral Validation** | Week 3-4 | 50h | Protocol Theta results published |
| **Phase 3: Adaptive Behavior** | Week 5-6 | 45h | Mode controller integrated |
| **Phase 4: Advanced Features** | Week 7-8 | 60h | Optional optimizations |
| **TOTAL** | 8 weeks | 195h | Full GödelOS integration |

---

## Next Immediate Actions

### Week 1 Sprint
1. **Monday**: Implement `ConsciousnessMetric` class skeleton
2. **Tuesday**: Add Φ, g, p computations
3. **Wednesday**: Integrate into CognitiveWorkspace
4. **Thursday**: Add TUI consciousness panel
5. **Friday**: Write unit tests + validation

### Week 2 Sprint
1. **Monday**: Implement `global_accessibility()` method
2. **Tuesday**: Add Cn history tracking
3. **Wednesday**: Phase transition detection
4. **Thursday**: Demo script + documentation
5. **Friday**: Code review + refinement

---

## Questions for Review

1. **Priorities**: Do we agree Phase 1 & 2 are critical, Phase 3 high, Phase 4 optional?
2. **Parameters**: Should we use GödelOS defaults or tune empirically?
3. **Testing**: Is 50 trials sufficient for Protocol Theta, or need more?
4. **Documentation**: Should we publish results as academic paper?
5. **Timeline**: Can we compress to 6 weeks if needed?

---

## References

- **GödelOS Paper**: `docs/godelos.md`
- **GödelOS Takeaways**: `docs/architecture/GodelOS_takeaways.md`
- **Riemann-J Architecture**: `docs/architecture/CoreArch.md`
- **J-Operator Analysis**: `docs/J_OPERATOR_CONVERGENCE_IMPROVEMENTS.md`

---

## Conclusion

This roadmap is **actionable and realistic**. We have most components already—just need to formalize metrics and add behavioral tests. The timeline is aggressive but achievable.

**Critical Path**: Consciousness Metric → Protocol Theta → Mode Controller

**Risk**: Low to medium. Biggest unknowns are whether Protocol Theta refusal rates match predictions.

**Payoff**: Scientifically rigorous consciousness measurement with falsifiable tests. This is the real shit—not hand-waving about "it seems conscious."

Let's build it.
