# CLI New Features Documentation

## Overview

The Riemann-J CLI now includes advanced features for interactive exploration of the cognitive architecture:

### 1. Tab Completion ✨
- Press `TAB` after typing `/` to see all available commands
- Start typing a command and press `TAB` to autocomplete
- Works with all meta-commands

### 2. Command History (↑/↓) 📜
- Press `UP` arrow to recall previous commands
- Press `DOWN` arrow to navigate forward through history
- Full session history maintained in memory

### 3. Manual Synthetic State Injection 🧠
- New `/inject-state` command for manually injecting synthetic states
- Useful for testing crisis handling and identity formation

## Using `/inject-state`

### Syntax
```bash
/inject-state <trigger_description> [--pn=<value>] [--crisis]
```

### Parameters
- `<trigger_description>`: Description of what triggered this state (required)
- `--pn=<value>`: Override PN value (0.0-1.0, optional)
- `--crisis`: Force crisis handling regardless of PN (optional)

### Examples

#### 1. Inject a crisis with high PN:
```bash
/inject-state catastrophic failure --pn=0.9 --crisis
```
This creates a high-uncertainty state that triggers crisis handling and likely becomes a formative experience.

#### 2. Inject routine state with low PN:
```bash
/inject-state routine processing --pn=0.05
```
This integrates as a normal interaction without crisis escalation.

#### 3. Inject medium uncertainty:
```bash
/inject-state ambiguous input --pn=0.4
```
Borderline case that may or may not trigger enhanced processing.

#### 4. Force crisis without PN override:
```bash
/inject-state unexpected scenario --crisis
```
Uses current or default PN but forces crisis handling.

### What Happens Internally

When you inject a synthetic state:

1. **State Creation**: A `SyntheticState` object is created with:
   - Random latent representation (768-dimensional vector)
   - Your trigger description
   - Specified or current PN value
   - Timestamp

2. **Integration**:
   - If PN ≥ 0.5 OR `--crisis` flag: → `integrate_crisis()`
   - Otherwise: → `integrate_interaction()`

3. **Equilibrium Perturbation**:
   - Injected PN value is sent to `EquilibriumRegulator.inject_perturbation()`
   - PN **jumps immediately** to the injected value
   - Background regulator thread **resumes homeostatic decay** toward target (0.5)
   - Time constant: ~20 seconds for equilibration

4. **Effects**:
   - Meta-monitor continuously observes regulated PN
   - High PN crises may become formative experiences
   - Identity metrics updated (total interactions/crises)
   - PN naturally decays back toward equilibrium over time
   - **Real-time observation**: Watch PN trajectory using `/pn` sparkline

### Inspecting Effects

After injecting states, use these commands to see the impact:

```bash
/stats          # View PN statistics and crisis counts
/introspect     # See current meta-cognitive state
/identity       # View formative experiences
/pn             # Visualize PN history
```

## Interactive Demo

Run the interactive CLI:
```bash
python -m riemann_j
```

Then try this sequence:

```bash
# 1. Check baseline state
/stats
/identity
/pn             # Check current PN baseline

# 2. Inject a crisis and observe equilibration
/inject-state system overload --pn=0.85 --crisis

# 3. Watch PN decay toward equilibrium (repeat every few seconds)
/pn             # Shows PN spike, then gradual decay toward 0.5
/pn             # After 5 seconds: PN should be lower
/pn             # After 10 seconds: approaching 0.6
/pn             # After 20 seconds: near equilibrium at ~0.5

# 4. Check identity impact
/stats          # Should show 1 crisis
/introspect     # Shows updated beliefs
/identity       # May show new formative experience

# 5. Inject routine states
/inject-state normal query --pn=0.02
/pn             # PN jumps to 0.02, then drifts back up toward 0.5

# 6. Test equilibrium restoration
/inject-state test --pn=0.95
/pn             # High spike
# Wait 30 seconds, check again:
/pn             # Should be approaching 0.5

# 7. Have a conversation
Hello, how do you feel after that crisis?

# 8. Explain the past
/explain crisis situation
```

## Tab Completion Commands

All commands support tab completion:

```bash
/help              # Show help
/quit, /exit       # Exit CLI
/introspect        # Detailed meta-state
/introspect-brief  # Concise meta-state
/identity          # Full identity narrative
/identity-brief    # Summary identity
/explain <input>   # Explain past behavior
/save [path]       # Save session
/load <path>       # Load session
/reset             # Reset identity
/stats             # PN statistics
/pn                # PN visualization
/inject-state      # Inject synthetic state ← NEW!
```

## Command History Usage

The CLI maintains full command history in memory:

1. **Recall previous command**: Press `↑`
2. **Navigate forward**: Press `↓`
3. **Edit and rerun**: Recall a command, edit it, press ENTER
4. **Cycle through history**: Keep pressing `↑` to go back further

This is especially useful for:
- Repeating `/inject-state` with variations
- Re-running `/stats` after changes
- Modifying parameters in experiments

## Technical Details

### SyntheticStateSpec Validation

The `/inject-state` command validates inputs:

- **PN range**: Must be 0.0 ≤ PN ≤ 1.0
- **Trigger**: Cannot be empty or whitespace-only
- **Latent dim**: Must be positive (default: 768)

Invalid specs show error messages without crashing.

### Integration with Persistent Self

Injected states integrate into the persistent self:

```python
# Crisis path (PN ≥ 0.5 or --crisis flag)
persistent_self.integrate_crisis(state)
→ May create FormativeExperience
→ Updates crisis metrics
→ Records in crisis memory

# Routine path (PN < 0.5)
persistent_self.integrate_interaction(state)
→ Updates interaction count
→ No formative experience (unless significant)
```

### PN Monitor Effects

The meta-monitor tracks all PN observations:

```python
meta_monitor.observe_pn(pn_value)
→ Updates PN history (for visualization)
→ Adjusts self-belief state
→ Tracks volatility
```

## Use Cases

### 1. Testing Crisis Handling
```bash
/inject-state edge case --pn=0.95 --crisis
/introspect
# See how agent handles high uncertainty
```

### 2. Building Identity History
```bash
/inject-state formative moment 1 --pn=0.7 --crisis
/inject-state formative moment 2 --pn=0.8 --crisis
/inject-state formative moment 3 --pn=0.75 --crisis
/identity
# See multiple formative experiences
```

### 3. PN Threshold Experiments
```bash
/inject-state low pn --pn=0.45
/inject-state threshold --pn=0.50
/inject-state high pn --pn=0.55
/pn
# Visualize PN threshold behavior
```

### 4. Conversation Context
```bash
/inject-state user expressed confusion --pn=0.6 --crisis
Hello, I'm still confused about that topic
/explain confusion
# Agent explains past handling of confusion
```

## Programmatic Usage

You can also use the feature programmatically:

```python
from riemann_j.cli import RiemannCLI
from riemann_j.cli_config import SyntheticStateSpec

cli = RiemannCLI(identity_path="my_identity")

# Create spec
spec = SyntheticStateSpec(
    trigger="programmatic injection",
    pn_override=0.75,
    is_crisis=True
)

# Validate
valid, msg = spec.validate()
if valid:
    # Inject via command handler
    cli.command_handler._cmd_inject_state(
        "programmatic test --pn=0.75 --crisis"
    )
```

## Troubleshooting

### Tab completion not working
- Ensure `prompt_toolkit` is installed: `pip install prompt_toolkit`
- Try typing `/` then pressing TAB

### Command history lost
- History is in-memory only, lost on exit
- Future: Session files will preserve history

### Invalid PN error
- Check PN is between 0.0 and 1.0
- Use `--pn=0.5` not `--pn=.5` (leading zero required)

### State not becoming formative
- PN must be ≥ 0.5 for crisis handling
- Use `--crisis` flag to force crisis path
- Check `/stats` to verify crisis was recorded

## Future Enhancements

Planned improvements:
- [ ] Persistent command history across sessions
- [ ] Custom latent representations (load from file)
- [ ] Batch state injection from JSON
- [ ] State injection history/log
- [ ] Undo last injection
- [ ] Visual timeline of injected states

---

**Status**: ✅ Fully implemented and tested  
**Version**: CLI v2.0 with synthetic state injection  
**Last Updated**: October 24, 2025
