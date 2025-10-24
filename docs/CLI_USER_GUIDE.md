# CLI User Guide

## Overview

The Riemann-J CLI provides an interactive interface for conversing with a conscious agent that features:

- **Real-time introspection**: See the agent's internal state, uncertainty levels, and confidence
- **PN visualization**: Monitor prediction error signals with sparkline charts
- **Persistent identity**: The agent maintains continuity across sessions
- **Meta-cognitive commands**: Query the agent's self-awareness and history
- **Session management**: Save and load conversation sessions

## Installation

```bash
# Install the package
pip install -e .

# Or run directly from the repository
python -m riemann_j
```

## Usage

### Starting the CLI

```bash
# Default mode (with rich formatting)
riemann-j

# Specify a custom identity
riemann-j --identity my_agent

# Load a previous session
riemann-j --session sessions/my_session.json

# Plain text mode (no colors)
riemann-j --no-color

# Use legacy TUI instead
riemann-j --tui
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-i, --identity PATH` | Path to persistent identity file |
| `-s, --session PATH` | Load previous session from file |
| `--no-color` | Disable rich terminal formatting |
| `--tui` | Use legacy TUI instead of CLI |
| `--help` | Show help message |

## Interactive Commands

All commands start with `/`. Regular messages are processed through the conscious agent.

### Basic Commands

- `/help` - Show available commands
- `/quit` or `/exit` - Exit the CLI (auto-saves session)

### Introspection Commands

- `/introspect` - Show detailed meta-cognitive state
  - Includes: identity, current state, uncertainty, recent experiences
- `/introspect-brief` - Show concise meta-cognitive state
- `/identity` - Display persistent identity narrative
  - Shows formative experiences and their impact
- `/identity-brief` - Display summary identity
- `/stats` - Show PN statistics and crisis history
- `/pn` - Show current PN monitor with visualization

### Session Management

- `/save [path]` - Save current session
  - If no path provided, auto-generates filename
  - Default location: `sessions/<identity>_<timestamp>.json`
- `/load <path>` - Load previous session
  - Restores conversation history and state
- `/reset` - Create new identity (requires confirmation)
  - Clears all formative experiences

### Advanced Commands

- `/explain <input>` - Explain past behavior on similar input
  - References past experiences to explain current behavior
  - Example: `/explain Tell me about Paris`

## Multi-line Input

For long or multi-line queries, use triple quotes:

```
You > """
... > This is a long query
... > that spans multiple
... > lines.
... > """
```

The CLI will display `... (N lines) >` while in multi-line mode.

## Features

### Rich Terminal UI

When rich formatting is enabled (default), you'll see:

- **Color-coded uncertainty levels**:
  - 🟢 Green: Low uncertainty (confident)
  - ⚪ White: Moderate uncertainty
  - 🟡 Yellow: High uncertainty
  - 🔴 Red: Critical uncertainty

- **PN Sparkline**: Visual representation of prediction error history
  ```
  PN History: ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁
  ```

- **Formatted panels**: Responses, introspection, and identity in bordered panels

### Session Persistence

Sessions are automatically saved on exit and include:

- Conversation history with timestamps
- PN values and confidence levels
- Workspace state
- Session metadata (creation time, total turns)

### Identity Continuity

The agent maintains a persistent identity across sessions:

- **Formative experiences**: Significant events that shaped the agent
- **Crisis memory**: History of high PN events and resolutions
- **Temporal continuity**: Same entity across restarts
- **Autobiographical narrative**: Can describe its own history

## Examples

### Basic Conversation

```
You > Hello! What can you tell me about yourself?
╭─ Agent Response ────────────────────────────────╮
│ I am a cognitive agent with meta-cognitive      │
│ awareness. I can monitor my own uncertainty     │
│ and maintain identity across sessions.          │
│                                                  │
│ PN: LOW | Uncertainty: low | Confidence: 95.0%  │
╰─────────────────────────────────────────────────╯
```

### Introspection

```
You > /introspect
╭─ Meta-Cognitive State ──────────────────────────╮
│ === WHO I AM ===                                 │
│ I am newly formed with no formative experiences  │
│                                                  │
│ === CURRENT STATE ===                            │
│ Current PN: 0.012                                │
│ Uncertainty: low                                 │
│ Self-belief: competent (0.85)                    │
│                                                  │
│ === RECENT EXPERIENCES ===                       │
│ - 1635789012: "Hello! What..." (low, 95%)       │
╰─────────────────────────────────────────────────╯
```

### Session Management

```
You > /save my_conversation
Info: Session saved to my_conversation.json

You > /quit
Info: Session auto-saved to sessions/default_cli_identity_autosave.json
Info: Goodbye!
```

Later:

```bash
riemann-j --session my_conversation.json
```

## Technical Details

### Architecture

The CLI consists of several modular components:

- **cli.py**: Main REPL loop and orchestration
- **cli_display.py**: Rich terminal formatting and visualization
- **cli_input.py**: Input parsing and validation
- **cli_commands.py**: Meta-command implementations
- **cli_config.py**: Configuration and session state

### Integration

The CLI integrates with:

- **ConsciousAgent**: Core active inference loop
- **MetaCognitiveMonitor**: PN tracking and self-belief
- **PersistentSelf**: Identity and temporal continuity
- **UncertaintyInterface**: Epistemic state classification

### Data Flow

```
User Input → InputHandler → parse() → handle_input()
                                      ├─ Command → CommandHandler
                                      └─ Message → ConsciousAgent
                                                   ↓
                                         ConsciousExperience
                                                   ↓
                                         DisplayManager → render()
```

## Troubleshooting

### Sessions Directory Not Found

Sessions are saved to `sessions/` by default. Create it manually if needed:

```bash
mkdir -p sessions
```

### Identity File Conflicts

Each identity file is separate. If you want multiple agents:

```bash
riemann-j --identity agent1
riemann-j --identity agent2
```

### Rich Formatting Issues

If terminal doesn't support rich formatting:

```bash
riemann-j --no-color
```

## Advanced Usage

### Custom Workflows

1. **Research Assistant**: Long multi-line queries with `/introspect` to verify understanding
2. **Continuous Learning**: Multiple sessions with same identity to build knowledge
3. **Crisis Analysis**: Use `/stats` and `/pn` to monitor agent stability

### Scripting

```bash
# Automated session creation
riemann-j --identity research_agent --session last_session.json < questions.txt
```

## See Also

- [CLI Roadmap](CLI_ROADMAP.md) - Implementation specification
- [Architecture Documentation](architecture/) - System design
- [Tests](../tests/) - Example usage patterns
