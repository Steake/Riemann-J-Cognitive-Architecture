#!/bin/sh
# Copilot Coding Agent Configuration Loader
#
# This script reads the agent configuration from config.yml and exports
# environment variables for use by agent runners, CI jobs, or Docker containers.
#
# Usage:
#   source .github/coding-agent/run_agent.sh
#   # OR
#   eval $(.github/coding-agent/run_agent.sh)
#
# Exported Variables:
#   AGENT_WORKSPACE_SIZE_GB - Workspace disk allocation in GB
#   AGENT_MEMORY_GB        - Recommended memory allocation in GB
#   AGENT_CPU_CORES        - Recommended CPU core count

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE" >&2
    exit 1
fi

# Function to parse YAML using yq if available, otherwise use grep/sed fallback
parse_yaml() {
    local key="$1"
    local config_file="$2"
    
    # Try yq first (if available)
    if command -v yq >/dev/null 2>&1; then
        yq eval ".$key" "$config_file" 2>/dev/null
    else
        # Fallback to grep/sed parsing for simple key-value pairs
        # This handles the basic format: "key: value"
        grep "^${key}:" "$config_file" | sed 's/^[^:]*:[[:space:]]*//' | head -n 1
    fi
}

# Parse configuration values
WORKSPACE_SIZE=$(parse_yaml "workspace_size_gb" "$CONFIG_FILE")
MEMORY_SIZE=$(parse_yaml "recommended_memory_gb" "$CONFIG_FILE")
CPU_CORES=$(parse_yaml "recommended_cpu_cores" "$CONFIG_FILE")

# Validate that we got values
if [ -z "$WORKSPACE_SIZE" ] || [ -z "$MEMORY_SIZE" ] || [ -z "$CPU_CORES" ]; then
    echo "Error: Failed to parse configuration from $CONFIG_FILE" >&2
    echo "Ensure the file contains workspace_size_gb, recommended_memory_gb, and recommended_cpu_cores" >&2
    exit 1
fi

# Export environment variables
export AGENT_WORKSPACE_SIZE_GB="$WORKSPACE_SIZE"
export AGENT_MEMORY_GB="$MEMORY_SIZE"
export AGENT_CPU_CORES="$CPU_CORES"

# Print exported variables (useful when eval'ing or sourcing)
echo "export AGENT_WORKSPACE_SIZE_GB=$AGENT_WORKSPACE_SIZE_GB"
echo "export AGENT_MEMORY_GB=$AGENT_MEMORY_GB"
echo "export AGENT_CPU_CORES=$AGENT_CPU_CORES"

# Print summary to stderr so it doesn't interfere with eval
echo "✓ Agent configuration loaded successfully" >&2
echo "  Workspace: ${AGENT_WORKSPACE_SIZE_GB}GB" >&2
echo "  Memory: ${AGENT_MEMORY_GB}GB" >&2
echo "  CPU Cores: ${AGENT_CPU_CORES}" >&2
