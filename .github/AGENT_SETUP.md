# Copilot Coding Agent Setup Guide

This repository includes configuration for running a GitHub Copilot coding agent with dedicated resource allocation. The configuration provides a 64GB workspace with recommended memory and CPU allocations for optimal performance.

## 📋 Overview

The agent configuration consists of:

- **`config.yml`** - Resource allocation settings (workspace size, memory, CPU)
- **`run_agent.sh`** - Helper script to export configuration as environment variables
- **This guide** - Setup instructions and usage examples

## 🚀 Quick Start

### 1. Load Configuration

To load the agent configuration into your environment:

```bash
# Option 1: Source the script directly
source .github/coding-agent/run_agent.sh

# Option 2: Use eval (works in any POSIX shell)
eval $(.github/coding-agent/run_agent.sh)
```

This exports three environment variables:
- `AGENT_WORKSPACE_SIZE_GB` - Workspace disk allocation (64GB)
- `AGENT_MEMORY_GB` - Recommended memory (32GB)
- `AGENT_CPU_CORES` - Recommended CPU cores (8)

### 2. Verify Configuration

```bash
echo "Workspace: ${AGENT_WORKSPACE_SIZE_GB}GB"
echo "Memory: ${AGENT_MEMORY_GB}GB"
echo "CPU Cores: ${AGENT_CPU_CORES}"
```

## 🐳 Using with Docker

When running the agent in a Docker container, pass the environment variables and configure resource limits:

```bash
# Load configuration
eval $(.github/coding-agent/run_agent.sh)

# Run Docker container with resource limits
docker run -d \
  --name copilot-agent \
  --memory="${AGENT_MEMORY_GB}g" \
  --cpus="${AGENT_CPU_CORES}" \
  --storage-opt size="${AGENT_WORKSPACE_SIZE_GB}g" \
  -e AGENT_WORKSPACE_SIZE_GB \
  -e AGENT_MEMORY_GB \
  -e AGENT_CPU_CORES \
  -v $(pwd):/workspace \
  your-agent-image:latest
```

**Note**: Docker storage limits require specific storage drivers (overlay2 with xfs, devicemapper). For disk space, ensure the host has sufficient capacity.

## ⚙️ GitHub Actions Integration

### Using GitHub-Hosted Runners

GitHub-hosted runners have fixed resources:
- Ubuntu latest: 7GB RAM, 14GB disk (SSD), 2 cores
- Larger runners: Up to 64GB RAM, 256GB disk, 16 cores (Team/Enterprise)

For this configuration (64GB workspace), you'll need **GitHub-hosted larger runners** or a **self-hosted runner**.

### Example Workflow

```yaml
name: Copilot Agent CI

on: [push, pull_request]

jobs:
  agent-tasks:
    runs-on: ubuntu-latest-8-cores  # Use larger runner
    steps:
      - uses: actions/checkout@v3
      
      - name: Load Agent Configuration
        id: agent-config
        run: |
          eval $(.github/coding-agent/run_agent.sh)
          echo "workspace_gb=$AGENT_WORKSPACE_SIZE_GB" >> $GITHUB_OUTPUT
          echo "memory_gb=$AGENT_MEMORY_GB" >> $GITHUB_OUTPUT
          echo "cpu_cores=$AGENT_CPU_CORES" >> $GITHUB_OUTPUT
      
      - name: Run Agent Tasks
        run: |
          echo "Running with ${AGENT_WORKSPACE_SIZE_GB}GB workspace"
          # Your agent commands here
```

## 🖥️ Self-Hosted Runners

For full control over resources, use a self-hosted runner:

### Prerequisites

Ensure your self-hosted runner meets the requirements:
- **Disk Space**: At least 64GB free
- **Memory**: 32GB RAM recommended
- **CPU**: 8 cores recommended

### Setup

1. **Configure Runner**: Follow [GitHub's self-hosted runner setup](https://docs.github.com/en/actions/hosting-your-own-runners/adding-self-hosted-runners)

2. **Load Configuration Before Jobs**:
   ```bash
   # In your runner's startup script or job
   cd /path/to/repository
   eval $(.github/coding-agent/run_agent.sh)
   ```

3. **Update Workflow** to use your self-hosted runner:
   ```yaml
   jobs:
     agent-tasks:
       runs-on: self-hosted
       # Rest of job configuration
   ```

## 🔧 Local Development

When developing or testing the agent locally:

```bash
# 1. Clone the repository
git clone https://github.com/Steake/Riemann-J-Cognitive-Architecture.git
cd Riemann-J-Cognitive-Architecture

# 2. Load agent configuration
eval $(.github/coding-agent/run_agent.sh)

# 3. Ensure sufficient disk space
df -h .  # Check available space

# 4. Run your agent or build
# The environment variables are now available to any tools or scripts
```

## 📝 Configuration Customization

To modify resource allocations, edit `.github/coding-agent/config.yml`:

```yaml
workspace_size_gb: 64      # Adjust workspace size
recommended_memory_gb: 32  # Adjust memory allocation  
recommended_cpu_cores: 8   # Adjust CPU cores
```

After changes, reload the configuration:
```bash
eval $(.github/coding-agent/run_agent.sh)
```

## 🛠️ Troubleshooting

### Script Fails to Parse Config

**Problem**: "Failed to parse configuration" error

**Solution**: 
- Ensure `config.yml` is properly formatted (valid YAML)
- Install `yq` for better YAML parsing: `sudo apt-get install yq` or `brew install yq`
- The script falls back to grep/sed parsing if yq is unavailable

### Insufficient Disk Space

**Problem**: Agent runs out of disk space during operation

**Solution**:
- Verify available disk space: `df -h`
- For Docker: Check Docker's storage location has sufficient space
- For GitHub Actions: Use larger runners or self-hosted runners
- Clean up old builds/artifacts before agent runs

### Memory or CPU Constraints

**Problem**: Agent performance is slow or jobs are killed

**Solution**:
- Reduce `recommended_memory_gb` or `recommended_cpu_cores` in config.yml
- Ensure Docker host or runner machine has adequate resources
- Monitor resource usage: `docker stats` or `htop`

## 📚 Additional Resources

- [GitHub Actions Runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners)
- [Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)
- [Docker Resource Constraints](https://docs.docker.com/config/containers/resource_constraints/)
- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)

## 📞 Support

For issues specific to this configuration, please open an issue in the repository or consult the main [README.md](../README.md).
