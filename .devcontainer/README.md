# GitHub Codespaces Configuration

This directory contains the configuration for GitHub Codespaces, enabling you to develop Riemann-J in a cloud-based development environment.

## Quick Start

Click the "Open in GitHub Codespaces" badge in the main README to launch a new Codespace with all dependencies pre-installed.

## What's Included

The devcontainer configuration provides:

- **Python 3.11** runtime environment
- **Git** for version control
- **VS Code Extensions**:
  - Python language support with IntelliSense
  - Pylance for advanced type checking
  - Black formatter for consistent code style
  - Jupyter support for interactive development

- **Automatic Setup**: All project dependencies are installed automatically via `pip install -e '.[dev]'`

## Configuration Details

### Base Image

We use the official Microsoft Python devcontainer image (`mcr.microsoft.com/devcontainers/python:3.11`) which provides:
- Python 3.11 pre-installed
- Common development tools (git, curl, etc.)
- VS Code Server compatibility

### Post-Create Command

After the container is created, the following command runs automatically:
```bash
pip install -e '.[dev]'
```

This installs:
- The riemann-j package in editable mode
- All runtime dependencies (transformers, torch, etc.)
- All development dependencies (pytest, black, flake8, etc.)

## Running the Application

Once your Codespace is ready, you can run Riemann-J:

```bash
# Using the installed command
riemann-j

# Or as a module
python -m riemann_j

# Or using the provided scripts
./run.sh
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
./test.sh --coverage

# Run specific test types
./test.sh --unit
./test.sh --bdd
```

## Customizing Your Environment

You can customize your Codespace by:

1. **Adding VS Code Settings**: Edit `devcontainer.json` under `customizations.vscode.settings`
2. **Installing Additional Extensions**: Add to `customizations.vscode.extensions`
3. **Running Additional Setup**: Modify the `postCreateCommand`

## Known Limitations

- **GPU Support**: Codespaces don't have GPU acceleration by default. The application will fall back to CPU inference.
- **Model Download**: First run will download the Phi-3.5-mini-instruct model (~3.8GB), which may take a few minutes.
- **Memory**: The application requires ~8GB RAM for optimal performance. Default 2-core Codespace machines have 8GB RAM, which should be sufficient. If you experience memory issues, consider upgrading to a larger machine type.

## Troubleshooting

### Model Download Fails
If the model download times out or fails:
```bash
# Pre-download the model manually
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct'); AutoModelForCausalLM.from_pretrained('microsoft/Phi-3.5-mini-instruct')"
```

### Dependencies Not Installed
If dependencies are missing, manually run:
```bash
pip install -e '.[dev]'
```

## Learn More

- [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces)
- [Dev Container Specification](https://containers.dev/)
- [Riemann-J Documentation](../docs/)
