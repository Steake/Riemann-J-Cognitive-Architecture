# Testing Guide

## Overview

Riemann-J includes a comprehensive test suite covering unit tests, integration tests, and Behavior-Driven Development (BDD) specifications.

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/riemann_j --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Test Categories

#### Unit Tests

Located in `tests/unit/`, these test individual components in isolation.

```bash
# Run all unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_pn_driver.py

# Run specific test
pytest tests/unit/test_config.py::TestConfiguration::test_pn_threshold_exists
```

#### BDD Tests

Located in `tests/bdd/`, these use Gherkin syntax for behavior specifications.

```bash
# Run all BDD tests
pytest tests/bdd/

# Run specific feature
pytest tests/bdd/features/pn_driver.feature

# Run specific scenario
pytest tests/bdd/features/pn_driver.feature -k "PN Driver starts"
```

#### Integration Tests

Located in `tests/integration/`, these test component interactions.

```bash
# Run integration tests
pytest tests/integration/
```

## Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── unit/                   # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_pn_driver.py
│   └── test_architecture.py
├── bdd/                    # BDD tests
│   ├── __init__.py
│   ├── features/           # Gherkin scenarios
│   │   ├── pn_driver.feature
│   │   ├── user_attractor.feature
│   │   └── j_operator.feature
│   └── step_defs/          # Step implementations
│       └── test_pn_driver_steps.py
└── integration/            # Integration tests
    └── __init__.py
```

## Writing Tests

### Unit Test Example

```python
import pytest
from riemann_j import config


class TestConfiguration:
    """Test suite for configuration parameters."""

    def test_pn_threshold_exists(self):
        """Verify PN_THRESHOLD is defined."""
        assert hasattr(config, 'PN_THRESHOLD')
        assert isinstance(config.PN_THRESHOLD, float)
        assert 0.0 <= config.PN_THRESHOLD <= 1.0
```

### BDD Test Example

**Feature File** (`tests/bdd/features/example.feature`):

```gherkin
Feature: Example Feature
  As a user
  I want to test something
  So that I can verify it works

  Scenario: Example scenario
    Given a precondition
    When an action occurs
    Then the expected result happens
```

**Step Definitions** (`tests/bdd/step_defs/test_example_steps.py`):

```python
from pytest_bdd import scenarios, given, when, then

scenarios('../features/example.feature')


@given('a precondition')
def precondition():
    pass


@when('an action occurs')
def action():
    pass


@then('the expected result happens')
def result():
    assert True
```

## Coverage Goals

We aim for:
- **Unit Tests**: >90% line coverage
- **BDD Tests**: All user-facing scenarios
- **Integration Tests**: Critical component interactions

### Current Coverage

```bash
# Generate coverage report
pytest --cov=src/riemann_j --cov-report=term-missing

# View detailed HTML report
pytest --cov=src/riemann_j --cov-report=html
open htmlcov/index.html
```

## Pass/Fail Criteria

### Unit Tests

- ✅ All configuration parameters exist and have correct types
- ✅ PN Driver calculates sigmoid correctly
- ✅ PN values stay in range [0.0, 1.0]
- ✅ J-Operator converges within max iterations
- ✅ User Attractors maintain state isolation
- ✅ Encoding/decoding preserves dimensionality

### BDD Tests

- ✅ PN Driver starts with low prediction error
- ✅ PN increases with accumulated steps
- ✅ PN resets when zero is found
- ✅ Signals are added to workspace correctly
- ✅ User attractors are created on first interaction
- ✅ GMM is updated periodically
- ✅ Affinity modifies user states appropriately
- ✅ J-Shift triggers when PN exceeds threshold
- ✅ Adaptive learning rate adjusts based on distance
- ✅ Lyapunov analysis completes successfully

### Integration Tests

- ✅ End-to-end user interaction flow
- ✅ Multi-user concurrent processing
- ✅ J-Shift integration with workspace
- ✅ State logging and recovery

## Continuous Integration

### GitHub Actions (example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=src/riemann_j --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Test Data

### Fixtures

Common fixtures are defined in `tests/conftest.py`:

```python
import pytest
from riemann_j import CognitiveWorkspace


@pytest.fixture
def workspace():
    """Provides a CognitiveWorkspace instance."""
    ws = CognitiveWorkspace()
    yield ws
    ws.close()


@pytest.fixture
def mock_state():
    """Provides a mock state vector."""
    import numpy as np
    return np.random.randn(768)
```

## Debugging Tests

### Verbose Output

```bash
# Show print statements
pytest -v -s

# Show detailed failure info
pytest -vv

# Stop at first failure
pytest -x
```

### Debugging in IPython

```python
# In test file
def test_something():
    import IPython; IPython.embed()
    # Test code here
```

### Using pdb

```bash
# Drop into debugger on failure
pytest --pdb

# Set breakpoint in code
import pdb; pdb.set_trace()
```

## Performance Testing

### Benchmarking

```python
import pytest
import time


@pytest.mark.benchmark
def test_encoding_speed(workspace):
    """Benchmark encoding performance."""
    start = time.time()
    for _ in range(100):
        workspace.symbolic_interface.encoder("test")
    duration = time.time() - start
    assert duration < 10.0  # Should complete in <10s
```

## Test Isolation

- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order
- Clean up resources (files, threads, etc.)

## Mocking

For tests that shouldn't hit external services:

```python
from unittest.mock import Mock, patch


def test_with_mock():
    with patch('riemann_j.shared_resources.model') as mock_model:
        mock_model.generate.return_value = Mock()
        # Test code here
```

## Common Issues

### Import Errors

If imports fail, ensure package is installed:

```bash
pip install -e .
```

### CUDA Errors

Tests run on CPU by default. To force CPU:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Threading Issues

For tests involving threads, use appropriate timeouts:

```python
import threading


def test_thread():
    thread = threading.Thread(target=some_function)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-bdd documentation](https://pytest-bdd.readthedocs.io/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-23  
**Status**: Production
