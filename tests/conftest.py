"""
Pytest configuration and fixtures.
"""
import pytest
import sys
from unittest.mock import Mock, MagicMock
import numpy as np
import torch


# Mock the shared_resources module before any imports
@pytest.fixture(scope="session", autouse=True)
def mock_shared_resources():
    """Mock shared resources to avoid loading large models during testing."""
    mock_module = MagicMock()
    
    # Mock global_workspace
    import queue
    mock_module.global_workspace = queue.PriorityQueue()
    
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.pad_token = "[PAD]"
    mock_tokenizer.eos_token = "[EOS]"
    mock_tokenizer.bos_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_module.tokenizer = mock_tokenizer
    
    # Mock model
    mock_model = Mock()
    mock_config = Mock()
    mock_config.hidden_size = 768
    mock_config.vocab_size = 50257
    mock_model.config = mock_config
    
    # Mock model forward pass
    def mock_forward(*args, **kwargs):
        batch_size = 1
        seq_len = 10
        hidden_size = 768
        
        result = Mock()
        # Create mock hidden states
        hidden_states = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(13)  # GPT-2 has 12 layers + embedding
        ])
        result.hidden_states = hidden_states
        return result
    
    mock_model.side_effect = mock_forward
    mock_model.eval = Mock(return_value=None)
    mock_model.to = Mock(return_value=mock_model)
    
    # Mock transformer layers
    mock_transformer = Mock()
    mock_layer = Mock()
    
    def mock_layer_forward(x):
        # Return same shape
        if isinstance(x, torch.Tensor):
            return (x,)
        return (torch.randn_like(x[0]),)
    
    mock_layer.side_effect = mock_layer_forward
    mock_transformer.h = [mock_layer]
    mock_model.transformer = mock_transformer
    
    # Mock generate
    def mock_generate(*args, **kwargs):
        return torch.randint(0, 50257, (1, 50))
    
    mock_model.generate = mock_generate
    
    mock_module.model = mock_model
    mock_module.device = "cpu"
    
    # Inject into sys.modules
    sys.modules['riemann_j.shared_resources'] = mock_module
    
    yield mock_module


@pytest.fixture
def mock_workspace():
    """Provides a mocked CognitiveWorkspace for testing."""
    # Import here to use mocked shared_resources
    from riemann_j.architecture import CognitiveWorkspace
    
    ws = CognitiveWorkspace()
    yield ws
    ws.close()


@pytest.fixture
def mock_state_vector():
    """Provides a mock state vector for testing."""
    return np.random.randn(768)
