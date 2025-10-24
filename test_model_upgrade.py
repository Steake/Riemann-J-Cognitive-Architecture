#!/usr/bin/env python3
"""
Test script for validating the model upgrade and hybrid projection head implementation.
Tests: Model loading, basic conversation, blend parameter effects.
"""

import torch

from riemann_j import config as cfg
from riemann_j.architecture import CognitiveWorkspace


def test_model_loading():
    """Test that Qwen2.5-3B-Instruct loads correctly."""
    print(f"Testing model loading: {cfg.TRANSFORMER_MODEL_NAME}")
    arch = CognitiveWorkspace()
    print(f"✓ Model loaded successfully: {arch.model.config.model_type}")
    print(f"  Model parameters: {sum(p.numel() for p in arch.model.parameters()):,}")
    print(f"  Vocab size: {arch.model.config.vocab_size}")
    return arch


def test_basic_conversation(arch):
    """Test basic conversational quality."""
    print("\n=== Testing Basic Conversation ===")
    test_prompts = [
        "What is 2+2?",
        "Tell me a fun fact about space.",
        "What's the capital of France?",
    ]

    for prompt in test_prompts:
        print(f"\nUser: {prompt}")
        response = arch.prompt_based_generate(prompt=prompt, max_new_tokens=50, temperature=0.7)
        print(f"Agent: {response[:200]}{'...' if len(response) > 200 else ''}")


def test_blend_parameter(arch):
    """Test hybrid projection blending at different alpha values."""
    print("\n=== Testing Blend Parameter ===")

    # Create a dummy state vector
    state_vec = torch.randn(cfg.SYMBOLIC_DIM).numpy()

    prompt = "Hello, how are you?"
    alphas = [0.0, 0.1, 0.5]

    for alpha in alphas:
        print(f"\nTesting with α={alpha}:")
        original_alpha = cfg.PROJECTION_BLEND_ALPHA
        cfg.PROJECTION_BLEND_ALPHA = alpha

        response = arch.prompt_based_generate(
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.7,
            state_vector=state_vec if alpha > 0 else None,
        )

        print(f"  Response: {response[:150]}{'...' if len(response) > 150 else ''}")
        cfg.PROJECTION_BLEND_ALPHA = original_alpha


def test_projection_head_status(arch):
    """Verify projection head is present and functional."""
    print("\n=== Testing Projection Head ===")
    print(f"USE_PROJECTION_HEAD: {cfg.USE_PROJECTION_HEAD}")
    print(f"PROJECTION_BLEND_ALPHA: {cfg.PROJECTION_BLEND_ALPHA}")
    print(f"Projection head architecture:")
    print(f"  Input: {cfg.SYMBOLIC_DIM} → Hidden: 256 → Output: {arch.model.config.vocab_size}")

    # Test forward pass
    dummy_state = torch.randn(1, cfg.SYMBOLIC_DIM).to(arch.device)
    logit_bias = arch.projection_head(dummy_state)
    print(f"  Forward pass successful: {logit_bias.shape}")
    print(f"  Output range: [{logit_bias.min().item():.4f}, {logit_bias.max().item():.4f}]")


def test_config_values():
    """Display current configuration values."""
    print("\n=== Configuration Values ===")
    print(f"Model: {cfg.TRANSFORMER_MODEL_NAME}")
    print(f"Symbolic dim: {cfg.SYMBOLIC_DIM}")
    print(f"Device: {cfg.DEVICE}")
    print(f"Max tokens: {cfg.MAX_NEW_TOKENS}")
    print(f"Temperature: {cfg.TEMPERATURE}")
    print(f"Projection blend α: {cfg.PROJECTION_BLEND_ALPHA}")
    print(f"Use projection head: {cfg.USE_PROJECTION_HEAD}")
    print(f"J-operator max iterations: {cfg.J_OPERATOR_MAX_ITERATIONS}")


if __name__ == "__main__":
    print("=" * 60)
    print("Model Upgrade & Hybrid Projection Head Test")
    print("=" * 60)

    try:
        # Configuration
        test_config_values()

        # Model loading
        print("\n" + "=" * 60)
        arch = test_model_loading()

        # Projection head
        print("\n" + "=" * 60)
        test_projection_head_status(arch)

        # Basic conversation
        print("\n" + "=" * 60)
        test_basic_conversation(arch)

        # Blend parameter
        print("\n" + "=" * 60)
        test_blend_parameter(arch)

        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
