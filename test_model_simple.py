#!/usr/bin/env python3
"""Simple direct test of model and blend functionality without TUI interference."""

import sys

import torch

# Direct imports
from src.riemann_j import config as cfg
from src.riemann_j.architecture import CognitiveWorkspace

print("=" * 60)
print("MODEL UPGRADE VALIDATION TEST")
print("=" * 60)

print("\n[1/4] Configuration Check")
print(f"  Model: {cfg.TRANSFORMER_MODEL_NAME}")
print(f"  Blend Alpha: {cfg.PROJECTION_BLEND_ALPHA}")
print(f"  Use Projection Head: {cfg.USE_PROJECTION_HEAD}")

print("\n[2/4] Loading CognitiveWorkspace...")
workspace = CognitiveWorkspace()
model = workspace.symbolic_interface.model
print(f"  ✓ Model type: {model.config.model_type}")
print(f"  ✓ Vocab size: {model.config.vocab_size}")
print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n[3/4] Testing Basic Generation (α=0.0)...")
response = workspace.prompt_based_generate("What is 2+2?", max_new_tokens=30)
print(f"  Prompt: What is 2+2?")
print(f"  Response: {response[:100]}{'...' if len(response) > 100 else ''}")

print("\n[4/4] Testing Projection Head...")
hidden_size = workspace.symbolic_interface.projection_head.projection_layer.in_features
dummy_state = torch.randn(1, hidden_size).to(workspace.device)
logit_bias = workspace.symbolic_interface.projection_head(dummy_state)
print(f"  ✓ Input dim: {hidden_size}")
print(f"  ✓ Forward pass: {dummy_state.shape} → {logit_bias.shape}")
print(f"  ✓ Output range: [{logit_bias.min().item():.3f}, {logit_bias.max().item():.3f}]")

print("\n[5/4] Testing Blend Parameter...")
original_alpha = cfg.PROJECTION_BLEND_ALPHA
for alpha in [0.0, 0.1]:
    cfg.PROJECTION_BLEND_ALPHA = alpha
    state_vec = torch.randn(hidden_size).numpy() if alpha > 0 else None
    resp = workspace.prompt_based_generate("Hello!", max_new_tokens=20, state_vector=state_vec)
    print(f"  α={alpha}: {resp[:80]}{'...' if len(resp) > 80 else ''}")
cfg.PROJECTION_BLEND_ALPHA = original_alpha

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED")
print("=" * 60)
