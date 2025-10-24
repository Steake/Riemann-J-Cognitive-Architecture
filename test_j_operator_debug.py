#!/usr/bin/env python3
"""Debug script to trace J-operator iteration behavior."""
import sys

sys.path.insert(0, "src")

import time

import numpy as np
import torch

from riemann_j.architecture import SymbolicInterface
from riemann_j.config import *
from riemann_j.pn_driver import PredictionErrorSignal
from riemann_j.shared_resources import device, model, tokenizer


def debug_j_operator():
    print("Initializing symbolic interface...")
    symbolic_interface = SymbolicInterface()

    print("Creating high PN signal...")
    pn_signal = PredictionErrorSignal(
        timestamp=time.time(),
        source="Test",
        priority=10,
        p_n=0.95,
        uncompressed_data={"t": 25.0, "steps_since_zero": 500000},
    )

    print("\nStarting J-operator iteration with diagnostics...")
    anomaly_desc = f"Internal system alert. PN={pn_signal.p_n:.4f}. Source: t={pn_signal.uncompressed_data['t']:.2f}."
    a_current_tensor = torch.tensor(symbolic_interface.encoder(anomaly_desc), device=device)

    print(f"Initial state norm: {torch.norm(a_current_tensor).item():.4f}")

    # Get first layer
    if hasattr(model, "model"):
        first_layer = model.model.layers[0]
        rotary_emb = getattr(model.model, "rotary_emb", None)
    else:
        first_layer = model.transformer.h[0]
        rotary_emb = None

    position_ids = torch.arange(0, 1, dtype=torch.long, device=device).unsqueeze(0)

    print("\nIteration details (first 10):")
    for i in range(min(10, J_OPERATOR_MAX_ITERATIONS)):
        # Recompute position embeddings
        if rotary_emb is not None:
            position_embeddings = rotary_emb(
                a_current_tensor.unsqueeze(0).unsqueeze(0), position_ids
            )
        else:
            position_embeddings = None

        with torch.no_grad():
            layer_output = first_layer(
                a_current_tensor.unsqueeze(0).unsqueeze(0),
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            a_target_tensor = layer_output[0].squeeze(0)

        distance = torch.norm(a_target_tensor - a_current_tensor).item()
        current_magnitude = torch.norm(a_current_tensor).item()

        relative_epsilon = J_OPERATOR_RELATIVE_EPSILON * current_magnitude
        absolute_epsilon = J_OPERATOR_STABILITY_EPSILON
        practical_convergence = distance < 0.01

        # Learning rate calculation (matching architecture.py)
        distance_normalized = distance / current_magnitude
        lr_scale = 0.5 + 0.5 * np.tanh(2.0 * (distance_normalized - 0.05))
        adaptive_lr = J_OPERATOR_INITIAL_LR * max(0.05, lr_scale)

        print(
            f"  Iter {i+1:3d}: distance={distance:.6f}, magnitude={current_magnitude:.4f}, "
            f"lr={adaptive_lr:.4f}, rel_eps={relative_epsilon:.6f}"
        )

        if distance < relative_epsilon or distance < absolute_epsilon or practical_convergence:
            print(f"\n✓ CONVERGED at iteration {i+1}")
            print(f"  Final distance: {distance:.6f}")
            print(f"  Convergence type: ", end="")
            if distance < absolute_epsilon:
                print("absolute")
            elif distance < relative_epsilon:
                print("relative")
            else:
                print("practical")
            return True

        # Update state
        a_current_tensor = a_current_tensor + adaptive_lr * (a_target_tensor - a_current_tensor)

    print(f"\n✗ Did not converge in first 10 iterations")
    print(f"  Last distance: {distance:.6f}")
    print(
        f"  Thresholds: absolute={absolute_epsilon:.6f}, relative={relative_epsilon:.6f}, practical=0.01"
    )
    return False


if __name__ == "__main__":
    debug_j_operator()
