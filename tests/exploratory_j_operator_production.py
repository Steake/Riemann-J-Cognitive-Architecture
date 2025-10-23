#!/usr/bin/env python3
"""
J-Operator Crisis Analysis - Production Model (Phi-3.5-mini-instruct)

Tests the architecture with the full production model (3.8B params) using
IMPROVED convergence heuristics. This validates whether the lightweight
model results hold at production scale.

WHY: The absolute epsilon threshold (1e-6) is unrealistic for large models.
This version uses relative convergence and Lyapunov-based early stopping.

WARNING: This will be slow on CPU. Expect 5-10 minutes for full run.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from riemann_j.architecture import CognitiveWorkspace
from riemann_j.pn_driver import PredictionErrorSignal

print("=" * 80)
print("J-OPERATOR CRISIS ANALYSIS - PRODUCTION MODEL")
print("Phi-3.5-mini-instruct (3.8B parameters)")
print("=" * 80)
print()

# ============================================================================
# PATCH: Improved J-Operator Convergence
# ============================================================================


def improved_j_operator_resolve(workspace_instance, pn_signal):
    """
    Improved J-Operator with relative convergence and Lyapunov early stopping.

    Changes from original:
    1. Relative epsilon: 0.01 * ||a_current|| instead of fixed 1e-6
    2. Lyapunov stability check: if negative for 10 consecutive iters, converge
    3. Increased max iterations to 100
    """
    from riemann_j.architecture import SyntheticState

    anomaly_desc = f"Internal system alert. PN={pn_signal.p_n:.4f}. Source: t={pn_signal.uncompressed_data['t']:.2f}."

    # Encode anomaly
    a_current_tensor = torch.tensor(
        workspace_instance.symbolic_interface.encoder(anomaly_desc), device="cpu"
    )

    trajectory = [a_current_tensor.cpu().numpy()]
    lyapunov_history = []

    MAX_ITERATIONS = 100
    RELATIVE_EPSILON_FACTOR = 0.01  # 1% of current magnitude
    LYAPUNOV_WINDOW = 10  # Check stability over last N iterations

    for i in range(MAX_ITERATIONS):
        with torch.no_grad():
            # Transform through first layer
            from riemann_j.shared_resources import model

            a_target_tensor = (
                model.transformer.h[0](a_current_tensor.unsqueeze(0).unsqueeze(0))[0]
                .squeeze(0)
                .squeeze(0)
            )

        distance = torch.norm(a_target_tensor - a_current_tensor).item()
        current_magnitude = torch.norm(a_current_tensor).item()

        # Relative convergence check
        relative_epsilon = RELATIVE_EPSILON_FACTOR * current_magnitude

        if distance < relative_epsilon:
            lyapunov_exp = _compute_lyapunov(trajectory)
            return SyntheticState(
                timestamp=time.time(),
                latent_representation=a_target_tensor.cpu().numpy(),
                source_trigger="RIEMANN_PN",
                p_n_at_creation=pn_signal.p_n,
                is_j_shift_product=True,
                status="CONVERGED_RELATIVE",
                analysis={
                    "lyapunov_exp": lyapunov_exp,
                    "iterations": i,
                    "final_distance": distance,
                    "convergence_criterion": "relative_epsilon",
                },
            )

        # Lyapunov-based early stopping
        if len(trajectory) >= LYAPUNOV_WINDOW:
            recent_lyapunov = _compute_lyapunov(trajectory[-LYAPUNOV_WINDOW:])
            lyapunov_history.append(recent_lyapunov)

            # If consistently negative, we've stabilized
            if len(lyapunov_history) >= 5 and all(l < -1.0 for l in lyapunov_history[-5:]):
                lyapunov_exp = _compute_lyapunov(trajectory)
                return SyntheticState(
                    timestamp=time.time(),
                    latent_representation=a_current_tensor.cpu().numpy(),
                    source_trigger="RIEMANN_PN",
                    p_n_at_creation=pn_signal.p_n,
                    is_j_shift_product=True,
                    status="CONVERGED_LYAPUNOV",
                    analysis={
                        "lyapunov_exp": lyapunov_exp,
                        "iterations": i,
                        "final_distance": distance,
                        "convergence_criterion": "lyapunov_stability",
                    },
                )

        # Adaptive learning rate
        lr = 0.3 / (1.0 + 5.0 * distance)
        a_current_tensor = a_current_tensor + lr * (a_target_tensor - a_current_tensor)
        trajectory.append(a_current_tensor.cpu().numpy())

    # Max iterations
    lyapunov_exp = _compute_lyapunov(trajectory)
    return SyntheticState(
        timestamp=time.time(),
        latent_representation=a_current_tensor.cpu().numpy(),
        source_trigger="RIEMANN_PN",
        p_n_at_creation=pn_signal.p_n,
        is_j_shift_product=True,
        status="ITER_LIMIT_EXCEEDED",
        analysis={
            "lyapunov_exp": lyapunov_exp,
            "iterations": MAX_ITERATIONS,
            "final_distance": distance,
            "convergence_criterion": "none",
        },
    )


def _compute_lyapunov(trajectory):
    """Compute Lyapunov exponent from trajectory."""
    if len(trajectory) < 2:
        return 0.0

    distances = [
        np.linalg.norm(trajectory[i + 1] - trajectory[i]) for i in range(len(trajectory) - 1)
    ]

    valid_distances = [d for d in distances if d > 1e-9]

    if not valid_distances:
        return -np.inf

    return np.mean(np.log(valid_distances))


# ============================================================================
# INITIALIZE PRODUCTION WORKSPACE
# ============================================================================

print("Initializing production CognitiveWorkspace...")
print("(This loads Phi-3.5-mini-instruct - will take ~30 seconds)")
print()

workspace = CognitiveWorkspace()

print("✓ Workspace initialized")
print(f"  Model: microsoft/Phi-3.5-mini-instruct")
print(f"  Device: cpu")
print()

# ============================================================================
# PHASE 1: Collect Baseline User States (Reduced Sample)
# ============================================================================
print("PHASE 1: Collecting baseline user states")
print("-" * 80)

user_inputs = [
    "Hello, explain yourself.",
    "What is quantum mechanics?",
    "Tell me about recursion.",
    "Describe neural networks.",
    "What is consciousness?",
    "Explain the halting problem.",
    "What is infinity?",
    "Describe a fractal.",
]

user_states = []
print("Processing user inputs (slower with 3.8B model)...")
for i, text in enumerate(user_inputs):
    print(f"  [{i+1}/{len(user_inputs)}] Processing: {text[:40]}...", end=" ", flush=True)
    start = time.time()

    _, state = workspace.process_user_input("test_user", text)
    user_states.append(
        {"representation": state.latent_representation, "timestamp": state.timestamp}
    )

    elapsed = time.time() - start
    print(f"({elapsed:.1f}s)")

print(f"\n✓ Collected {len(user_states)} baseline user states")
print()

# ============================================================================
# PHASE 2: Force J-Operator Crises (Improved Convergence)
# ============================================================================
print("PHASE 2: J-Operator crises with IMPROVED convergence heuristic")
print("-" * 80)

j_states = []
j_metadata = []
pn_values = [0.92, 0.96]  # Just 2 to keep runtime reasonable

print("Running improved J-Operator resolutions...")
for i, pn_val in enumerate(pn_values):
    print(f"  [{i+1}/2] Crisis at PN={pn_val:.2f}...", end=" ", flush=True)

    fake_pn_signal = PredictionErrorSignal(
        timestamp=time.time(),
        source="PRODUCTION_CRISIS_TEST",
        priority=1,
        p_n=pn_val,
        uncompressed_data={"t": 100.0 + i * 10, "steps_since_zero": 1000 + i * 100},
    )

    start = time.time()
    # Use improved resolution
    j_state = improved_j_operator_resolve(workspace, fake_pn_signal)
    elapsed = time.time() - start

    j_states.append(
        {
            "representation": j_state.latent_representation,
            "timestamp": j_state.timestamp,
            "status": j_state.status,
            "analysis": j_state.analysis,
            "pn": j_state.p_n_at_creation,
        }
    )

    status_symbol = "✓" if "CONVERGED" in j_state.status else "⚠"
    print(f"{status_symbol} ({elapsed:.1f}s)")
    print(f"        Status: {j_state.status}")
    print(f"        Lyapunov: {j_state.analysis['lyapunov_exp']:.6f}")
    print(f"        Iterations: {j_state.analysis['iterations']}")
    print(f"        Final distance: {j_state.analysis['final_distance']:.6f}")
    print(f"        Criterion: {j_state.analysis['convergence_criterion']}")

print(f"\n✓ Generated {len(j_states)} J-Operator crisis states")
print()

# ============================================================================
# PHASE 3: Latent Space Analysis
# ============================================================================
print("PHASE 3: Analyzing latent space separation")
print("-" * 80)

user_reps = np.vstack([s["representation"] for s in user_states])
j_reps = np.vstack([s["representation"] for s in j_states])
all_reps = np.vstack([user_reps, j_reps])

labels = np.array([0] * len(user_states) + [1] * len(j_states))

print(f"User states shape: {user_reps.shape}")
print(f"J-Operator states shape: {j_reps.shape}")
print()

# PCA reduction
print("Performing PCA dimensionality reduction...")
pca = PCA(n_components=min(10, all_reps.shape[0], all_reps.shape[1]))
projected = pca.fit_transform(all_reps)

print(f"  PCA components: {pca.n_components_}")
print(f"  Variance explained (first 3): {pca.explained_variance_ratio_[:3]}")
print(f"  Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
print()

# Compute centroids
user_centroid = projected[: len(user_states)].mean(axis=0)
j_centroid = projected[len(user_states) :].mean(axis=0)

centroid_distance = np.linalg.norm(user_centroid - j_centroid)

# Distance analysis
user_proj = projected[: len(user_states)]
j_proj = projected[len(user_states) :]

user_distances = cdist(user_proj, user_proj).mean()
j_distances = cdist(j_proj, j_proj).mean()
between_distances = cdist(user_proj, j_proj).mean()

print("DISTANCE METRICS:")
print(f"  Centroid separation: {centroid_distance:.4f}")
print(f"  Mean within-user distance: {user_distances:.4f}")
print(f"  Mean within-J distance: {j_distances:.4f}")
print(f"  Mean between-cluster distance: {between_distances:.4f}")
sep_ratio = between_distances / ((user_distances + j_distances) / 2)
print(f"  Separation ratio: {sep_ratio:.4f}")
print()

# Silhouette score
silhouette = silhouette_score(projected, labels)
print(f"SILHOUETTE SCORE: {silhouette:.4f}")
print()

# Lyapunov analysis
print("LYAPUNOV EXPONENT ANALYSIS:")
lyapunov_vals = [s["analysis"]["lyapunov_exp"] for s in j_states]
print(f"  Mean: {np.mean(lyapunov_vals):.6f}")
print(f"  Std:  {np.std(lyapunov_vals):.6f}")
print(f"  Min:  {np.min(lyapunov_vals):.6f}")
print(f"  Max:  {np.max(lyapunov_vals):.6f}")

negative_count = sum(1 for v in lyapunov_vals if v < 0)
print(f"  Negative (stable): {negative_count}/{len(lyapunov_vals)}")
print()

# Convergence statistics
print("CONVERGENCE STATISTICS:")
for criterion in ["relative_epsilon", "lyapunov_stability", "none"]:
    count = sum(1 for s in j_states if s["analysis"]["convergence_criterion"] == criterion)
    if count > 0:
        print(f"  {criterion}: {count}/{len(j_states)}")
avg_iter = np.mean([s["analysis"]["iterations"] for s in j_states])
print(f"  Average iterations: {avg_iter:.1f}")
print()

# Save results
output_file = Path(__file__).parent.parent / "crisis_analysis_production.npz"
np.savez(
    output_file,
    user_states=user_proj,
    j_states=j_proj,
    user_centroid=user_centroid,
    j_centroid=j_centroid,
    variance_explained=pca.explained_variance_ratio_,
    lyapunov_values=lyapunov_vals,
    separation_ratio=sep_ratio,
    silhouette=silhouette,
)
print(f"✓ Results saved to: {output_file}")
print()

# ============================================================================
# COMPARISON TO LIGHTWEIGHT MODEL
# ============================================================================
print("PHASE 4: Comparison to lightweight model results")
print("-" * 80)

try:
    lightweight_results = np.load(Path(__file__).parent.parent / "crisis_analysis_real_model.npz")

    print("DistilGPT-2 (82M) vs Phi-3.5 (3.8B):")
    print()
    print(f"  Separation Ratio:")
    print(f"    DistilGPT-2: {lightweight_results['separation_ratio']:.4f}")
    print(f"    Phi-3.5:     {sep_ratio:.4f}")
    print(f"    Change:      {sep_ratio - lightweight_results['separation_ratio']:+.4f}")
    print()

    print(f"  Silhouette Score:")
    print(f"    DistilGPT-2: {lightweight_results['silhouette']:.4f}")
    print(f"    Phi-3.5:     {silhouette:.4f}")
    print(f"    Change:      {silhouette - lightweight_results['silhouette']:+.4f}")
    print()

    print(f"  First PC Variance:")
    print(f"    DistilGPT-2: {lightweight_results['variance_explained'][0]:.4f}")
    print(f"    Phi-3.5:     {pca.explained_variance_ratio_[0]:.4f}")
    print()

    if sep_ratio > lightweight_results["separation_ratio"]:
        print("✓ SCALE IMPROVES SEPARATION: Larger model = stronger distinction")
    elif sep_ratio > lightweight_results["separation_ratio"] * 0.8:
        print("✓ SEPARATION MAINTAINED: Results hold at production scale")
    else:
        print("⚠ SCALE DEGRADES SEPARATION: Larger model reduces distinction")

except FileNotFoundError:
    print("⚠ Lightweight model results not found for comparison")

print()

# ============================================================================
# CONCLUSIONS
# ============================================================================
print("=" * 80)
print("PRODUCTION MODEL ANALYSIS CONCLUSIONS")
print("=" * 80)
print()

print("Key Findings:")
print()

# Criterion 1: Cluster separation
if sep_ratio > 1.5:
    print("✓✓ STRONG SEPARATION: J-Operator states form a distinct regime")
    print(f"   (Separation ratio: {sep_ratio:.2f} > 1.5)")
elif sep_ratio > 1.2:
    print("✓ DISTINCT CLUSTERS: J-Operator states separable")
    print(f"  (Separation ratio: {sep_ratio:.2f} > 1.2)")
else:
    print("✗ WEAK SEPARATION: Overlapping clusters")
    print(f"  (Separation ratio: {sep_ratio:.2f})")
print()

# Criterion 2: Convergence success
converged_count = sum(1 for s in j_states if "CONVERGED" in s["status"])
conv_rate = converged_count / len(j_states)
if conv_rate > 0.5:
    print(f"✓ IMPROVED CONVERGENCE: {conv_rate*100:.0f}% converged with new heuristic")
else:
    print(f"⚠ CONVERGENCE ISSUES: {conv_rate*100:.0f}% converged")
print()

# Criterion 3: Lyapunov stability
negative_ratio = sum(1 for v in lyapunov_vals if v < 0) / len(lyapunov_vals)
if negative_ratio > 0.8:
    print(f"✓ STABLE DYNAMICS: {negative_ratio*100:.0f}% negative Lyapunov")
else:
    print(f"⚠ UNSTABLE DYNAMICS: {negative_ratio*100:.0f}% negative Lyapunov")
print()

# Criterion 4: Silhouette quality
if silhouette > 0.4:
    print(f"✓✓ EXCELLENT CLUSTERING: Silhouette = {silhouette:.3f}")
elif silhouette > 0.2:
    print(f"✓ GOOD CLUSTERING: Silhouette = {silhouette:.3f}")
else:
    print(f"⚠ WEAK CLUSTERING: Silhouette = {silhouette:.3f}")
print()

print("FINAL VERDICT:")
print()

score = 0
if sep_ratio > 1.5:
    score += 2
elif sep_ratio > 1.2:
    score += 1
if conv_rate > 0.5:
    score += 1
if negative_ratio > 0.8:
    score += 1
if silhouette > 0.3:
    score += 1

if score >= 4:
    print("✓✓✓ PRODUCTION MODEL FULLY VALIDATES ARCHITECTURE")
    print("    The improved convergence heuristic works. Crisis states")
    print("    are measurably distinct at 3.8B parameter scale.")
    print("    The A ≠ s axiom holds in production.")
elif score >= 2:
    print("⚠⚠ PARTIAL VALIDATION")
    print("    Some distinction observed but not as strong as lightweight model.")
    print("    May need further tuning for production scale.")
else:
    print("✗✗✗ PRODUCTION SCALE FAILS")
    print("    Architecture does not maintain distinction at 3.8B parameters.")

print()
print("=" * 80)
print()
print("IMPROVED CONVERGENCE HEURISTICS:")
print("  - Relative epsilon: 0.01 * ||state|| (adaptive to magnitude)")
print("  - Lyapunov early stopping: stable for 5 consecutive iterations")
print("  - Increased max iterations: 100 (was 50)")

workspace.close()
