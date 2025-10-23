#!/usr/bin/env python3
"""
Exploratory Test: J-Operator Crisis State Analysis

Tests whether J-Operator-induced crisis states are meaningfully distinct from
normal user-driven states in the latent space. This is a critical test of the
A ≠ s axiom under stress conditions.

WHY: If J-Operator states aren't distinct from user states, the "crisis resolution"
is just fancy noise. If they cluster separately with different Lyapunov characteristics,
the mathematical friction genuinely creates a different cognitive regime.
"""
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import warnings

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from riemann_j.architecture import CognitiveWorkspace, SyntheticState
from riemann_j.pn_driver import PredictionErrorSignal

warnings.filterwarnings("ignore")

print("=" * 80)
print("J-OPERATOR CRISIS STATE ANALYSIS")
print("=" * 80)
print()

# Initialize workspace
print("Initializing CognitiveWorkspace...")
workspace = CognitiveWorkspace()
print("✓ Workspace initialized")
print()

# ============================================================================
# PHASE 1: Collect Baseline User States
# ============================================================================
print("PHASE 1: Collecting baseline user states")
print("-" * 80)

user_id = "crisis_test_user"
user_states = []
user_inputs = [
    "Hello, how are you today?",
    "Tell me about the weather.",
    "What is your favorite color?",
    "Explain quantum mechanics.",
    "I'm feeling curious about mathematics.",
    "Can you help me understand recursion?",
    "What makes a good algorithm?",
    "Describe the concept of entropy.",
    "How do neural networks learn?",
    "What is consciousness?",
    "Tell me a story about a robot.",
    "Explain the halting problem.",
    "What is the meaning of life?",
    "Describe a fractal.",
    "How does memory work?",
    "What is infinity?",
    "Explain the Riemann hypothesis.",
    "Tell me about strange attractors.",
    "What is emergence?",
    "How do you feel about uncertainty?",
]

for i, text in enumerate(user_inputs):
    try:
        _, state = workspace.process_user_input(user_id, text)
        user_states.append(
            {
                "representation": state.latent_representation,
                "timestamp": state.timestamp,
                "source": state.source_trigger,
            }
        )
        print(f"  [{i+1:2d}/20] Collected user state (PN={state.p_n_at_creation:.4f})")
    except Exception as e:
        print(f"  [ERROR] Failed to process input {i+1}: {e}")

print(f"\n✓ Collected {len(user_states)} baseline user states")
print()

# ============================================================================
# PHASE 2: Force J-Operator Crises
# ============================================================================
print("PHASE 2: Forcing J-Operator crises")
print("-" * 80)

j_states = []
pn_values = [0.91, 0.93, 0.95, 0.97, 0.99]

for i, pn_val in enumerate(pn_values):
    print(f"  [{i+1}/5] Injecting high PN signal (PN={pn_val:.2f})...")

    # Create fake high-PN signal to trigger J-Operator
    fake_pn_signal = PredictionErrorSignal(
        timestamp=time.time(),
        source="CRISIS_TEST",
        priority=1,
        p_n=pn_val,
        uncompressed_data={"t": 100.0 + i * 10, "steps_since_zero": 1000 + i * 100},
    )

    try:
        # Trigger J-Operator resolution directly
        j_state = workspace._j_operator_resolve(fake_pn_signal)
        j_states.append(
            {
                "representation": j_state.latent_representation,
                "timestamp": j_state.timestamp,
                "source": j_state.source_trigger,
                "status": j_state.status,
                "lyapunov": j_state.analysis.get("lyapunov_exp", None),
                "iterations": j_state.analysis.get("iterations", None),
                "pn": j_state.p_n_at_creation,
            }
        )

        status_symbol = "✓" if j_state.status == "CONVERGED" else "⚠"
        print(f"        {status_symbol} Status: {j_state.status}")
        print(f"          Lyapunov: {j_state.analysis.get('lyapunov_exp', 'N/A'):.6f}")
        print(f"          Iterations: {j_state.analysis.get('iterations', 'N/A')}")
    except Exception as e:
        print(f"  [ERROR] J-Operator resolution failed: {e}")
        import traceback

        traceback.print_exc()

print(f"\n✓ Generated {len(j_states)} J-Operator crisis states")
print()

# ============================================================================
# PHASE 3: Latent Space Analysis
# ============================================================================
print("PHASE 3: Analyzing latent space separation")
print("-" * 80)

if len(user_states) > 0 and len(j_states) > 0:
    # Stack all representations
    user_reps = np.vstack([s["representation"] for s in user_states])
    j_reps = np.vstack([s["representation"] for s in j_states])
    all_reps = np.vstack([user_reps, j_reps])

    # Create labels (0 = user, 1 = j-operator)
    labels = np.array([0] * len(user_states) + [1] * len(j_states))

    print(f"User states shape: {user_reps.shape}")
    print(f"J-Operator states shape: {j_reps.shape}")
    print()

    # PCA reduction for visualization and analysis
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

    # Measure separation
    centroid_distance = np.linalg.norm(user_centroid - j_centroid)

    # Cross-distance matrix
    user_proj = projected[: len(user_states)]
    j_proj = projected[len(user_states) :]

    # Within-cluster distances
    user_distances = cdist(user_proj, user_proj).mean()
    j_distances = cdist(j_proj, j_proj).mean()

    # Between-cluster distances
    between_distances = cdist(user_proj, j_proj).mean()

    print("DISTANCE METRICS:")
    print(f"  Centroid separation: {centroid_distance:.4f}")
    print(f"  Mean within-user distance: {user_distances:.4f}")
    print(f"  Mean within-J distance: {j_distances:.4f}")
    print(f"  Mean between-cluster distance: {between_distances:.4f}")
    print(f"  Separation ratio: {between_distances / ((user_distances + j_distances) / 2):.4f}")
    print()

    # Silhouette score (measure of cluster separation)
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(projected, labels)
        print(f"SILHOUETTE SCORE: {silhouette:.4f}")
        print("  (Range: -1 to 1, higher = better separation)")
        print()

    # Lyapunov analysis
    print("LYAPUNOV EXPONENT ANALYSIS:")
    lyapunov_vals = [s["lyapunov"] for s in j_states if s["lyapunov"] is not None]
    if lyapunov_vals:
        print(f"  Mean: {np.mean(lyapunov_vals):.6f}")
        print(f"  Std:  {np.std(lyapunov_vals):.6f}")
        print(f"  Min:  {np.min(lyapunov_vals):.6f}")
        print(f"  Max:  {np.max(lyapunov_vals):.6f}")

        negative_count = sum(1 for v in lyapunov_vals if v < 0)
        print(f"  Negative (stable): {negative_count}/{len(lyapunov_vals)}")
    print()

    # Save projection for visualization
    output_file = Path(__file__).parent.parent / "crisis_analysis_results.npz"
    np.savez(
        output_file,
        user_states=user_proj,
        j_states=j_proj,
        user_centroid=user_centroid,
        j_centroid=j_centroid,
        variance_explained=pca.explained_variance_ratio_,
        lyapunov_values=lyapunov_vals if lyapunov_vals else np.array([]),
    )
    print(f"✓ Results saved to: {output_file}")
    print()

else:
    print("[ERROR] Insufficient data collected for analysis")
    print()

# ============================================================================
# PHASE 4: User Attractor Persistence Test
# ============================================================================
print("PHASE 4: Testing attractor persistence after crisis")
print("-" * 80)

# Get the user attractor state before crisis
user_attractor = workspace.get_or_create_user(user_id)
pre_crisis_history_size = len(user_attractor.state_history)

if hasattr(user_attractor.gmm, "means_"):
    pre_crisis_means = user_attractor.gmm.means_.copy()
    print(f"Pre-crisis attractor history: {pre_crisis_history_size} states")
    print(f"Pre-crisis GMM components: {pre_crisis_means.shape[0]}")
    print()

    # Add more user interactions post-crisis
    post_crisis_inputs = [
        "Let's continue our conversation.",
        "What do you think about what just happened?",
        "Are you still there?",
        "Everything okay?",
        "Tell me more about yourself.",
    ]

    for text in post_crisis_inputs:
        workspace.process_user_input(user_id, text)

    post_crisis_history_size = len(user_attractor.state_history)

    if hasattr(user_attractor.gmm, "means_"):
        post_crisis_means = user_attractor.gmm.means_

        # Measure drift in GMM centroids
        mean_drift = np.linalg.norm(post_crisis_means - pre_crisis_means, axis=1).mean()

        print(f"Post-crisis attractor history: {post_crisis_history_size} states")
        print(f"Mean centroid drift: {mean_drift:.6f}")
        print()

        if mean_drift < 0.1:
            print("✓ Attractor field STABLE (minimal drift)")
        elif mean_drift < 0.5:
            print("⚠ Attractor field MODERATELY AFFECTED")
        else:
            print("✗ Attractor field SIGNIFICANTLY ALTERED")
else:
    print("⚠ GMM not trained yet, skipping persistence test")

print()

# ============================================================================
# CONCLUSIONS
# ============================================================================
print("=" * 80)
print("ANALYSIS CONCLUSIONS")
print("=" * 80)
print()

if len(j_states) > 0 and len(user_states) > 0:
    print("Key Findings:")
    print()

    # Criterion 1: Cluster separation
    if "between_distances" in locals():
        sep_ratio = between_distances / ((user_distances + j_distances) / 2)
        if sep_ratio > 1.2:
            print("✓ DISTINCT CLUSTERS: J-Operator states form a separate regime")
            print(f"  (Separation ratio: {sep_ratio:.2f} > 1.2)")
        else:
            print("✗ WEAK SEPARATION: J-Operator states overlap with user states")
            print(f"  (Separation ratio: {sep_ratio:.2f} <= 1.2)")
        print()

    # Criterion 2: Lyapunov stability
    if lyapunov_vals:
        negative_ratio = sum(1 for v in lyapunov_vals if v < 0) / len(lyapunov_vals)
        if negative_ratio > 0.6:
            print("✓ CONVERGENT DYNAMICS: Majority of J-Operator resolutions are stable")
            print(f"  ({negative_ratio*100:.0f}% negative Lyapunov exponents)")
        else:
            print("⚠ UNSTABLE DYNAMICS: J-Operator may not be converging properly")
            print(f"  ({negative_ratio*100:.0f}% negative Lyapunov exponents)")
        print()

    # Criterion 3: Variance capture
    if "pca" in locals():
        var_first_3 = pca.explained_variance_ratio_[:3].sum()
        if var_first_3 > 0.5:
            print(f"✓ LOW-DIM STRUCTURE: First 3 PCs capture {var_first_3*100:.1f}% variance")
        else:
            print(f"⚠ HIGH-DIM SPREAD: First 3 PCs only capture {var_first_3*100:.1f}% variance")
        print()

    print("VERDICT:")
    print()
    if sep_ratio > 1.2 and negative_ratio > 0.6:
        print("✓✓✓ J-OPERATOR CREATES GENUINELY DISTINCT COGNITIVE REGIME")
        print("    The mathematical friction under crisis produces measurably")
        print("    different attractor dynamics than normal user interaction.")
        print("    The A ≠ s axiom holds under stress.")
    elif sep_ratio > 1.1:
        print("⚠⚠ MODERATE DISTINCTION")
        print("    J-Operator states show some separation but overlap exists.")
        print("    May need tuning of convergence parameters.")
    else:
        print("✗✗✗ INSUFFICIENT DISTINCTION")
        print("    J-Operator states are not meaningfully different from user states.")
        print("    The 'crisis resolution' is just noise, not a cognitive shift.")
else:
    print("✗ INSUFFICIENT DATA: Unable to reach conclusions")

print()
print("=" * 80)

# Cleanup
workspace.close()
