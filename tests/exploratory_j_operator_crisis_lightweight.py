#!/usr/bin/env python3
"""
Lightweight J-Operator Crisis State Analysis (No Model Loading)

Tests J-Operator convergence dynamics using synthetic latent states instead of
loading the full 3.8B parameter model. Focuses purely on the mathematical
properties of crisis resolution vs normal operation.

WHY: Model loading is too expensive for exploratory testing. This version tests
the core hypothesis: does J-Operator convergence create distinct attractor dynamics?
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

from riemann_j.pn_driver import PredictionErrorSignal

warnings.filterwarnings("ignore")

# ============================================================================
# MOCK COMPONENTS (to avoid loading 3.8B model)
# ============================================================================


class MockEncoder:
    """Generates realistic latent vectors without model inference."""

    def __init__(self, dim=768):
        self.dim = dim
        np.random.seed(42)

    def encode(self, text: str) -> np.ndarray:
        """Generate pseudo-random but deterministic latent vector."""
        # Use hash of text for reproducibility
        seed = hash(text) % (2**32)
        rng = np.random.RandomState(seed)

        # Generate latent vector with realistic statistics
        base = rng.randn(self.dim) * 0.1
        # Add some structure based on text length and content
        length_bias = len(text) / 100.0
        base += rng.randn(self.dim) * length_bias

        return base


class MockJOperator:
    """Simulates J-Operator convergence without full model."""

    def __init__(self, dim=768):
        self.dim = dim
        self.initial_lr = 0.5
        self.adaptive_rate = 2.0
        self.epsilon = 1e-4
        self.max_iter = 100

    def resolve(self, pn_signal) -> dict:
        """Simulate J-Operator resolution with realistic convergence dynamics."""
        # Initialize from anomaly encoding
        anomaly_text = f"PN={pn_signal.p_n:.4f}_t={pn_signal.uncompressed_data['t']:.2f}"
        encoder = MockEncoder(self.dim)
        a_current = encoder.encode(anomaly_text)

        trajectory = [a_current.copy()]

        # Simulate convergence toward an attractor
        # Create a target that represents "resolved crisis state"
        target_seed = int(pn_signal.p_n * 1000)
        rng = np.random.RandomState(target_seed)
        a_target_base = rng.randn(self.dim) * 0.05  # Smaller magnitude for crisis states

        for i in range(self.max_iter):
            # Simulate one layer of transformer processing
            # Add non-linearity and pull toward target
            a_target = a_target_base + rng.randn(self.dim) * 0.01

            distance = np.linalg.norm(a_target - a_current)

            if distance < self.epsilon:
                # Converged
                lyapunov = self._compute_lyapunov(trajectory)
                return {
                    "representation": a_current,
                    "status": "CONVERGED",
                    "lyapunov": lyapunov,
                    "iterations": i,
                    "converged": True,
                }

            # Adaptive learning rate
            lr = self.initial_lr / (1.0 + self.adaptive_rate * distance)
            a_current = a_current + lr * (a_target - a_current)
            trajectory.append(a_current.copy())

        # Max iterations reached
        lyapunov = self._compute_lyapunov(trajectory)
        return {
            "representation": a_current,
            "status": "ITER_LIMIT_EXCEEDED",
            "lyapunov": lyapunov,
            "iterations": self.max_iter,
            "converged": False,
        }

    def _compute_lyapunov(self, trajectory):
        """Compute Lyapunov exponent from trajectory."""
        if len(trajectory) < 2:
            return 0.0

        distances = [
            np.linalg.norm(trajectory[i + 1] - trajectory[i]) for i in range(len(trajectory) - 1)
        ]

        # Filter out near-zero distances
        valid_distances = [d for d in distances if d > 1e-9]

        if not valid_distances:
            return -np.inf

        return np.mean(np.log(valid_distances))


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("=" * 80)
print("J-OPERATOR CRISIS STATE ANALYSIS (LIGHTWEIGHT)")
print("=" * 80)
print()

encoder = MockEncoder(dim=768)
j_operator = MockJOperator(dim=768)

# ============================================================================
# PHASE 1: Collect Baseline User States
# ============================================================================
print("PHASE 1: Collecting baseline user states")
print("-" * 80)

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

user_states = []
for i, text in enumerate(user_inputs):
    state = encoder.encode(text)
    user_states.append(state)
    print(f"  [{i+1:2d}/20] Collected user state")

print(f"\n✓ Collected {len(user_states)} baseline user states")
print()

# ============================================================================
# PHASE 2: Force J-Operator Crises
# ============================================================================
print("PHASE 2: Forcing J-Operator crises")
print("-" * 80)

j_states = []
j_metadata = []
pn_values = [0.91, 0.93, 0.95, 0.97, 0.99]

for i, pn_val in enumerate(pn_values):
    print(f"  [{i+1}/5] Injecting high PN signal (PN={pn_val:.2f})...")

    fake_pn_signal = PredictionErrorSignal(
        timestamp=time.time(),
        source="CRISIS_TEST",
        priority=1,
        p_n=pn_val,
        uncompressed_data={"t": 100.0 + i * 10, "steps_since_zero": 1000 + i * 100},
    )

    result = j_operator.resolve(fake_pn_signal)
    j_states.append(result["representation"])
    j_metadata.append(result)

    status_symbol = "✓" if result["converged"] else "⚠"
    print(f"        {status_symbol} Status: {result['status']}")
    print(f"          Lyapunov: {result['lyapunov']:.6f}")
    print(f"          Iterations: {result['iterations']}")

print(f"\n✓ Generated {len(j_states)} J-Operator crisis states")
print()

# ============================================================================
# PHASE 3: Latent Space Analysis
# ============================================================================
print("PHASE 3: Analyzing latent space separation")
print("-" * 80)

# Stack all representations
user_reps = np.vstack(user_states)
j_reps = np.vstack(j_states)
all_reps = np.vstack([user_reps, j_reps])

# Create labels (0 = user, 1 = j-operator)
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
sep_ratio = between_distances / ((user_distances + j_distances) / 2)
print(f"  Separation ratio: {sep_ratio:.4f}")
print()

# Silhouette score
silhouette = silhouette_score(projected, labels)
print(f"SILHOUETTE SCORE: {silhouette:.4f}")
print("  (Range: -1 to 1, higher = better separation)")
print()

# Lyapunov analysis
print("LYAPUNOV EXPONENT ANALYSIS:")
lyapunov_vals = [m["lyapunov"] for m in j_metadata]
print(f"  Mean: {np.mean(lyapunov_vals):.6f}")
print(f"  Std:  {np.std(lyapunov_vals):.6f}")
print(f"  Min:  {np.min(lyapunov_vals):.6f}")
print(f"  Max:  {np.max(lyapunov_vals):.6f}")

negative_count = sum(1 for v in lyapunov_vals if v < 0)
print(f"  Negative (stable): {negative_count}/{len(lyapunov_vals)}")
print()

# Convergence statistics
converged_count = sum(1 for m in j_metadata if m["converged"])
print(f"CONVERGENCE STATISTICS:")
print(f"  Converged: {converged_count}/{len(j_metadata)}")
avg_iter = np.mean([m["iterations"] for m in j_metadata])
print(f"  Average iterations: {avg_iter:.1f}")
print()

# Save results
output_file = Path(__file__).parent.parent / "crisis_analysis_results.npz"
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
# PHASE 4: Statistical Significance Test
# ============================================================================
print("PHASE 4: Statistical significance testing")
print("-" * 80)

# Permutation test for separation
n_permutations = 1000
observed_sep = sep_ratio

print(f"Running {n_permutations} permutations...")
permuted_seps = []

for _ in range(n_permutations):
    # Shuffle labels
    shuffled_labels = np.random.permutation(labels)

    # Recompute separation with shuffled labels
    shuffled_user_proj = projected[shuffled_labels == 0]
    shuffled_j_proj = projected[shuffled_labels == 1]

    if len(shuffled_user_proj) > 0 and len(shuffled_j_proj) > 0:
        user_dist = cdist(shuffled_user_proj, shuffled_user_proj).mean()
        j_dist = cdist(shuffled_j_proj, shuffled_j_proj).mean()
        between_dist = cdist(shuffled_user_proj, shuffled_j_proj).mean()

        perm_ratio = between_dist / ((user_dist + j_dist) / 2)
        permuted_seps.append(perm_ratio)

p_value = np.mean([s >= observed_sep for s in permuted_seps])

print(f"  Observed separation ratio: {observed_sep:.4f}")
print(f"  Mean permuted ratio: {np.mean(permuted_seps):.4f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"  ✓ STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print(f"  ✗ NOT SIGNIFICANT (p >= 0.05)")
print()

# ============================================================================
# CONCLUSIONS
# ============================================================================
print("=" * 80)
print("ANALYSIS CONCLUSIONS")
print("=" * 80)
print()

print("Key Findings:")
print()

# Criterion 1: Cluster separation
if sep_ratio > 1.2:
    print("✓ DISTINCT CLUSTERS: J-Operator states form a separate regime")
    print(f"  (Separation ratio: {sep_ratio:.2f} > 1.2)")
elif sep_ratio > 1.05:
    print("⚠ MODERATE SEPARATION: J-Operator states partially distinct")
    print(f"  (Separation ratio: {sep_ratio:.2f})")
else:
    print("✗ WEAK SEPARATION: J-Operator states overlap with user states")
    print(f"  (Separation ratio: {sep_ratio:.2f} <= 1.05)")
print()

# Criterion 2: Lyapunov stability
negative_ratio = sum(1 for v in lyapunov_vals if v < 0) / len(lyapunov_vals)
if negative_ratio > 0.6:
    print("✓ CONVERGENT DYNAMICS: Majority of J-Operator resolutions are stable")
    print(f"  ({negative_ratio*100:.0f}% negative Lyapunov exponents)")
elif negative_ratio > 0.3:
    print("⚠ MIXED DYNAMICS: Some convergence, some chaos")
    print(f"  ({negative_ratio*100:.0f}% negative Lyapunov exponents)")
else:
    print("✗ UNSTABLE DYNAMICS: J-Operator is not converging")
    print(f"  ({negative_ratio*100:.0f}% negative Lyapunov exponents)")
print()

# Criterion 3: Statistical significance
if p_value < 0.05:
    print(f"✓ STATISTICALLY SIGNIFICANT: p = {p_value:.4f}")
else:
    print(f"✗ NOT SIGNIFICANT: p = {p_value:.4f}")
print()

# Criterion 4: Silhouette score
if silhouette > 0.3:
    print(f"✓ STRONG CLUSTER QUALITY: Silhouette = {silhouette:.3f}")
elif silhouette > 0.1:
    print(f"⚠ MODERATE CLUSTER QUALITY: Silhouette = {silhouette:.3f}")
else:
    print(f"✗ POOR CLUSTER QUALITY: Silhouette = {silhouette:.3f}")
print()

print("VERDICT:")
print()

score = 0
if sep_ratio > 1.2:
    score += 1
if negative_ratio > 0.6:
    score += 1
if p_value < 0.05:
    score += 1
if silhouette > 0.3:
    score += 1

if score >= 3:
    print("✓✓✓ J-OPERATOR CREATES GENUINELY DISTINCT COGNITIVE REGIME")
    print("    The mathematical friction under crisis produces measurably")
    print("    different attractor dynamics than normal user interaction.")
    print("    The A ≠ s axiom holds under stress.")
elif score >= 2:
    print("⚠⚠ MODERATE DISTINCTION")
    print("    J-Operator states show some separation but results are mixed.")
    print("    May need tuning of convergence parameters or more data.")
else:
    print("✗✗✗ INSUFFICIENT DISTINCTION")
    print("    J-Operator states are not meaningfully different from user states.")
    print("    The 'crisis resolution' may be just noise, not a cognitive shift.")

print()
print("=" * 80)
print()
print("NOTE: This analysis uses synthetic latent states (no model loading).")
print("      For production validation, run with the full model.")
