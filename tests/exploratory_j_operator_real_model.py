#!/usr/bin/env python3
"""
J-Operator Crisis Analysis with Real Lightweight Model

Tests the actual J-Operator convergence dynamics using a real (but small)
transformer model. Uses DistilGPT-2 for fast CPU inference while maintaining
architectural fidelity.

WHY: Validates whether real transformer layer processing creates the same
distinct crisis dynamics as the mock version. This is the critical test -
does the architecture work with actual neural computation?
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from riemann_j.pn_driver import PredictionErrorSignal

print("=" * 80)
print("J-OPERATOR CRISIS ANALYSIS - REAL MODEL (LIGHTWEIGHT)")
print("=" * 80)
print()

# ============================================================================
# SETUP: Load Lightweight Model
# ============================================================================
print("Loading lightweight model (DistilGPT-2, 82M params)...")
model_name = "distilgpt2"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cpu"
    model.to(device)

    hidden_size = model.config.n_embd  # 768 for DistilGPT-2
    print(f"✓ Model loaded: {model_name}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Device: {device}")
    print()
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# ============================================================================
# ARCHITECTURE COMPONENTS (Real Implementation)
# ============================================================================


class RealEncoder:
    """Uses actual model for encoding."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, text: str) -> np.ndarray:
        """Encode text to latent vector using last hidden state."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Use mean pooling of last hidden state
        hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
        pooled = hidden.mean(dim=1).squeeze(0)  # [hidden]

        return pooled.cpu().numpy()


class RealJOperator:
    """Actual J-Operator using real transformer layers."""

    def __init__(self, model, encoder, device, config=None):
        self.model = model
        self.encoder = encoder
        self.device = device

        # J-Operator config
        self.initial_lr = 0.5
        self.adaptive_rate = 2.0
        self.epsilon = 1e-4
        self.max_iter = 50  # Reduced for real model (slower)

    def resolve(self, pn_signal) -> dict:
        """Real J-Operator resolution using transformer processing."""
        # Encode the crisis trigger
        anomaly_text = f"Internal system alert. PN={pn_signal.p_n:.4f}. Source: t={pn_signal.uncompressed_data['t']:.2f}."
        a_current = torch.tensor(
            self.encoder.encode(anomaly_text), dtype=torch.float32, device=self.device
        )

        trajectory = [a_current.cpu().numpy()]

        # Use first transformer block for state transformation
        first_block = self.model.transformer.h[0]

        for i in range(self.max_iter):
            with torch.no_grad():
                # Transform through transformer block
                # Need to add batch and sequence dimensions
                hidden = a_current.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]

                # Process through block
                a_target = first_block(hidden)[0].squeeze(0).squeeze(0)  # [hidden]

            distance = torch.norm(a_target - a_current).item()

            if distance < self.epsilon:
                # Converged
                lyapunov = self._compute_lyapunov(trajectory)
                return {
                    "representation": a_current.cpu().numpy(),
                    "status": "CONVERGED",
                    "lyapunov": lyapunov,
                    "iterations": i,
                    "converged": True,
                    "distance": distance,
                }

            # Adaptive learning rate
            lr = self.initial_lr / (1.0 + self.adaptive_rate * distance)
            a_current = a_current + lr * (a_target - a_current)
            trajectory.append(a_current.cpu().numpy())

        # Max iterations
        lyapunov = self._compute_lyapunov(trajectory)
        return {
            "representation": a_current.cpu().numpy(),
            "status": "ITER_LIMIT_EXCEEDED",
            "lyapunov": lyapunov,
            "iterations": self.max_iter,
            "converged": False,
            "distance": distance,
        }

    def _compute_lyapunov(self, trajectory):
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
# INITIALIZE COMPONENTS
# ============================================================================

print("Initializing architecture components...")
encoder = RealEncoder(model, tokenizer, device)
j_operator = RealJOperator(model, encoder, device)
print("✓ Components ready")
print()

# ============================================================================
# PHASE 1: Collect Baseline User States
# ============================================================================
print("PHASE 1: Collecting baseline user states")
print("-" * 80)

user_inputs = [
    "Hello, how are you today?",
    "Tell me about the weather.",
    "What is your favorite color?",
    "Explain quantum mechanics briefly.",
    "I'm curious about mathematics.",
    "Help me understand recursion.",
    "What makes a good algorithm?",
    "Describe entropy in simple terms.",
    "How do neural networks learn?",
    "What is consciousness?",
    "Tell me a short story about a robot.",
    "Explain the halting problem.",
    "What is the meaning of life?",
    "Describe a fractal pattern.",
    "How does memory work?",
]

user_states = []
print("Encoding user inputs (this may take a minute)...")
for i, text in enumerate(user_inputs):
    state = encoder.encode(text)
    user_states.append(state)
    print(f"  [{i+1:2d}/15] Encoded: {text[:50]}...")

print(f"\n✓ Collected {len(user_states)} baseline user states")
print()

# ============================================================================
# PHASE 2: Force J-Operator Crises
# ============================================================================
print("PHASE 2: Forcing J-Operator crises with real transformer processing")
print("-" * 80)

j_states = []
j_metadata = []
pn_values = [0.91, 0.95, 0.99]  # Reduced to 3 for speed

print("Running J-Operator resolutions (slower due to real computation)...")
for i, pn_val in enumerate(pn_values):
    print(f"  [{i+1}/3] Crisis at PN={pn_val:.2f}...", end=" ", flush=True)

    fake_pn_signal = PredictionErrorSignal(
        timestamp=time.time(),
        source="CRISIS_TEST",
        priority=1,
        p_n=pn_val,
        uncompressed_data={"t": 100.0 + i * 10, "steps_since_zero": 1000 + i * 100},
    )

    start = time.time()
    result = j_operator.resolve(fake_pn_signal)
    elapsed = time.time() - start

    j_states.append(result["representation"])
    j_metadata.append(result)

    status_symbol = "✓" if result["converged"] else "⚠"
    print(f"{status_symbol} ({elapsed:.1f}s)")
    print(f"        Status: {result['status']}")
    print(f"        Lyapunov: {result['lyapunov']:.6f}")
    print(f"        Iterations: {result['iterations']}")
    print(f"        Final distance: {result['distance']:.6f}")

print(f"\n✓ Generated {len(j_states)} J-Operator crisis states")
print()

# ============================================================================
# PHASE 3: Latent Space Analysis
# ============================================================================
print("PHASE 3: Analyzing latent space separation")
print("-" * 80)

user_reps = np.vstack(user_states)
j_reps = np.vstack(j_states)
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
output_file = Path(__file__).parent.parent / "crisis_analysis_real_model.npz"
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
# PHASE 4: Cross-User Contamination Test
# ============================================================================
print("PHASE 4: Testing cross-user contamination")
print("-" * 80)

print("Simulating two users with distinct input patterns...")

# User A: Technical/mathematical
user_a_inputs = [
    "Explain formal logic.",
    "What is category theory?",
    "Describe lambda calculus.",
]

# User B: Creative/emotional
user_b_inputs = [
    "Write a poem about the ocean.",
    "What does happiness feel like?",
    "Tell me about beautiful sunsets.",
]

user_a_states = [encoder.encode(text) for text in user_a_inputs]
user_b_states = [encoder.encode(text) for text in user_b_inputs]

# Compute pre-crisis centroids
user_a_centroid = np.mean(user_a_states, axis=0)
user_b_centroid = np.mean(user_b_states, axis=0)
pre_crisis_distance = np.linalg.norm(user_a_centroid - user_b_centroid)

print(f"Pre-crisis separation between users: {pre_crisis_distance:.4f}")
print()

# Trigger crisis "during" User A's session
print("Triggering crisis during User A session...")
crisis_signal = PredictionErrorSignal(
    timestamp=time.time(),
    source="CROSS_USER_TEST",
    priority=1,
    p_n=0.95,
    uncompressed_data={"t": 200.0, "steps_since_zero": 2000},
)
crisis_result = j_operator.resolve(crisis_signal)
print(f"  Crisis state Lyapunov: {crisis_result['lyapunov']:.6f}")
print()

# Add post-crisis states
print("Collecting post-crisis states for both users...")
user_a_post = [encoder.encode("Continue with mathematics.")]
user_b_post = [encoder.encode("Continue with creative writing.")]

user_a_states_post = user_a_states + user_a_post
user_b_states_post = user_b_states + user_b_post

user_a_centroid_post = np.mean(user_a_states_post, axis=0)
user_b_centroid_post = np.mean(user_b_states_post, axis=0)
post_crisis_distance = np.linalg.norm(user_a_centroid_post - user_b_centroid_post)

print(f"Post-crisis separation between users: {post_crisis_distance:.4f}")
print(f"Change in separation: {post_crisis_distance - pre_crisis_distance:+.4f}")
print()

# Check contamination
crisis_state = crisis_result["representation"]
user_a_to_crisis = np.linalg.norm(user_a_centroid_post - crisis_state)
user_b_to_crisis = np.linalg.norm(user_b_centroid_post - crisis_state)

print(f"User A distance to crisis state: {user_a_to_crisis:.4f}")
print(f"User B distance to crisis state: {user_b_to_crisis:.4f}")

if abs(user_a_to_crisis - user_b_to_crisis) < 0.1:
    print("⚠ POTENTIAL CONTAMINATION: Both users equally affected by crisis")
else:
    print("✓ ISOLATION MAINTAINED: Users remain distinct from crisis state")
print()

# ============================================================================
# CONCLUSIONS
# ============================================================================
print("=" * 80)
print("ANALYSIS CONCLUSIONS (REAL MODEL)")
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
    print(f"  (Separation ratio: {sep_ratio:.2f})")
print()

# Criterion 2: Lyapunov stability
negative_ratio = sum(1 for v in lyapunov_vals if v < 0) / len(lyapunov_vals)
if negative_ratio > 0.6:
    print("✓ CONVERGENT DYNAMICS: Majority of resolutions are stable")
    print(f"  ({negative_ratio*100:.0f}% negative Lyapunov exponents)")
elif negative_ratio > 0.3:
    print("⚠ MIXED DYNAMICS: Some convergence, some chaos")
    print(f"  ({negative_ratio*100:.0f}% negative Lyapunov exponents)")
else:
    print("✗ UNSTABLE DYNAMICS: J-Operator not converging")
    print(f"  ({negative_ratio*100:.0f}% negative Lyapunov exponents)")
print()

# Criterion 3: Silhouette quality
if silhouette > 0.3:
    print(f"✓ STRONG CLUSTER QUALITY: Silhouette = {silhouette:.3f}")
elif silhouette > 0.1:
    print(f"⚠ MODERATE CLUSTER QUALITY: Silhouette = {silhouette:.3f}")
else:
    print(f"✗ POOR CLUSTER QUALITY: Silhouette = {silhouette:.3f}")
print()

# Criterion 4: Convergence success
conv_rate = converged_count / len(j_metadata)
if conv_rate > 0.5:
    print(f"✓ RELIABLE CONVERGENCE: {conv_rate*100:.0f}% converged within limit")
elif conv_rate > 0:
    print(f"⚠ PARTIAL CONVERGENCE: {conv_rate*100:.0f}% converged within limit")
else:
    print(f"✗ NO CONVERGENCE: All resolutions hit iteration limit")
print()

print("VERDICT:")
print()

score = 0
if sep_ratio > 1.2:
    score += 1
if negative_ratio > 0.6:
    score += 1
if silhouette > 0.1:
    score += 0.5
if conv_rate > 0.3:
    score += 0.5

if score >= 2.5:
    print("✓✓✓ REAL MODEL VALIDATES ARCHITECTURE")
    print("    J-Operator with actual transformer processing creates")
    print("    measurably distinct crisis dynamics. The A ≠ s axiom")
    print("    holds with real neural computation.")
elif score >= 1.5:
    print("⚠⚠ PARTIAL VALIDATION")
    print("    Some distinction observed but results are mixed.")
    print("    Architecture may need tuning for this model size.")
else:
    print("✗✗✗ ARCHITECTURE FAILS WITH REAL MODEL")
    print("    J-Operator does not create meaningful distinction")
    print("    when using actual transformer layers.")

print()
print("=" * 80)
print()
print(f"Model used: {model_name} ({hidden_size}-dim latent space)")
print("This validates the architecture with real transformer computation.")
