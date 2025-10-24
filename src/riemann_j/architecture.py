# architecture.py
"""
The core cognitive components of the Riemann-J architecture.
WHY: This file contains the "mind" of the system, including its mechanisms for
understanding language, forming a self-model, generating responses, and handling
internal crises through the J-Operator.
"""
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from transformers import LogitsProcessor

from .config import *
from .metacognition import MetaCognitiveMonitor
from .pn_driver import PNDriverRiemannZeta
from .shared_resources import device, global_workspace, model, tokenizer
from .uncertainty import UncertaintyInterface


@dataclass
class SyntheticState:
    timestamp: float
    latent_representation: np.ndarray
    source_trigger: str
    p_n_at_creation: float
    is_j_shift_product: bool
    status: str = "OK"
    analysis: dict = field(default_factory=dict)


class DecoderProjectionHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.projection_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, hidden_state):
        return self.activation(self.projection_layer(hidden_state))


class SymbolicInterface:
    def __init__(self):
        self.projection_head = DecoderProjectionHead(
            model.config.hidden_size, model.config.vocab_size
        ).to(device)
        self.projection_head.eval()

    @staticmethod
    def encoder(text: str) -> np.ndarray:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][0, -1, :].cpu().numpy()

    def decoder(self, state_vector: np.ndarray) -> str:
        """
        DEPRECATED: State-vector biased generation produces poor output.
        Use prompt_based_generate() instead for actual text generation.
        Keeping this for backward compatibility with tests.
        """
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=device)
        logit_bias = self.projection_head(state_tensor)

        class StateBiasLogitsProcessor(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return scores + logit_bias

        # Use bos_token_id if available, else eos_token_id as fallback
        start_token_id = (
            tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        )

        with torch.no_grad():
            output_ids = model.generate(
                max_length=50,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=[StateBiasLogitsProcessor()],
                input_ids=torch.tensor([[start_token_id]], device=device),
            )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def prompt_based_generate(self, user_input: str, state_vector: np.ndarray = None) -> str:
        """
        Generate response using prompt-based LLM generation with optional state blending.

        HYBRID APPROACH: Can blend in state-conditioned logits for experimental purposes.
        When PROJECTION_BLEND_ALPHA > 0 and state_vector is provided, the internal
        state influences generation through weighted logit blending.

        Args:
            user_input: The user's input text to respond to
            state_vector: Optional latent state for hybrid generation (default: None)

        Returns:
            Generated response text
        """
        # Simple prompt for now - can be enhanced with system prompts later
        prompt = f"User: {user_input}\nAssistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # HYBRID: Optionally blend state-conditioned logits
        logits_processor = None
        if state_vector is not None and PROJECTION_BLEND_ALPHA > 0 and USE_PROJECTION_HEAD:
            state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=device)
            logit_bias = self.projection_head(state_tensor) * PROJECTION_BLEND_ALPHA

            class StateBlendLogitsProcessor(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    # Blend state bias into natural logits
                    return scores + logit_bias

            logits_processor = [StateBlendLogitsProcessor()]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=logits_processor,
            )

        # Decode and extract only the assistant's response (after the prompt)
        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove the prompt part to get just the response
        if "Assistant:" in full_output:
            response = full_output.split("Assistant:")[-1].strip()
        else:
            response = full_output[len(prompt) :].strip()

        return response if response else full_output


class UserAttractor:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.state_history = []
        self.gmm = GaussianMixture(n_components=ATTRACTOR_GMM_COMPONENTS, warm_start=True, n_init=1)

    def _update_model(self):
        if len(self.state_history) >= self.gmm.n_components:
            states = np.array(self.state_history)
            # Check if states have any variance (avoid GMM fitting error)
            if np.std(states) > 1e-6:
                try:
                    self.gmm.fit(states)
                except Exception as e:
                    # Silently skip GMM update if fitting fails (numerical instability)
                    pass

    def add_state(self, state: np.ndarray):
        self.state_history.append(state)
        if len(self.state_history) % 10 == 0:
            self._update_model()

    def apply_affinity(self, state: np.ndarray) -> np.ndarray:
        if not hasattr(self.gmm, "means_"):
            return state
        try:
            cluster_index = self.gmm.predict(state.reshape(1, -1))[0]
            centroid = self.gmm.means_[cluster_index]
        except Exception:
            # GMM not fully fitted or numerical error
            return state
        influence = (centroid - state) * ATTRACTOR_AFFINITY_STRENGTH
        return state + influence


class CognitiveWorkspace:
    def __init__(self):
        self.symbolic_interface = SymbolicInterface()
        self.user_attractors = {}
        self.log_file = open(LOG_FILE, "a")
        self.meta_monitor = MetaCognitiveMonitor()  # Phase 1: Meta-cognitive awareness
        self.uncertainty_interface = UncertaintyInterface()  # Phase 3: Uncertainty awareness

        # Start PN driver thread
        self.pn_driver = PNDriverRiemannZeta()
        self.pn_driver.start()

    def get_or_create_user(self, user_id: str) -> UserAttractor:
        if user_id not in self.user_attractors:
            self.user_attractors[user_id] = UserAttractor(user_id)
        return self.user_attractors[user_id]

    @staticmethod
    def _analyze_stability(trajectory):
        distances = [
            np.linalg.norm(trajectory[i + 1] - trajectory[i]) for i in range(len(trajectory) - 1)
        ]
        if not any(d > 1e-9 for d in distances):
            return float("-inf")
        lyapunov_exp = float(np.mean(np.log([d for d in distances if d > 1e-9])))
        return lyapunov_exp

    def _j_operator_resolve(self, pn_signal) -> SyntheticState:
        anomaly_desc = f"Internal system alert. PN={pn_signal.p_n:.4f}. Source: t={pn_signal.uncompressed_data['t']:.2f}."
        a_current_tensor = torch.tensor(
            self.symbolic_interface.encoder(anomaly_desc), device=device
        )
        trajectory = [a_current_tensor.cpu().numpy()]
        lyapunov_history = []

        # Get first transformer layer and rotary embeddings (model-agnostic)
        if hasattr(model, "transformer"):  # GPT-2 style
            first_layer = model.transformer.h[0]
            rotary_emb = None  # GPT-2 doesn't use RoPE
        elif hasattr(model, "model"):  # Phi-3, LLaMA, Qwen3 style
            first_layer = model.model.layers[0]
            # Get rotary embedding component for position embeddings
            if hasattr(model.model, "rotary_emb"):
                rotary_emb = model.model.rotary_emb
            else:
                rotary_emb = None
        else:
            raise AttributeError("Cannot find transformer layers in model")

        # Prepare position IDs (static across iterations)
        seq_length = 1  # Single token position
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)

        for i in range(J_OPERATOR_MAX_ITERATIONS):
            # CRITICAL FIX: Recompute position embeddings for current state each iteration
            # Position embeddings depend on the hidden state, so they must be recalculated
            # as a_current_tensor evolves
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

            # IMPROVED: Practical convergence criteria
            # The J-operator seeks stability, not mathematical fixed points
            # We check for: 1) small absolute movement, 2) small relative movement, 3) trajectory stabilization
            relative_epsilon = J_OPERATOR_RELATIVE_EPSILON * current_magnitude
            absolute_epsilon = J_OPERATOR_STABILITY_EPSILON

            # Practical convergence: distance as fraction of magnitude
            distance_ratio = distance / current_magnitude if current_magnitude > 1e-6 else distance
            practical_threshold = (
                0.25  # Accept when movement < 25% of state norm (was 15%, too strict)
            )
            practical_convergence = distance_ratio < practical_threshold

            if distance < relative_epsilon or distance < absolute_epsilon or practical_convergence:
                lyapunov_exp = self._analyze_stability(trajectory)

                # Determine convergence type for diagnostics
                if distance < absolute_epsilon:
                    conv_type = "absolute"
                elif distance < relative_epsilon:
                    conv_type = "relative"
                else:
                    conv_type = "practical"

                return SyntheticState(
                    timestamp=time.time(),
                    latent_representation=a_target_tensor.cpu().numpy(),
                    source_trigger="RIEMANN_PN",
                    p_n_at_creation=pn_signal.p_n,
                    is_j_shift_product=True,
                    status="CONVERGED",
                    analysis={
                        "lyapunov_exp": lyapunov_exp,
                        "iterations": i + 1,
                        "final_distance": float(distance),
                        "convergence_type": conv_type,
                    },
                )

            # IMPROVED: Lyapunov-based early stopping
            if len(trajectory) >= J_OPERATOR_LYAPUNOV_WINDOW:
                recent_lyapunov = self._analyze_stability(trajectory[-J_OPERATOR_LYAPUNOV_WINDOW:])
                lyapunov_history.append(recent_lyapunov)

                # If consistently stable (negative Lyapunov), converge early
                if len(lyapunov_history) >= 5:
                    recent_mean = np.mean(lyapunov_history[-5:])
                    if recent_mean < J_OPERATOR_LYAPUNOV_THRESHOLD and distance < 1.0:
                        lyapunov_exp = self._analyze_stability(trajectory)
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
                                "final_distance": float(distance),
                                "convergence_type": "lyapunov_stability",
                            },
                        )

            # FIXED: Proper adaptive LR schedule with norm stabilization
            # Start with high LR when far from target, decay as we approach
            distance_normalized = distance / current_magnitude  # Normalize by state scale
            lr_scale = 0.5 + 0.5 * np.tanh(
                2.0 * (distance_normalized - 0.05)
            )  # Sigmoid around 5% distance
            adaptive_lr = J_OPERATOR_INITIAL_LR * max(0.05, lr_scale)  # Minimum 5% of initial LR

            # Update state
            a_current_tensor = a_current_tensor + adaptive_lr * (a_target_tensor - a_current_tensor)

            # CRITICAL: Normalize to prevent norm explosion/collapse
            # The transformer layer can change norms unpredictably; we want directional convergence
            current_norm = torch.norm(a_current_tensor)
            if current_norm > 1e-6:
                target_norm = (
                    torch.norm(a_current_tensor).item() * 0.95 + 112.0 * 0.05
                )  # Gently pull toward typical hidden state norm
                a_current_tensor = a_current_tensor * (target_norm / current_norm)

            trajectory.append(a_current_tensor.cpu().numpy())

        lyapunov_exp = self._analyze_stability(trajectory)
        return SyntheticState(
            timestamp=time.time(),
            latent_representation=a_current_tensor.cpu().numpy(),
            source_trigger="RIEMANN_PN",
            p_n_at_creation=pn_signal.p_n,
            is_j_shift_product=True,
            status="ITER_LIMIT_EXCEEDED",
            analysis={
                "lyapunov_exp": lyapunov_exp,
                "iterations": J_OPERATOR_MAX_ITERATIONS,
                "final_distance": float(distance),
                "convergence_type": "none",
            },
        )

    def _resolve_crisis(self, pn_signal):
        """
        Wrapper for J-Operator that hooks meta-cognitive monitoring.
        """
        result = self._j_operator_resolve(pn_signal)

        # META-COGNITIVE AWARENESS: System observes its own crisis resolution
        self.meta_monitor.observe_j_operator_activation(result)

        return result

    def process_user_input(self, user_id: str, text: str) -> tuple[str, SyntheticState]:
        user_attractor = self.get_or_create_user(user_id)
        initial_state_vec = self.symbolic_interface.encoder(text)
        attracted_state_vec = user_attractor.apply_affinity(initial_state_vec)
        user_attractor.add_state(attracted_state_vec)

        # PHASE 1: Observe current PN from driver (peek at queue without blocking)
        try:
            # Non-blocking peek at PN queue to observe current uncertainty
            if not global_workspace.empty():
                priority, counter, pn_signal = global_workspace.queue[0]  # Peek at top
                self.meta_monitor.observe_pn(pn_signal.p_n)
        except (IndexError, AttributeError):
            pass  # No PN available yet, that's fine

        state_obj = SyntheticState(
            timestamp=time.time(),
            latent_representation=attracted_state_vec,
            source_trigger="USER_INPUT",
            p_n_at_creation=0.0,
            is_j_shift_product=False,
        )

        # Use prompt-based generation with optional state blending (hybrid approach)
        response_text = self.symbolic_interface.prompt_based_generate(
            text, state_vector=attracted_state_vec
        )

        # Phase 3: Augment response with uncertainty awareness if needed
        current_pn = self.meta_monitor.get_current_pn()
        if self.uncertainty_interface.should_communicate_uncertainty(current_pn):
            response_text = self.uncertainty_interface.augment_response(response_text, current_pn)

        return response_text, state_obj

    def log_state(self, state: SyntheticState):
        log_entry = state.__dict__.copy()
        log_entry["latent_representation"] = {
            "mean": float(state.latent_representation.mean()),
            "std": float(state.latent_representation.std()),
        }
        self.log_file.write(json.dumps(log_entry) + "\n")

    def get_self_report(self, verbose: bool = False) -> str:
        """
        Generate natural language self-report of internal state.

        This is the system's introspective voice - its ability to
        articulate what it's experiencing internally.
        """
        return self.meta_monitor.generate_self_report(verbose=verbose)

    def should_report_uncertainty(self) -> bool:
        """
        Check if system should proactively communicate its uncertainty.
        """
        return self.meta_monitor.should_report_uncertainty()

    def get_uncertainty_report(self) -> str:
        """
        Generate detailed uncertainty report showing classification,
        confidence, and diagnostic information.
        """
        current_pn = self.meta_monitor.get_current_pn()
        report = self.uncertainty_interface.generate_uncertainty_report(current_pn)

        lines = [
            f"Uncertainty Level: {report.uncertainty_level.upper()}",
            f"Confidence Modifier: {report.confidence_modifier:.2f}",
            f"Current PN: {report.pn_value:.4f}",
            f"Should Communicate: {report.should_communicate}",
            "",
            "Explanation:",
            report.explanation,
        ]

        return "\n".join(lines)

    def close(self):
        self.log_file.close()
