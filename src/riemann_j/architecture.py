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
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=device)
        logit_bias = self.projection_head(state_tensor)

        class StateBiasLogitsProcessor(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return scores + logit_bias

        with torch.no_grad():
            output_ids = model.generate(
                max_length=50,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=[StateBiasLogitsProcessor()],
                input_ids=torch.tensor([[tokenizer.bos_token_id]], device=device),
            )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)


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
            return -np.inf
        lyapunov_exp = np.mean(np.log([d for d in distances if d > 1e-9]))
        return lyapunov_exp

    def _j_operator_resolve(self, pn_signal) -> SyntheticState:
        anomaly_desc = f"Internal system alert. PN={pn_signal.p_n:.4f}. Source: t={pn_signal.uncompressed_data['t']:.2f}."
        a_current_tensor = torch.tensor(
            self.symbolic_interface.encoder(anomaly_desc), device=device
        )
        trajectory = [a_current_tensor.cpu().numpy()]
        lyapunov_history = []

        # Get first transformer layer (model-agnostic)
        if hasattr(model, "transformer"):  # GPT-2 style
            first_layer = model.transformer.h[0]
        elif hasattr(model, "model"):  # Phi-3, LLaMA style
            first_layer = model.model.layers[0]
        else:
            raise AttributeError("Cannot find transformer layers in model")

        for i in range(J_OPERATOR_MAX_ITERATIONS):
            with torch.no_grad():
                a_target_tensor = first_layer(a_current_tensor.unsqueeze(0).unsqueeze(0))[
                    0
                ].squeeze(0)

            distance = torch.norm(a_target_tensor - a_current_tensor).item()
            current_magnitude = torch.norm(a_current_tensor).item()

            # IMPROVED: Relative convergence check
            relative_epsilon = J_OPERATOR_RELATIVE_EPSILON * current_magnitude
            absolute_epsilon = J_OPERATOR_STABILITY_EPSILON

            if distance < relative_epsilon or distance < absolute_epsilon:
                lyapunov_exp = self._analyze_stability(trajectory)
                return SyntheticState(
                    timestamp=time.time(),
                    latent_representation=a_target_tensor.cpu().numpy(),
                    source_trigger="RIEMANN_PN",
                    p_n_at_creation=pn_signal.p_n,
                    is_j_shift_product=True,
                    status="CONVERGED",
                    analysis={
                        "lyapunov_exp": lyapunov_exp,
                        "iterations": i,
                        "final_distance": distance,
                        "convergence_type": (
                            "relative" if distance < relative_epsilon else "absolute"
                        ),
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
                                "final_distance": distance,
                                "convergence_type": "lyapunov_stability",
                            },
                        )

            adaptive_lr = J_OPERATOR_INITIAL_LR / (1.0 + J_OPERATOR_ADAPTIVE_LR_RATE * distance)
            a_current_tensor = a_current_tensor + adaptive_lr * (a_target_tensor - a_current_tensor)
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
                "final_distance": distance,
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
                priority, pn_signal = global_workspace.queue[0]  # Peek at top
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
        response_text = self.symbolic_interface.decoder(attracted_state_vec)

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
