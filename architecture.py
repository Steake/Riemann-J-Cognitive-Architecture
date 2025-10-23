# architecture.py
"""
The core cognitive components of the Riemann-J architecture.
WHY: This file contains the "mind" of the system, including its mechanisms for
understanding language, forming a self-model, generating responses, and handling
internal crises through the J-Operator.
"""
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from transformers import LogitsProcessor
from sklearn.mixture import GaussianMixture
import json
import time

from shared_resources import tokenizer, model, device
from config import *

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
        self.projection_head = DecoderProjectionHead(model.config.hidden_size, model.config.vocab_size).to(device)
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
                input_ids=torch.tensor([[tokenizer.bos_token_id]], device=device)
            )
        
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

class UserAttractor:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.state_history = []
        self.gmm = GaussianMixture(n_components=ATTRACTOR_GMM_COMPONENTS, warm_start=True, n_init=1)

    def _update_model(self):
        if len(self.state_history) >= self.gmm.n_components:
            self.gmm.fit(np.array(self.state_history))

    def add_state(self, state: np.ndarray):
        self.state_history.append(state)
        if len(self.state_history) % 10 == 0:
            self._update_model()

    def apply_affinity(self, state: np.ndarray) -> np.ndarray:
        if not hasattr(self.gmm, 'means_'): return state
        cluster_index = self.gmm.predict(state.reshape(1, -1))[0]
        centroid = self.gmm.means_[cluster_index]
        influence = (centroid - state) * ATTRACTOR_AFFINITY_STRENGTH
        return state + influence

class CognitiveWorkspace:
    def __init__(self):
        self.symbolic_interface = SymbolicInterface()
        self.user_attractors = {}
        self.log_file = open(LOG_FILE, 'a')

    def get_or_create_user(self, user_id: str) -> UserAttractor:
        if user_id not in self.user_attractors:
            self.user_attractors[user_id] = UserAttractor(user_id)
        return self.user_attractors[user_id]

    @staticmethod
    def _analyze_stability(trajectory):
        distances = [np.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)]
        if not any(d > 1e-9 for d in distances): return -np.inf
        lyapunov_exp = np.mean(np.log([d for d in distances if d > 1e-9]))
        return lyapunov_exp

    def _j_operator_resolve(self, pn_signal) -> SyntheticState:
        anomaly_desc = f"Internal system alert. PN={pn_signal.p_n:.4f}. Source: t={pn_signal.uncompressed_data['t']:.2f}."
        a_current_tensor = torch.tensor(self.symbolic_interface.encoder(anomaly_desc), device=device)
        trajectory = [a_current_tensor.cpu().numpy()]

        for i in range(J_OPERATOR_MAX_ITERATIONS):
            with torch.no_grad():
                a_target_tensor = model.transformer.h[0](a_current_tensor.unsqueeze(0).unsqueeze(0))[0].squeeze(0)
            
            distance = torch.norm(a_target_tensor - a_current_tensor).item()

            if distance < J_OPERATOR_STABILITY_EPSILON:
                lyapunov_exp = self._analyze_stability(trajectory)
                return SyntheticState(timestamp=time.time(), latent_representation=a_target_tensor.cpu().numpy(), source_trigger="RIEMANN_PN", p_n_at_creation=pn_signal.p_n, is_j_shift_product=True, status="CONVERGED", analysis={'lyapunov_exp': lyapunov_exp, 'iterations': i})
            
            adaptive_lr = J_OPERATOR_INITIAL_LR / (1.0 + J_OPERATOR_ADAPTIVE_LR_RATE * distance)
            a_current_tensor = a_current_tensor + adaptive_lr * (a_target_tensor - a_current_tensor)
            trajectory.append(a_current_tensor.cpu().numpy())

        lyapunov_exp = self._analyze_stability(trajectory)
        return SyntheticState(timestamp=time.time(), latent_representation=a_current_tensor.cpu().numpy(), source_trigger="RIEMANN_PN", p_n_at_creation=pn_signal.p_n, is_j_shift_product=True, status="ITER_LIMIT_EXCEEDED", analysis={'lyapunov_exp': lyapunov_exp, 'iterations': J_OPERATOR_MAX_ITERATIONS})

    def process_user_input(self, user_id: str, text: str) -> tuple[str, SyntheticState]:
        user_attractor = self.get_or_create_user(user_id)
        initial_state_vec = self.symbolic_interface.encoder(text)
        attracted_state_vec = user_attractor.apply_affinity(initial_state_vec)
        user_attractor.add_state(attracted_state_vec)

        state_obj = SyntheticState(timestamp=time.time(), latent_representation=attracted_state_vec, source_trigger="USER_INPUT", p_n_at_creation=0.0, is_j_shift_product=False)
        response_text = self.symbolic_interface.decoder(attracted_state_vec)
        return response_text, state_obj

    def log_state(self, state: SyntheticState):
        log_entry = state.__dict__.copy()
        log_entry['latent_representation'] = {'mean': float(state.latent_representation.mean()), 'std': float(state.latent_representation.std())}
        self.log_file.write(json.dumps(log_entry) + '\n')

    def close(self):
        self.log_file.close()
