# shared_resources.py
"""
Initializes and holds all global, thread-safe singleton objects.
WHY: This prevents re-initialization of heavy objects like the LLM and ensures
that different threads are accessing the same instances of shared state.
"""
import queue
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config

# The Global Workspace for inter-thread communication, sorted by priority.
global_workspace = queue.PriorityQueue()

print("Loading Causal Language Model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(config.TRANSFORMER_MODEL_NAME)
model.eval() # IMPORTANT: Disable dropout and other training-specific layers.

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on device: {device}")
