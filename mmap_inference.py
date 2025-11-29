#!/usr/bin/env python3
"""
Memory-mapped inference for DeepSeek-Math-V2
Loads non-expert layers to RAM, experts loaded on-demand from disk via mmap.
"""

import os
import json
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F

MODEL_PATH = "/mnt/d/models/DeepSeek-Math-V2"

@dataclass
class ModelConfig:
    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    num_experts: int = 256
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rms_norm_eps: float = 1e-6


class MMapExpertLoader:
    """Manages memory-mapped loading of expert weights."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.index = self._load_index()
        self.file_handles: Dict[str, safe_open] = {}
        self._build_expert_map()

    def _load_index(self):
        with open(os.path.join(self.model_path, "model.safetensors.index.json")) as f:
            return json.load(f)

    def _build_expert_map(self):
        """Build mapping of expert tensors to files."""
        self.expert_map = {}  # (layer, expert_id, tensor_name) -> (file, tensor_name)
        self.non_expert_tensors = {}  # tensor_name -> (file, tensor_name)

        for name, file in self.index["weight_map"].items():
            if "experts" in name and "shared_experts" not in name:
                # Parse: model.layers.X.mlp.experts.Y.tensor_name
                parts = name.split(".")
                layer_idx = int(parts[2])
                expert_idx = int(parts[5])
                tensor_type = ".".join(parts[6:])
                self.expert_map[(layer_idx, expert_idx, tensor_type)] = (file, name)
            else:
                self.non_expert_tensors[name] = (file, name)

    def _get_file_handle(self, filename: str):
        if filename not in self.file_handles:
            filepath = os.path.join(self.model_path, filename)
            self.file_handles[filename] = safe_open(filepath, framework="pt", device="cpu")
        return self.file_handles[filename]

    def load_expert_tensor(self, layer: int, expert_id: int, tensor_type: str) -> torch.Tensor:
        """Load a specific expert tensor on demand."""
        key = (layer, expert_id, tensor_type)
        if key not in self.expert_map:
            raise KeyError(f"Expert tensor not found: {key}")

        file, name = self.expert_map[key]
        handle = self._get_file_handle(file)
        return handle.get_tensor(name)

    def load_non_expert_tensor(self, name: str) -> torch.Tensor:
        """Load a non-expert tensor."""
        if name not in self.non_expert_tensors:
            raise KeyError(f"Tensor not found: {name}")

        file, full_name = self.non_expert_tensors[name]
        handle = self._get_file_handle(file)
        return handle.get_tensor(full_name)

    def get_activated_experts(self, layer: int, expert_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Load all tensors for activated experts in a layer."""
        experts = {}
        tensor_types = ["gate_proj.weight", "gate_proj.weight_scale_inv",
                       "up_proj.weight", "up_proj.weight_scale_inv",
                       "down_proj.weight", "down_proj.weight_scale_inv"]

        for expert_id in expert_ids:
            for tensor_type in tensor_types:
                key = (layer, expert_id, tensor_type)
                if key in self.expert_map:
                    experts[f"{expert_id}.{tensor_type}"] = self.load_expert_tensor(layer, expert_id, tensor_type)

        return experts

    def close(self):
        self.file_handles.clear()


def test_mmap_loading():
    """Test that mmap loading works."""
    print("Initializing MMap Expert Loader...")
    loader = MMapExpertLoader(MODEL_PATH)

    print(f"\nFound {len(loader.expert_map)} expert tensor entries")
    print(f"Found {len(loader.non_expert_tensors)} non-expert tensor entries")

    # Test loading a non-expert tensor
    print("\nTesting non-expert tensor loading...")
    for name in list(loader.non_expert_tensors.keys())[:3]:
        tensor = loader.load_non_expert_tensor(name)
        print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")

    # Test loading expert tensors
    print("\nTesting expert tensor loading (simulating 8 activated experts)...")
    layer = 35  # Test layer
    activated = [0, 15, 42, 99, 128, 200, 230, 255]  # Random 8 experts

    import time
    start = time.time()
    experts = loader.get_activated_experts(layer, activated)
    elapsed = time.time() - start

    print(f"  Loaded {len(experts)} tensors for {len(activated)} experts in {elapsed:.3f}s")

    # Calculate memory
    total_bytes = sum(t.numel() * t.element_size() for t in experts.values())
    print(f"  Memory for activated experts: {total_bytes / 1e6:.1f} MB")

    loader.close()
    print("\nMMap loading test successful!")


if __name__ == "__main__":
    test_mmap_loading()
