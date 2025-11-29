#!/usr/bin/env python3
"""
CPU-based inference for DeepSeek-Math-V2 with memory-mapped expert loading.
This implementation trades speed for memory efficiency.

WARNING: This will be VERY slow (expect ~1-5 tokens per minute on CPU).
"""

import os
import json
import math
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

MODEL_PATH = "/mnt/d/models/DeepSeek-Math-V2"
BLOCK_SIZE = 128


@dataclass
class ModelArgs:
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3  # First 3 layers are dense
    n_heads: int = 128
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    score_func: str = "sigmoid"
    route_scale: float = 2.5
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    original_seq_len: int = 4096
    max_seq_len: int = 4096
    max_batch_size: int = 1
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


def dequantize_fp8(weight_fp8: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weight to BF16 using block-wise scale."""
    if weight_fp8.dtype != torch.float8_e4m3fn:
        return weight_fp8.to(torch.bfloat16)

    shape = weight_fp8.shape
    out_features, in_features = shape

    # Apply block-wise scaling
    out_blocks = (out_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    in_blocks = (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Pad if necessary
    pad_out = out_blocks * BLOCK_SIZE - out_features
    pad_in = in_blocks * BLOCK_SIZE - in_features

    weight = weight_fp8.to(torch.float32)
    if pad_out > 0 or pad_in > 0:
        weight = F.pad(weight, (0, pad_in, 0, pad_out))

    # Reshape to blocks and apply scale
    weight = weight.view(out_blocks, BLOCK_SIZE, in_blocks, BLOCK_SIZE)
    scale = scale_inv.view(out_blocks, 1, in_blocks, 1)
    weight = weight * scale

    # Reshape back and remove padding
    weight = weight.view(out_blocks * BLOCK_SIZE, in_blocks * BLOCK_SIZE)
    weight = weight[:out_features, :in_features]

    return weight.to(torch.bfloat16)


class TensorLoader:
    """Memory-mapped tensor loading with caching."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.index = self._load_index()
        self.file_handles: Dict[str, safe_open] = {}
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_size_limit = 100 * 1024 * 1024 * 1024  # 100GB cache limit
        self.current_cache_size = 0

    def _load_index(self):
        with open(os.path.join(self.model_path, "model.safetensors.index.json")) as f:
            return json.load(f)

    def _get_handle(self, filename: str):
        if filename not in self.file_handles:
            filepath = os.path.join(self.model_path, filename)
            self.file_handles[filename] = safe_open(filepath, framework="pt", device="cpu")
        return self.file_handles[filename]

    def load(self, name: str, cache: bool = True) -> torch.Tensor:
        """Load a tensor, optionally caching it."""
        if name in self.cache:
            return self.cache[name]

        if name not in self.index["weight_map"]:
            raise KeyError(f"Tensor not found: {name}")

        filename = self.index["weight_map"][name]
        handle = self._get_handle(filename)
        tensor = handle.get_tensor(name)

        # Dequantize FP8 tensors
        if tensor.dtype == torch.float8_e4m3fn:
            scale_name = name.replace(".weight", ".weight_scale_inv")
            if scale_name in self.index["weight_map"]:
                scale = self.load(scale_name, cache=False)
                tensor = dequantize_fp8(tensor, scale)

        if cache:
            tensor_size = tensor.numel() * tensor.element_size()
            if self.current_cache_size + tensor_size < self.cache_size_limit:
                self.cache[name] = tensor
                self.current_cache_size += tensor_size

        return tensor

    def load_expert(self, layer: int, expert_id: int) -> Dict[str, torch.Tensor]:
        """Load all tensors for a specific expert."""
        prefix = f"model.layers.{layer}.mlp.experts.{expert_id}"
        tensors = {}
        for suffix in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
            name = f"{prefix}.{suffix}"
            if name in self.index["weight_map"]:
                tensors[suffix.split(".")[0]] = self.load(name, cache=False)
        return tensors

    def close(self):
        self.file_handles.clear()
        self.cache.clear()


class RMSNorm:
    """RMSNorm without nn.Module overhead."""
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        self.weight = weight.float()
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """Precompute RoPE frequencies."""
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings."""
    dtype = x.dtype
    shape = x.shape
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Linear layer forward pass."""
    return F.linear(x, weight, bias)


class DeepSeekV32CPU:
    """CPU-based DeepSeek-V3.2 model with mmap expert loading."""

    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args
        self.loader = TensorLoader(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print("Loading non-expert weights into memory...")
        self._load_base_weights()
        print(f"Base weights loaded. Cache size: {self.loader.current_cache_size / 1e9:.2f} GB")

        self.freqs_cis = precompute_freqs_cis(args)

    def _load_base_weights(self):
        """Pre-load all non-expert weights into cache."""
        for name in tqdm(self.loader.index["weight_map"].keys(), desc="Loading weights"):
            # Skip expert weights (load on demand)
            if "experts" in name and "shared_experts" not in name:
                continue
            # Skip scale tensors (loaded with weights)
            if "scale_inv" in name:
                continue
            try:
                self.loader.load(name, cache=True)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token embedding."""
        embed_weight = self.loader.load("model.embed_tokens.weight")
        return F.embedding(tokens, embed_weight)

    def rms_norm(self, x: torch.Tensor, layer: int, name: str) -> torch.Tensor:
        """Apply RMSNorm."""
        weight = self.loader.load(f"model.layers.{layer}.{name}.weight")
        norm = RMSNorm(weight, self.args.rms_norm_eps)
        return norm(x)

    def attention(self, x: torch.Tensor, layer: int, start_pos: int) -> torch.Tensor:
        """Simplified attention (no KV cache, no sparse attention for simplicity)."""
        bsz, seqlen, _ = x.shape

        # Load attention weights
        wq_a = self.loader.load(f"model.layers.{layer}.self_attn.q_a_proj.weight")
        wq_b = self.loader.load(f"model.layers.{layer}.self_attn.q_b_proj.weight")
        wkv_a = self.loader.load(f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight")
        wkv_b = self.loader.load(f"model.layers.{layer}.self_attn.kv_b_proj.weight")
        wo = self.loader.load(f"model.layers.{layer}.self_attn.o_proj.weight")

        # Q projection with LoRA
        q = linear(x, wq_a)
        q = self.rms_norm_inline(q, layer, "self_attn.q_a_layernorm")
        q = linear(q, wq_b)
        q = q.view(bsz, seqlen, self.args.n_heads, self.args.qk_nope_head_dim + self.args.qk_rope_head_dim)

        # KV projection with LoRA
        kv = linear(x, wkv_a)
        kv_rank = kv[:, :, :self.args.kv_lora_rank]
        k_pe = kv[:, :, self.args.kv_lora_rank:]
        kv_rank = self.rms_norm_inline(kv_rank, layer, "self_attn.kv_a_layernorm")
        kv = linear(kv_rank, wkv_b)
        kv = kv.view(bsz, seqlen, self.args.n_heads, self.args.qk_nope_head_dim + self.args.v_head_dim)

        # Split Q and KV
        q_nope, q_pe = q.split([self.args.qk_nope_head_dim, self.args.qk_rope_head_dim], dim=-1)
        k_nope, v = kv.split([self.args.qk_nope_head_dim, self.args.v_head_dim], dim=-1)

        # Apply RoPE
        freqs = self.freqs_cis[start_pos:start_pos + seqlen]
        q_pe = apply_rotary_emb(q_pe, freqs)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2).expand(-1, -1, self.args.n_heads, -1), freqs)

        # Combine
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # Attention scores
        scale = (self.args.qk_nope_head_dim + self.args.qk_rope_head_dim) ** -0.5
        scores = torch.einsum("bshd,bthd->bsht", q, k) * scale

        # Causal mask
        mask = torch.triu(torch.full((seqlen, seqlen), float("-inf")), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(2)

        # Softmax and value
        attn = scores.softmax(dim=-1).to(v.dtype)
        out = torch.einsum("bsht,bthd->bshd", attn, v)

        # Output projection
        out = out.flatten(2)
        out = linear(out, wo)

        return out

    def rms_norm_inline(self, x: torch.Tensor, layer: int, name: str) -> torch.Tensor:
        """RMSNorm for inline usage."""
        weight_name = f"model.layers.{layer}.{name}.weight"
        if weight_name in self.loader.index["weight_map"]:
            weight = self.loader.load(weight_name)
        else:
            # Fallback for norms that might have different naming
            return x
        norm = RMSNorm(weight, self.args.rms_norm_eps)
        return norm(x)

    def mlp_dense(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Dense MLP for first few layers."""
        w1 = self.loader.load(f"model.layers.{layer}.mlp.gate_proj.weight")
        w2 = self.loader.load(f"model.layers.{layer}.mlp.down_proj.weight")
        w3 = self.loader.load(f"model.layers.{layer}.mlp.up_proj.weight")

        gate = linear(x, w1)
        up = linear(x, w3)
        hidden = F.silu(gate.float()) * up.float()
        return linear(hidden.to(x.dtype), w2)

    def mlp_moe(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """MoE MLP with on-demand expert loading."""
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)  # [bsz * seqlen, dim]

        # Load gate weights
        gate_weight = self.loader.load(f"model.layers.{layer}.mlp.gate.weight")

        # Route tokens to experts
        scores = linear(x_flat.float(), gate_weight.float())
        scores = scores.sigmoid()

        # Load bias if exists
        bias_name = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
        if bias_name in self.loader.index["weight_map"]:
            bias = self.loader.load(bias_name)
            scores_biased = scores + bias
        else:
            scores_biased = scores

        # Top-k expert selection
        topk = self.args.n_activated_experts
        topk_scores, topk_indices = scores_biased.topk(topk, dim=-1)

        # Get original scores for selected experts
        weights = scores.gather(1, topk_indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.args.route_scale

        # Shared experts
        shared_out = torch.zeros_like(x_flat)
        for se_idx in range(self.args.n_shared_experts):
            prefix = f"model.layers.{layer}.mlp.shared_experts"
            if self.args.n_shared_experts == 1:
                w1 = self.loader.load(f"{prefix}.gate_proj.weight")
                w2 = self.loader.load(f"{prefix}.down_proj.weight")
                w3 = self.loader.load(f"{prefix}.up_proj.weight")
            else:
                w1 = self.loader.load(f"{prefix}.{se_idx}.gate_proj.weight")
                w2 = self.loader.load(f"{prefix}.{se_idx}.down_proj.weight")
                w3 = self.loader.load(f"{prefix}.{se_idx}.up_proj.weight")

            gate = linear(x_flat, w1)
            up = linear(x_flat, w3)
            hidden = F.silu(gate.float()) * up.float()
            shared_out = shared_out + linear(hidden.to(x_flat.dtype), w2)

        # Routed experts (load on demand)
        expert_out = torch.zeros_like(x_flat)
        unique_experts = topk_indices.unique().tolist()

        for expert_id in unique_experts:
            # Find which tokens go to this expert
            mask = (topk_indices == expert_id).any(dim=-1)
            if not mask.any():
                continue

            token_indices = mask.nonzero(as_tuple=True)[0]
            expert_tokens = x_flat[token_indices]

            # Get weights for this expert
            expert_positions = (topk_indices[token_indices] == expert_id)
            token_weights = (weights[token_indices] * expert_positions.float()).sum(dim=-1, keepdim=True)

            # Load expert weights on demand
            expert_tensors = self.loader.load_expert(layer, expert_id)
            if not expert_tensors:
                continue

            # Expert forward
            gate = linear(expert_tokens, expert_tensors["gate_proj"])
            up = linear(expert_tokens, expert_tensors["up_proj"])
            hidden = F.silu(gate.float()) * up.float()
            out = linear(hidden.to(expert_tokens.dtype), expert_tensors["down_proj"])

            # Weighted contribution
            expert_out[token_indices] = expert_out[token_indices] + (out * token_weights).to(expert_out.dtype)

        return (shared_out + expert_out).view(bsz, seqlen, dim)

    def transformer_block(self, x: torch.Tensor, layer: int, start_pos: int) -> torch.Tensor:
        """Single transformer block."""
        # Pre-attention norm
        h = self.rms_norm(x, layer, "input_layernorm")

        # Attention
        attn_out = self.attention(h, layer, start_pos)
        x = x + attn_out

        # Pre-MLP norm
        h = self.rms_norm(x, layer, "post_attention_layernorm")

        # MLP (dense or MoE)
        if layer < self.args.n_dense_layers:
            mlp_out = self.mlp_dense(h, layer)
        else:
            mlp_out = self.mlp_moe(h, layer)

        x = x + mlp_out
        return x

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        bsz, seqlen = tokens.shape

        # Embedding
        x = self.embed(tokens)

        # Transformer blocks
        for layer in tqdm(range(self.args.n_layers), desc="Layers"):
            x = self.transformer_block(x, layer, 0)

        # Final norm
        norm_weight = self.loader.load("model.norm.weight")
        norm = RMSNorm(norm_weight, self.args.rms_norm_eps)
        x = norm(x)

        # LM head
        lm_head = self.loader.load("lm_head.weight")
        logits = linear(x, lm_head)

        return logits

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        tokens = self.tokenizer.encode(prompt, return_tensors="pt")

        print(f"Prompt tokens: {tokens.shape[1]}")
        print("Generating (this will be slow)...")

        for i in range(max_new_tokens):
            logits = self.forward(tokens)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

            # Decode and print
            decoded = self.tokenizer.decode(next_token.item())
            print(decoded, end="", flush=True)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        print()
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)


def main():
    print("=" * 60)
    print("DeepSeek-Math-V2 CPU Inference")
    print("WARNING: This will be VERY slow!")
    print("=" * 60)

    args = ModelArgs()
    model = DeepSeekV32CPU(MODEL_PATH, args)

    # Test generation
    prompt = "What is 2 + 2?"
    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    result = model.generate(prompt, max_new_tokens=20, temperature=0.7)
    print("-" * 40)
    print(f"Full response: {result}")


if __name__ == "__main__":
    main()
