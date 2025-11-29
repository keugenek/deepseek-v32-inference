#!/usr/bin/env python3
"""
CPU-based inference for DeepSeek-Math-V2 with GPU COMPUTE offload.
Weights loaded from disk/CPU, but matrix multiplications run on GPU.
This should be significantly faster than pure CPU compute.
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

# GPU for compute and KV cache
GPU_DEVICE = torch.device("cuda:0")

@dataclass
class ModelArgs:
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
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
    max_seq_len: int = 2048
    max_batch_size: int = 1
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    def get_softmax_scale(self) -> float:
        """Compute attention softmax scale with YaRN mscale adjustment."""
        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        base_scale = head_dim ** -0.5
        # YaRN mscale adjustment when using extended context
        if self.max_seq_len > self.original_seq_len:
            yarn_mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
            return base_scale * yarn_mscale * yarn_mscale
        return base_scale


def dequantize_fp8(weight_fp8: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weight to BF16 using block-wise scale."""
    if weight_fp8.dtype != torch.float8_e4m3fn:
        return weight_fp8.to(torch.bfloat16)

    shape = weight_fp8.shape
    out_features, in_features = shape

    out_blocks = (out_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    in_blocks = (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE

    pad_out = out_blocks * BLOCK_SIZE - out_features
    pad_in = in_blocks * BLOCK_SIZE - in_features

    weight = weight_fp8.to(torch.float32)
    if pad_out > 0 or pad_in > 0:
        weight = F.pad(weight, (0, pad_in, 0, pad_out))

    weight = weight.view(out_blocks, BLOCK_SIZE, in_blocks, BLOCK_SIZE)
    scale = scale_inv.view(out_blocks, 1, in_blocks, 1)
    weight = weight * scale

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
        self.cache_size_limit = 100 * 1024 * 1024 * 1024
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
        if name in self.cache:
            return self.cache[name]

        if name not in self.index["weight_map"]:
            raise KeyError(f"Tensor not found: {name}")

        filename = self.index["weight_map"][name]
        handle = self._get_handle(filename)
        tensor = handle.get_tensor(name)

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


def rms_norm_gpu(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization on GPU."""
    dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight.float() * x).to(dtype)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to(GPU_DEVICE)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    shape = x.shape
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class KVCache:
    """GPU-based KV cache."""

    def __init__(self, args: ModelArgs, device: torch.device):
        self.device = device
        self.n_layers = args.n_layers
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size

        print(f"Allocating KV cache on {device}...")
        self.kv_cache = torch.zeros(
            args.n_layers, args.max_batch_size, args.max_seq_len, args.kv_lora_rank,
            dtype=torch.bfloat16, device=device
        )
        self.pe_cache = torch.zeros(
            args.n_layers, args.max_batch_size, args.max_seq_len, args.qk_rope_head_dim,
            dtype=torch.bfloat16, device=device
        )

        cache_size = (self.kv_cache.numel() + self.pe_cache.numel()) * 2
        print(f"KV cache allocated: {cache_size / 1e6:.1f} MB")

    def update(self, layer: int, kv: torch.Tensor, pe: torch.Tensor, start_pos: int, seq_len: int):
        bsz = kv.size(0)
        self.kv_cache[layer, :bsz, start_pos:start_pos + seq_len] = kv
        self.pe_cache[layer, :bsz, start_pos:start_pos + seq_len] = pe

    def get(self, layer: int, bsz: int, end_pos: int):
        return (
            self.kv_cache[layer, :bsz, :end_pos],
            self.pe_cache[layer, :bsz, :end_pos]
        )


class DeepSeekV32GPUCompute:
    """Model with GPU compute offload - weights load from CPU, compute on GPU."""

    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args
        self.loader = TensorLoader(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = GPU_DEVICE

        # Initialize GPU KV cache
        self.kv_cache = KVCache(args, self.device)

        print("Loading non-expert weights into memory...")
        self._load_base_weights()
        print(f"Base weights loaded. Cache size: {self.loader.current_cache_size / 1e9:.2f} GB")

        self.freqs_cis = precompute_freqs_cis(args)

    def _load_base_weights(self):
        for name in tqdm(self.loader.index["weight_map"].keys(), desc="Loading weights"):
            if "experts" in name and "shared_experts" not in name:
                continue
            if "scale_inv" in name:
                continue
            try:
                self.loader.load(name, cache=True)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")

    def to_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transfer tensor to GPU."""
        return tensor.to(self.device)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        embed_weight = self.loader.load("model.embed_tokens.weight")
        # Embedding on CPU, then transfer to GPU
        result = F.embedding(tokens.cpu(), embed_weight)
        return result.to(self.device)

    def attention_gpu(self, x: torch.Tensor, layer: int, start_pos: int) -> torch.Tensor:
        """Attention with all compute on GPU."""
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen

        # Load weights and transfer to GPU
        wq_a = self.to_gpu(self.loader.load(f"model.layers.{layer}.self_attn.q_a_proj.weight"))
        wq_b = self.to_gpu(self.loader.load(f"model.layers.{layer}.self_attn.q_b_proj.weight"))
        wkv_a = self.to_gpu(self.loader.load(f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight"))
        wkv_b = self.to_gpu(self.loader.load(f"model.layers.{layer}.self_attn.kv_b_proj.weight"))
        wo = self.to_gpu(self.loader.load(f"model.layers.{layer}.self_attn.o_proj.weight"))

        # Q projection (GPU)
        q = F.linear(x, wq_a)

        # Q layernorm
        q_norm_name = f"model.layers.{layer}.self_attn.q_a_layernorm.weight"
        if q_norm_name in self.loader.index["weight_map"]:
            q_norm = self.to_gpu(self.loader.load(q_norm_name))
            q = rms_norm_gpu(q, q_norm, self.args.rms_norm_eps)

        q = F.linear(q, wq_b)
        q = q.view(bsz, seqlen, self.args.n_heads, self.args.qk_nope_head_dim + self.args.qk_rope_head_dim)

        # KV projection (GPU)
        kv = F.linear(x, wkv_a)
        kv_rank = kv[:, :, :self.args.kv_lora_rank]
        k_pe = kv[:, :, self.args.kv_lora_rank:]

        # KV layernorm
        kv_norm_name = f"model.layers.{layer}.self_attn.kv_a_layernorm.weight"
        if kv_norm_name in self.loader.index["weight_map"]:
            kv_norm = self.to_gpu(self.loader.load(kv_norm_name))
            kv_rank = rms_norm_gpu(kv_rank, kv_norm, self.args.rms_norm_eps)

        # Apply RoPE
        freqs = self.freqs_cis[start_pos:end_pos]
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs).squeeze(2)

        # Update KV cache
        self.kv_cache.update(layer, kv_rank, k_pe, start_pos, seqlen)
        cached_kv, cached_pe = self.kv_cache.get(layer, bsz, end_pos)

        # Split Q
        q_nope, q_pe = q.split([self.args.qk_nope_head_dim, self.args.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs)

        # Attention computation
        if seqlen > 1:
            # Prefill: full attention
            kv_expanded = F.linear(cached_kv, wkv_b)
            kv_expanded = kv_expanded.view(bsz, end_pos, self.args.n_heads,
                                           self.args.qk_nope_head_dim + self.args.v_head_dim)
            k_nope, v = kv_expanded.split([self.args.qk_nope_head_dim, self.args.v_head_dim], dim=-1)

            k_pe_expanded = cached_pe.unsqueeze(2).expand(-1, -1, self.args.n_heads, -1)
            k = torch.cat([k_nope, k_pe_expanded], dim=-1)
            q = torch.cat([q_nope, q_pe], dim=-1)

            scale = self.args.get_softmax_scale()
            scores = torch.einsum("bshd,bthd->bsht", q.float(), k.float()) * scale

            mask = torch.triu(torch.full((seqlen, end_pos), float("-inf"), device=self.device),
                            diagonal=end_pos - seqlen + 1)
            scores = scores + mask.unsqueeze(0).unsqueeze(2)

            attn = scores.softmax(dim=-1).to(v.dtype)
            out = torch.einsum("bsht,bthd->bshd", attn, v)
        else:
            # Decode: compressed attention
            wkv_b_reshaped = wkv_b.view(self.args.n_heads, -1, self.args.kv_lora_rank)
            q_nope_proj = torch.einsum("bshd,hdc->bshc", q_nope.float(),
                                        wkv_b_reshaped[:, :self.args.qk_nope_head_dim].float())

            scale = self.args.get_softmax_scale()
            scores = (torch.einsum("bshc,btc->bsht", q_nope_proj, cached_kv.float()) +
                     torch.einsum("bshr,btr->bsht", q_pe.float(), cached_pe.float())) * scale

            attn = scores.softmax(dim=-1)
            out = torch.einsum("bsht,btc->bshc", attn, cached_kv.float())
            out = torch.einsum("bshc,hdc->bshd", out, wkv_b_reshaped[:, -self.args.v_head_dim:].float())
            out = out.to(torch.bfloat16)

        # Output projection
        out = out.flatten(2)
        out = F.linear(out, wo)

        return out

    def mlp_dense_gpu(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Dense MLP on GPU."""
        w1 = self.to_gpu(self.loader.load(f"model.layers.{layer}.mlp.gate_proj.weight"))
        w2 = self.to_gpu(self.loader.load(f"model.layers.{layer}.mlp.down_proj.weight"))
        w3 = self.to_gpu(self.loader.load(f"model.layers.{layer}.mlp.up_proj.weight"))

        gate = F.linear(x, w1)
        up = F.linear(x, w3)
        hidden = F.silu(gate.float()) * up.float()
        return F.linear(hidden.to(x.dtype), w2)

    def mlp_moe_gpu(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """MoE MLP with GPU compute."""
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)

        # Gate on GPU
        gate_weight = self.to_gpu(self.loader.load(f"model.layers.{layer}.mlp.gate.weight"))
        scores = F.linear(x_flat.float(), gate_weight.float())
        scores = scores.sigmoid()

        bias_name = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
        if bias_name in self.loader.index["weight_map"]:
            bias = self.to_gpu(self.loader.load(bias_name))
            scores_biased = scores + bias
        else:
            scores_biased = scores

        topk = self.args.n_activated_experts
        topk_scores, topk_indices = scores_biased.topk(topk, dim=-1)

        weights = scores.gather(1, topk_indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.args.route_scale

        # Shared experts on GPU
        shared_out = torch.zeros_like(x_flat)
        for se_idx in range(self.args.n_shared_experts):
            prefix = f"model.layers.{layer}.mlp.shared_experts"
            if self.args.n_shared_experts == 1:
                w1 = self.to_gpu(self.loader.load(f"{prefix}.gate_proj.weight"))
                w2 = self.to_gpu(self.loader.load(f"{prefix}.down_proj.weight"))
                w3 = self.to_gpu(self.loader.load(f"{prefix}.up_proj.weight"))
            else:
                w1 = self.to_gpu(self.loader.load(f"{prefix}.{se_idx}.gate_proj.weight"))
                w2 = self.to_gpu(self.loader.load(f"{prefix}.{se_idx}.down_proj.weight"))
                w3 = self.to_gpu(self.loader.load(f"{prefix}.{se_idx}.up_proj.weight"))

            gate = F.linear(x_flat, w1)
            up = F.linear(x_flat, w3)
            hidden = F.silu(gate.float()) * up.float()
            shared_out = shared_out + F.linear(hidden.to(x_flat.dtype), w2)

        # Routed experts on GPU
        expert_out = torch.zeros_like(x_flat)
        unique_experts = topk_indices.unique().tolist()

        for expert_id in unique_experts:
            mask = (topk_indices == expert_id).any(dim=-1)
            if not mask.any():
                continue

            token_indices = mask.nonzero(as_tuple=True)[0]
            expert_tokens = x_flat[token_indices]

            expert_positions = (topk_indices[token_indices] == expert_id)
            token_weights = (weights[token_indices] * expert_positions.float()).sum(dim=-1, keepdim=True)

            # Load expert weights and transfer to GPU
            expert_tensors = self.loader.load_expert(layer, expert_id)
            if not expert_tensors:
                continue

            # Transfer to GPU for compute
            gate_w = self.to_gpu(expert_tensors["gate_proj"])
            up_w = self.to_gpu(expert_tensors["up_proj"])
            down_w = self.to_gpu(expert_tensors["down_proj"])

            # Compute on GPU
            gate = F.linear(expert_tokens, gate_w)
            up = F.linear(expert_tokens, up_w)
            hidden = F.silu(gate.float()) * up.float()
            out = F.linear(hidden.to(expert_tokens.dtype), down_w)

            expert_out[token_indices] = expert_out[token_indices] + (out * token_weights).to(expert_out.dtype)

        return (shared_out + expert_out).view(bsz, seqlen, dim)

    def transformer_block(self, x: torch.Tensor, layer: int, start_pos: int) -> torch.Tensor:
        # Input norm
        norm_w = self.to_gpu(self.loader.load(f"model.layers.{layer}.input_layernorm.weight"))
        h = rms_norm_gpu(x, norm_w, self.args.rms_norm_eps)

        # Attention
        attn_out = self.attention_gpu(h, layer, start_pos)
        x = x + attn_out

        # Post-attention norm
        post_norm_w = self.to_gpu(self.loader.load(f"model.layers.{layer}.post_attention_layernorm.weight"))
        h = rms_norm_gpu(x, post_norm_w, self.args.rms_norm_eps)

        # MLP
        if layer < self.args.n_dense_layers:
            mlp_out = self.mlp_dense_gpu(h, layer)
        else:
            mlp_out = self.mlp_moe_gpu(h, layer)

        x = x + mlp_out
        return x

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        x = self.embed(tokens)

        for layer in tqdm(range(self.args.n_layers), desc="Layers", leave=False):
            x = self.transformer_block(x, layer, start_pos)

        # Final norm
        norm_weight = self.to_gpu(self.loader.load("model.norm.weight"))
        x = rms_norm_gpu(x, norm_weight, self.args.rms_norm_eps)

        # LM head
        lm_head = self.to_gpu(self.loader.load("lm_head.weight"))
        logits = F.linear(x, lm_head)

        return logits

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7, use_chat_format: bool = True) -> str:
        if use_chat_format:
            # Apply chat template for proper formatting
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # Don't add BOS again - chat template already includes it
            tokens = self.tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt")
        else:
            tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        seq_len = tokens.shape[1]

        print(f"Prompt tokens: {seq_len}")
        print("Processing prompt...")

        # Prefill
        logits = self.forward(tokens, start_pos=0)

        print("Generating tokens...")
        generated = []

        for i in range(max_new_tokens):
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            token_tensor = next_token.unsqueeze(0)
            logits = self.forward(token_tensor, start_pos=seq_len + i)

            decoded = self.tokenizer.decode(next_token.item())
            print(decoded, end="", flush=True)
            generated.append(next_token.item())

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        print()
        full_tokens = tokens[0].tolist() + generated
        return self.tokenizer.decode(full_tokens, skip_special_tokens=True)


def main():
    print("=" * 60)
    print("DeepSeek-Math-V2 CPU Weights + GPU Compute")
    print("=" * 60)

    # Check GPU memory
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info(0)[0] / 1e9
        print(f"GPU 0 free memory: {free_mem:.1f} GB")

    args = ModelArgs()
    model = DeepSeekV32GPUCompute(MODEL_PATH, args)

    prompt = "What is 2 + 2?"
    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    result = model.generate(prompt, max_new_tokens=20, temperature=0.7)
    print("-" * 40)
    print(f"Full response: {result}")


if __name__ == "__main__":
    main()
