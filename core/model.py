"""Qwen3.5 model definition for 3KingsInference.

Implements the hybrid architecture:
- Full attention layers (every 4th layer): Qwen3NextAttention with partial RoPE + output gating
- Linear attention layers (others): GatedDeltaNet state-space model
- MoE FFN: SwitchGLU with top-K routing + shared expert

Expert weights are NOT held in RAM — they stream from SSD via Mjolnir.
KV cache on attention layers is compressed via Vision (TurboQuant).
"""

import math
from dataclasses import dataclass
from typing import Optional, Any

import mlx.core as mx
import mlx.nn as nn

from core.config import ModelConfig


@dataclass
class ModelArgs:
    """Model arguments derived from ModelConfig + model JSON config."""
    hidden_size: int = 4096
    num_hidden_layers: int = 60
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 14336
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 131072
    full_attention_interval: int = 4
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 4
    moe_intermediate_size: int = 12288
    shared_expert_intermediate_size: int = 12288
    norm_topk_prob: bool = True
    # GatedDeltaNet
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    @classmethod
    def from_model_config(cls, cfg: ModelConfig) -> "ModelArgs":
        return cls(
            hidden_size=cfg.hidden_dim,
            num_hidden_layers=cfg.n_layers,
            num_attention_heads=cfg.n_heads,
            num_key_value_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            vocab_size=cfg.vocab_size,
            rope_theta=cfg.rope_theta,
            num_experts=cfg.n_experts,
            num_experts_per_tok=cfg.k_active,
            moe_intermediate_size=cfg.expert_intermediate,
            shared_expert_intermediate_size=cfg.expert_intermediate,
            max_position_embeddings=cfg.max_context,
        )


# ─── RMSNorm ───────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class RMSNormGated(nn.Module):
    """RMSNorm with SwiGLU gating for GatedDeltaNet output."""
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        normed = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is not None:
            return nn.silu(gate) * normed
        return normed


# ─── RoPE ───────────────────────────────────────────────────

class PartialRoPE(nn.Module):
    """Rotary position embeddings applied to a fraction of head_dim."""
    def __init__(self, head_dim: int, partial_factor: float = 0.25,
                 base: float = 1000000.0, max_seq_len: int = 131072):
        super().__init__()
        self.dims = int(head_dim * partial_factor)
        self.rope = nn.RoPE(self.dims, traditional=False, base=base)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return self.rope(x, offset=offset)


# ─── Full Attention (Qwen3NextAttention) ───────────────────

class Qwen3NextAttention(nn.Module):
    """Full softmax attention with per-head RMSNorm, partial RoPE, and output gating.

    Used every full_attention_interval layers (default: every 4th).
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5

        # Q projects to heads + gate (2x for gating)
        self.q_proj = nn.Linear(args.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, args.hidden_size, bias=False)

        # Per-head norms
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # RoPE
        self.rope = PartialRoPE(
            self.head_dim,
            partial_factor=args.partial_rotary_factor,
            base=args.rope_theta,
            max_seq_len=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        # Project Q (with gate), K, V
        qg = self.q_proj(x)
        q, gate = mx.split(qg, 2, axis=-1)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to heads: (B, L, H, D) -> (B, H, L, D)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        gate = gate.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Per-head normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        if cache is not None:
            offset = cache.offset
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        # GQA: expand KV heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        output = weights @ v

        # Output gating: output * sigmoid(gate)
        output = output * mx.sigmoid(gate)

        # Merge heads
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# ─── GatedDeltaNet (Linear Attention) ──────────────────────

class GatedDeltaNet(nn.Module):
    """Gated Delta Network — linear attention with state-space recurrence.

    No KV cache needed. Uses conv_state (kernel_size-1 frames) +
    rnn_state (H_v x D_k x D_v recurrent matrix).
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.key_dim = args.linear_key_head_dim
        self.value_dim = args.linear_value_head_dim

        total_k_dim = self.num_k_heads * self.key_dim
        total_v_dim = self.num_v_heads * self.value_dim
        conv_dim = total_k_dim * 2 + total_v_dim

        # Projections
        self.in_proj_qkv = nn.Linear(args.hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(args.hidden_size, total_v_dim, bias=False)
        self.in_proj_b = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = nn.Linear(total_v_dim, args.hidden_size, bias=False)

        # Depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.linear_conv_kernel_dim,
            groups=conv_dim,
            padding=0,
        )
        self.conv_kernel_size = args.linear_conv_kernel_dim

        # Learnable parameters for recurrence
        self.dt_bias = mx.zeros((self.num_v_heads,))
        self.A_log = mx.zeros((self.num_v_heads,))

        # Output norm with gating
        self.norm = RMSNormGated(total_v_dim, eps=args.rms_norm_eps)

        self._total_k_dim = total_k_dim
        self._total_v_dim = total_v_dim

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.in_proj_qkv(x)    # (B, L, conv_dim)
        z = self.in_proj_z(x)         # (B, L, total_v_dim)
        beta = mx.sigmoid(self.in_proj_b(x))   # (B, L, num_v_heads)
        a = self.in_proj_a(x)         # (B, L, num_v_heads)

        # Depthwise conv with causal padding
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            conv_input = mx.concatenate([conv_state, qkv.transpose(0, 2, 1)], axis=-1)
            # Update conv state
            new_conv_state = conv_input[..., -(self.conv_kernel_size - 1):]
        else:
            # Pad for causal conv
            pad = mx.zeros((B, self._total_k_dim * 2 + self._total_v_dim, self.conv_kernel_size - 1))
            conv_input = mx.concatenate([pad, qkv.transpose(0, 2, 1)], axis=-1)
            new_conv_state = conv_input[..., -(self.conv_kernel_size - 1):]

        conv_out = nn.silu(self.conv1d(conv_input))  # (B, conv_dim, L)
        conv_out = conv_out.transpose(0, 2, 1)       # (B, L, conv_dim)

        # Split into q, k, v
        q = conv_out[..., :self._total_k_dim]
        k = conv_out[..., self._total_k_dim:self._total_k_dim * 2]
        v = conv_out[..., self._total_k_dim * 2:]

        # Reshape to heads
        q = q.reshape(B, L, self.num_k_heads, self.key_dim)
        k = k.reshape(B, L, self.num_k_heads, self.key_dim)
        v = v.reshape(B, L, self.num_v_heads, self.value_dim)

        # Compute decay gate
        g = mx.exp(-mx.exp(self.A_log) * nn.softplus(a + self.dt_bias))  # (B, L, num_v_heads)

        # Gated delta recurrence
        if cache is not None and cache[1] is not None:
            rnn_state = cache[1]
        else:
            rnn_state = mx.zeros((B, self.num_v_heads, self.key_dim, self.value_dim))

        outputs = []
        for t in range(L):
            q_t = q[:, t]     # (B, Hk, Dk)
            k_t = k[:, t]     # (B, Hk, Dk)
            v_t = v[:, t]     # (B, Hv, Dv)
            g_t = g[:, t]     # (B, Hv)
            beta_t = beta[:, t]  # (B, Hv)

            # Decay state
            rnn_state = rnn_state * g_t[:, :, None, None]

            # Compute memory retrieval (for delta rule)
            # Expand k for value heads: (B, Hk, Dk) -> (B, Hv, Dk) via repeat
            heads_per_kv = self.num_v_heads // self.num_k_heads
            k_expanded = mx.repeat(k_t, heads_per_kv, axis=1)
            q_expanded = mx.repeat(q_t, heads_per_kv, axis=1)

            # kv_mem = state @ k  -> (B, Hv, Dv)
            kv_mem = mx.sum(rnn_state * k_expanded[:, :, :, None], axis=2)

            # Delta update
            delta = (v_t - kv_mem) * beta_t[:, :, None]

            # Update state: state += outer(k, delta)
            rnn_state = rnn_state + k_expanded[:, :, :, None] * delta[:, :, None, :]

            # Output: y = state @ q -> (B, Hv, Dv)
            y_t = mx.sum(rnn_state * q_expanded[:, :, :, None], axis=2)
            outputs.append(y_t)

        new_rnn_state = rnn_state
        y = mx.stack(outputs, axis=1)  # (B, L, Hv, Dv)
        y = y.reshape(B, L, -1)        # (B, L, total_v_dim)

        # Gated output norm
        y = self.norm(y, z)

        # Update cache
        if cache is not None:
            cache[0] = new_conv_state
            cache[1] = new_rnn_state

        return self.out_proj(y)


# ─── MoE FFN with SSD Streaming ────────────────────────────

class SharedExpert(nn.Module):
    """Shared expert that always runs (SwiGLU MLP)."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.shared_expert_intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.shared_expert_intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.shared_expert_intermediate_size, args.hidden_size, bias=False)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_weight = mx.sigmoid(self.shared_expert_gate(x))
        hidden = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        return gate_weight * self.down_proj(hidden)


class StreamingMoEBlock(nn.Module):
    """MoE block that streams expert weights from SSD via Mjolnir.

    Expert weights are NOT nn.Module parameters — they're loaded
    on-demand from mmap'd files by the ExpertLoader.

    Only the router gate, shared expert, and routing logic are in RAM.
    """
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = args.num_experts
        self.k_active = args.num_experts_per_tok
        self.hidden_size = args.hidden_size
        self.expert_intermediate = args.moe_intermediate_size
        self.norm_topk_prob = args.norm_topk_prob

        # Router gate (small — stays in RAM)
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)

        # Shared expert (small — stays in RAM)
        self.shared_expert = SharedExpert(args)

        # Expert loader is set externally by the pipeline
        self.expert_loader = None

    def set_expert_loader(self, loader):
        """Attach Mjolnir expert loader for SSD streaming."""
        self.expert_loader = loader

    def _dequant_expert_weights(self, raw: mx.array, bits: int) -> mx.array:
        """Dequantize packed expert weights to float.

        For 4-bit: each uint8 byte holds 2 values.
        Affine dequant: (nibble * scale + bias).
        """
        if bits == 4:
            low = (raw & 0x0F).astype(mx.float32)
            high = ((raw >> 4) & 0x0F).astype(mx.float32)
            return mx.concatenate([low, high], axis=-1)
        return raw.astype(mx.float32)

    def _expert_forward(self, x: mx.array, expert_idx: int) -> mx.array:
        """Forward through a single expert using streamed weights.

        Uses SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))
        """
        if self.expert_loader is None:
            raise RuntimeError("Expert loader not attached. Call set_expert_loader().")

        gate_up_raw, down_raw = self.expert_loader.load_expert_pair(
            self.layer_idx, expert_idx
        )

        # Dequantize
        gate_up = self._dequant_expert_weights(gate_up_raw, self.expert_loader.config.bits)
        down = self._dequant_expert_weights(down_raw, self.expert_loader.config.bits)

        # Reshape: gate_up is [gate_proj | up_proj] fused
        total_params = gate_up.shape[0]
        half = total_params // 2

        # gate_proj and up_proj: (hidden, intermediate)
        gate_w = gate_up[:half].reshape(self.hidden_size, self.expert_intermediate)
        up_w = gate_up[half:].reshape(self.hidden_size, self.expert_intermediate)
        down_w = down.reshape(self.expert_intermediate, self.hidden_size)

        # SwiGLU
        hidden = nn.silu(x @ gate_w) * (x @ up_w)
        return hidden @ down_w

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape

        # Compute routing
        logits = self.gate(x)  # (B, L, num_experts)

        # Top-K selection
        top_k_indices = mx.argpartition(-logits, kth=self.k_active - 1, axis=-1)
        top_k_indices = top_k_indices[..., :self.k_active]
        top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
        gate_weights = mx.softmax(top_k_logits, axis=-1)

        if self.norm_topk_prob:
            gate_weights = gate_weights / mx.sum(gate_weights, axis=-1, keepdims=True)

        # Process each token through its selected experts
        # For simplicity in the streaming path, process per-token
        flat_x = x.reshape(-1, D)  # (B*L, D)
        flat_indices = top_k_indices.reshape(-1, self.k_active)  # (B*L, K)
        flat_weights = gate_weights.reshape(-1, self.k_active)   # (B*L, K)

        outputs = []
        for i in range(flat_x.shape[0]):
            token = flat_x[i:i+1]  # (1, D)
            expert_sum = mx.zeros_like(token)

            for j in range(self.k_active):
                eidx = flat_indices[i, j].item()
                w = flat_weights[i, j]
                expert_out = self._expert_forward(token, eidx)
                expert_sum = expert_sum + w * expert_out

            outputs.append(expert_sum)

        routed_output = mx.concatenate(outputs, axis=0).reshape(B, L, D)

        # Shared expert (always runs, in RAM)
        shared_output = self.shared_expert(x)

        return routed_output + shared_output


# ─── Decoder Layer ──────────────────────────────────────────

class DecoderLayer(nn.Module):
    """Single transformer layer — either full attention or GatedDeltaNet + MoE."""
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0

        # Attention: full or linear
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Qwen3NextAttention(args)

        # FFN: MoE with SSD streaming
        if args.num_experts > 0:
            self.mlp = StreamingMoEBlock(args, layer_idx)
        else:
            # Dense MLP fallback
            self.mlp = SharedExpert(args)

        # Layer norms
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Pre-norm attention
        residual = x
        x_norm = self.input_layernorm(x)

        if self.is_linear:
            attn_out = self.linear_attn(x_norm, mask=mask, cache=cache)
        else:
            attn_out = self.self_attn(x_norm, mask=mask, cache=cache)

        x = residual + attn_out

        # Pre-norm FFN (MoE)
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))

        return x


# ─── Full Model ─────────────────────────────────────────────

class Qwen3_5Model(nn.Module):
    """Qwen3.5 transformer backbone."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        x = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, mask=mask, cache=layer_cache)

        return self.norm(x)

    def make_cache(self) -> list:
        """Create cache objects for each layer.

        Full attention layers -> [None, None] placeholder for KV cache
        Linear layers -> [None, None] for [conv_state, rnn_state]
        """
        caches = []
        for layer in self.layers:
            caches.append([None, None])
        return caches


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 with language model head."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = Qwen3_5Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        hidden = self.model(input_ids, mask=mask, cache=cache)
        return self.lm_head(hidden)

    def make_cache(self) -> list:
        return self.model.make_cache()

    def set_expert_loaders(self, expert_loader):
        """Attach Mjolnir expert loader to all MoE layers."""
        for layer in self.model.layers:
            if hasattr(layer.mlp, 'set_expert_loader'):
                layer.mlp.set_expert_loader(expert_loader)
