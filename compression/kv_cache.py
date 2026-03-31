"""Compressed KV cache manager — bridges TurboQuant with attention layers.

Stores key/value tensors in compressed form using asymmetric quantization
(4-bit keys, 5-bit values by default). Supports incremental token append
for autoregressive generation.

Replaces the standard KV cache that would hold uncompressed fp16/bf16 tensors.
At 4.5-bit effective, this is 3.4x smaller than fp16.
"""

from typing import Optional
from dataclasses import dataclass

import mlx.core as mx

from compression.quantizer import TurboQuantMSE, AsymmetricQuantizer, QuantizedTensor


class CompressedKVCache:
    """TurboQuant-compressed KV cache for a single attention layer.

    Usage:
        cache = CompressedKVCache(head_dim=128, key_bits=4, value_bits=5)
        k_out, v_out = cache.update_and_fetch(new_keys, new_values)
        # k_out, v_out are decompressed for attention computation
    """

    def __init__(
        self,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        key_bits: int = 4,
        value_bits: int = 5,
        seed: int = 42,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.dtype = dtype

        self.key_quantizer = TurboQuantMSE(head_dim, key_bits, seed=seed)
        self.value_quantizer = TurboQuantMSE(head_dim, value_bits, seed=seed)

        # Stored compressed tokens: lists of QuantizedTensor
        self._compressed_keys: list[QuantizedTensor] = []
        self._compressed_values: list[QuantizedTensor] = []

        # Track sequence length
        self._seq_len = 0

    @property
    def offset(self) -> int:
        """Current sequence position (for RoPE offset)."""
        return self._seq_len

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Compress new KV tokens and return full decompressed cache.

        Args:
            keys: (B, H, L_new, D) new key tensors
            values: (B, H, L_new, D) new value tensors

        Returns:
            all_keys: (B, H, L_total, D) decompressed full key cache
            all_values: (B, H, L_total, D) decompressed full value cache
        """
        B, H, L_new, D = keys.shape

        # Compress new tokens: reshape to (B*H*L_new, D) for quantization
        k_flat = keys.reshape(-1, D).astype(mx.float32)
        v_flat = values.reshape(-1, D).astype(mx.float32)

        q_keys = self.key_quantizer.quantize(k_flat)
        q_values = self.value_quantizer.quantize(v_flat)

        self._compressed_keys.append(q_keys)
        self._compressed_values.append(q_values)
        self._seq_len += L_new

        # Decompress everything for attention
        all_k_parts = []
        all_v_parts = []
        for qk, qv in zip(self._compressed_keys, self._compressed_values):
            dk = self.key_quantizer.dequantize(qk).astype(self.dtype)
            dv = self.value_quantizer.dequantize(qv).astype(self.dtype)
            all_k_parts.append(dk)
            all_v_parts.append(dv)

        all_k = mx.concatenate(all_k_parts, axis=0)  # (B*H*L_total, D)
        all_v = mx.concatenate(all_v_parts, axis=0)

        # Reshape back to (B, H, L_total, D)
        all_k = all_k.reshape(B, H, -1, D)
        all_v = all_v.reshape(B, H, -1, D)

        return all_k, all_v

    def clear(self):
        """Reset the cache."""
        self._compressed_keys.clear()
        self._compressed_values.clear()
        self._seq_len = 0

    def memory_bytes(self) -> int:
        """Estimate memory usage of compressed cache."""
        total = 0
        for qk in self._compressed_keys:
            total += qk.packed_indices.nbytes + qk.norms.nbytes
        for qv in self._compressed_values:
            total += qv.packed_indices.nbytes + qv.norms.nbytes
        return total

    def compression_stats(self) -> dict:
        """Return compression statistics."""
        compressed = self.memory_bytes()
        uncompressed = self._seq_len * self.num_kv_heads * self.head_dim * 2 * 2  # K+V, fp16
        ratio = uncompressed / compressed if compressed > 0 else 0
        return {
            "seq_len": self._seq_len,
            "compressed_bytes": compressed,
            "uncompressed_bytes": uncompressed,
            "compression_ratio": ratio,
            "effective_bits": (self.key_bits + self.value_bits) / 2.0,
        }


class HybridCacheManager:
    """Manages caches for the full model — compressed for attention, state for DeltaNet.

    Attention layers get CompressedKVCache (TurboQuant).
    DeltaNet layers get simple state arrays (conv_state + rnn_state).
    """

    def __init__(
        self,
        n_layers: int,
        full_attention_interval: int,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        key_bits: int = 4,
        value_bits: int = 5,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        self.caches = []
        for i in range(n_layers):
            is_attention = (i + 1) % full_attention_interval == 0
            if is_attention:
                self.caches.append(
                    CompressedKVCache(
                        head_dim=head_dim,
                        num_kv_heads=num_kv_heads,
                        key_bits=key_bits,
                        value_bits=value_bits,
                        dtype=dtype,
                    )
                )
            else:
                # DeltaNet: [conv_state, rnn_state]
                self.caches.append([None, None])

    def __getitem__(self, idx: int):
        return self.caches[idx]

    def __len__(self):
        return len(self.caches)

    def clear(self):
        for cache in self.caches:
            if isinstance(cache, CompressedKVCache):
                cache.clear()
            else:
                cache[0] = None
                cache[1] = None

    def total_memory_bytes(self) -> int:
        total = 0
        for cache in self.caches:
            if isinstance(cache, CompressedKVCache):
                total += cache.memory_bytes()
        return total

    def print_stats(self):
        n_attn = sum(1 for c in self.caches if isinstance(c, CompressedKVCache))
        n_delta = len(self.caches) - n_attn
        mem = self.total_memory_bytes()
        print(f"Cache: {n_attn} attention (compressed) + {n_delta} DeltaNet (state)")
        print(f"Compressed KV memory: {mem / 1024**2:.1f} MB")
