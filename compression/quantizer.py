"""TurboQuant MSE quantizer — Algorithm 1 from the paper.

Implements the MSE-optimal quantization path ONLY.
We do NOT implement TurboQuant_prod (Algorithm 2) because the
(b-1)-bit centroid resolution loss is amplified exponentially
by softmax, making it worse than pure b-bit MSE in practice.

Pipeline:
  1. Normalize: x_unit = x / ||x||, store ||x||
  2. Rotate: y = WHT(signs ⊙ x_unit)
  3. Quantize: idx_j = nearest_centroid(y_j) per coordinate
  4. Pack indices into bit-packed uint32 words

Dequantize:
  1. Unpack indices → look up centroids
  2. Inverse rotate: x̃ = signs ⊙ WHT(centroids)
  3. Rescale by stored norms

Reference: TurboQuant (arXiv:2504.19874), Algorithm 1
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from compression.codebook import get_codebook, quantize_to_indices
from compression.packing import pack_indices, unpack_indices, packed_size
from compression.rotation import (
    Rotation, generate_rotation,
    rotate_forward, rotate_inverse, safe_normalize,
)


@dataclass
class QuantizedTensor:
    """Compressed representation of a tensor."""
    packed_indices: mx.array   # (..., n_words) uint32 bit-packed
    norms: mx.array            # (..., 1) float32 L2 norms
    bits: int                  # bits per coordinate
    head_dim: int              # original last dimension


class TurboQuantMSE:
    """TurboQuant MSE quantizer (Algorithm 1).

    Args:
        head_dim: Attention head dimension (must be power of 2)
        bits: Bits per coordinate (1, 2, 3, or 4)
        seed: Random seed for rotation matrix
        norm_bake: If True, fold norms into centroids during dequant
                   (eliminates 2 element-wise ops in attention)
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 3,
        seed: int = 42,
        norm_bake: bool = False,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.n_centroids = 2 ** bits
        self.norm_bake = norm_bake

        # Precompute rotation
        self.rotation = generate_rotation(head_dim, seed=seed)

        # Precompute codebook (scaled for head_dim)
        self.centroids, self.boundaries = get_codebook(bits, head_dim)
        mx.eval(self.centroids, self.boundaries)

    def quantize(self, x: mx.array) -> QuantizedTensor:
        """Quantize vectors.

        Args:
            x: (..., head_dim) float32 input vectors

        Returns:
            QuantizedTensor with packed indices and norms
        """
        # 1. Normalize to unit sphere
        x_unit, norms = safe_normalize(x)

        # 2. Rotate
        y = rotate_forward(x_unit, self.rotation)

        # 3. Quantize each coordinate independently
        indices = quantize_to_indices(y, self.boundaries)

        # 4. Pack
        packed = pack_indices(indices, self.bits)

        return QuantizedTensor(
            packed_indices=packed,
            norms=norms,
            bits=self.bits,
            head_dim=self.head_dim,
        )

    def dequantize(self, qt: QuantizedTensor) -> mx.array:
        """Reconstruct vectors from quantized representation.

        Args:
            qt: QuantizedTensor from quantize()

        Returns:
            x_hat: (..., head_dim) float32 reconstructed vectors
        """
        # 1. Unpack indices
        indices = unpack_indices(qt.packed_indices, qt.bits, qt.head_dim)

        # 2. Look up centroids
        y_hat = self.centroids[indices]

        # 3. Inverse rotation
        x_hat = rotate_inverse(y_hat, self.rotation)

        # 4. Rescale by original norms
        x_hat = x_hat * qt.norms

        return x_hat

    def dequantize_rotated(self, qt: QuantizedTensor) -> mx.array:
        """Dequantize but stay in rotated space (skip inverse rotation).

        Useful for fused attention: compute scores in rotated space
        where queries are also rotated.

        Args:
            qt: QuantizedTensor

        Returns:
            y_hat: (..., head_dim) centroids in rotated space
        """
        indices = unpack_indices(qt.packed_indices, qt.bits, qt.head_dim)
        y_hat = self.centroids[indices]
        if self.norm_bake:
            y_hat = y_hat * qt.norms
        return y_hat

    def compression_ratio(self) -> float:
        """Compression ratio vs fp16 (16 bits per coordinate)."""
        # Effective bits: b bits per coord + amortized norm storage
        # Norm: 1 float32 (32 bits) per vector of head_dim coordinates
        effective_bits = self.bits + 32.0 / self.head_dim
        return 16.0 / effective_bits

    def theoretical_mse(self) -> float:
        """Theoretical MSE distortion from the paper (for unit vectors).

        Returns the expected ||x - x̃||² for ||x|| = 1.
        """
        # From Theorem 1, numerically computed values
        _mse_table = {1: 0.3634, 2: 0.1175, 3: 0.03045, 4: 0.00883}
        if self.bits in _mse_table:
            return _mse_table[self.bits]
        # General bound: sqrt(3)*pi/2 * 1/4^b
        import math
        return (math.sqrt(3) * math.pi / 2) * (1.0 / 4 ** self.bits)


class AsymmetricQuantizer:
    """Asymmetric quantizer using different precision for keys and values.

    Keys carry directional information (for Q·K dot products) and
    tolerate more compression. Values carry magnitude information
    (for weighted sums) and need higher precision.

    Default: keys at 3-bit, values at 4-bit.
    """

    def __init__(
        self,
        head_dim: int = 128,
        key_bits: int = 3,
        value_bits: int = 4,
        seed: int = 42,
    ):
        self.key_quantizer = TurboQuantMSE(head_dim, key_bits, seed=seed)
        self.value_quantizer = TurboQuantMSE(head_dim, value_bits, seed=seed)
        self.key_bits = key_bits
        self.value_bits = value_bits

    def quantize_kv(
        self, keys: mx.array, values: mx.array
    ) -> tuple[QuantizedTensor, QuantizedTensor]:
        """Quantize keys and values at different precisions."""
        return self.key_quantizer.quantize(keys), self.value_quantizer.quantize(values)

    def dequantize_kv(
        self, q_keys: QuantizedTensor, q_values: QuantizedTensor
    ) -> tuple[mx.array, mx.array]:
        """Dequantize keys and values."""
        return (
            self.key_quantizer.dequantize(q_keys),
            self.value_quantizer.dequantize(q_values),
        )

    def effective_bits(self) -> float:
        """Average bits per coordinate across K and V."""
        return (self.key_bits + self.value_bits) / 2.0

    def compression_ratio(self) -> float:
        """Compression ratio vs fp16 for combined K+V cache."""
        eff = self.effective_bits() + 32.0 / self.key_quantizer.head_dim
        return 16.0 / eff
