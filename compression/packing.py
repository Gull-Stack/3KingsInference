"""Bit packing utilities for quantized indices.

Packs integer indices (0..2^b-1) into uint32 words:
- 2-bit: 16 values per uint32 word
- 3-bit: 10 values per uint32 word (30 bits used, 2 wasted)
- 4-bit: 8 values per uint32 word

IMPORTANT: 3-bit uses proper 10-per-word packing, NOT wasteful
4-bit containers that throw away 25% of compression.
"""

import mlx.core as mx


def _vals_per_word(bits: int) -> int:
    """Number of values packed per uint32 word."""
    if bits == 1:
        return 32
    if bits == 2:
        return 16
    if bits == 3:
        return 10
    if bits == 4:
        return 8
    if bits == 5:
        return 6  # 30 bits used, 2 wasted (same as 3-bit)
    raise ValueError(f"Unsupported bits: {bits}. Use 1, 2, 3, 4, or 5.")


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack quantized indices into uint32 words.

    Args:
        indices: (..., D) uint32 indices in [0, 2^bits)
        bits: Bits per index (2, 3, or 4)

    Returns:
        packed: (..., ceil(D / vals_per_word)) uint32
    """
    vpw = _vals_per_word(bits)
    D = indices.shape[-1]
    batch_shape = indices.shape[:-1]

    # Pad D to multiple of vals_per_word
    padded_D = ((D + vpw - 1) // vpw) * vpw
    if padded_D > D:
        pad_width = padded_D - D
        indices = mx.concatenate([
            indices,
            mx.zeros((*batch_shape, pad_width), dtype=mx.uint32)
        ], axis=-1)

    # Reshape to (..., n_words, vpw)
    reshaped = indices.reshape(*batch_shape, -1, vpw)

    # Create shift amounts: [0, bits, 2*bits, ...]
    shifts = mx.array([i * bits for i in range(vpw)], dtype=mx.uint32)

    # Pack: shift each value and OR them together
    packed = mx.sum(reshaped << shifts, axis=-1).astype(mx.uint32)
    return packed


def unpack_indices(packed: mx.array, bits: int, D: int) -> mx.array:
    """Unpack uint32 words back to quantized indices.

    Args:
        packed: (..., n_words) uint32 packed words
        bits: Bits per index (2, 3, or 4)
        D: Original dimension (number of values to unpack)

    Returns:
        indices: (..., D) uint32 indices in [0, 2^bits)
    """
    vpw = _vals_per_word(bits)
    mask = (1 << bits) - 1
    batch_shape = packed.shape[:-1]

    # Create shift amounts
    shifts = mx.array([i * bits for i in range(vpw)], dtype=mx.uint32)

    # Expand and shift: (..., n_words, 1) >> (vpw,) → (..., n_words, vpw)
    expanded = mx.expand_dims(packed, axis=-1)
    unpacked = (expanded >> shifts) & mask

    # Flatten last two dims
    unpacked = unpacked.reshape(*batch_shape, -1)

    # Trim to original D
    return unpacked[..., :D].astype(mx.uint32)


def packed_size(D: int, bits: int) -> int:
    """Number of uint32 words needed to pack D values at given bit width."""
    vpw = _vals_per_word(bits)
    return (D + vpw - 1) // vpw
