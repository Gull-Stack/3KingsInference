"""Lloyd-Max optimal scalar quantizer codebooks for N(0,1) distribution.

Hardcoded optimal centroids and boundaries for b=1,2,3,4 bits.
These minimize MSE under the standard normal distribution and are
mathematical constants — no runtime computation needed.

The centroids are computed via iterative conditional-expectation
(Lloyd's algorithm) with numerical integration. At d>=64, the
Beta distribution of rotated unit-sphere coordinates is well-
approximated by N(0, 1/d), so we use N(0,1) codebooks scaled
by 1/sqrt(head_dim).

Reference: TurboQuant (arXiv:2504.19874), Section 3.1
"""

import math
import mlx.core as mx


# Lloyd-Max centroids for N(0,1), sorted ascending.
# Boundaries are midpoints between consecutive centroids.
_CENTROIDS: dict[int, list[float]] = {
    1: [-0.7978845608028654, 0.7978845608028654],
    2: [
        -1.510417608611893, -0.4527800346911237,
        0.4527800346911237, 1.510417608611893,
    ],
    3: [
        -2.151945705166112, -1.3439092791423422,
        -0.7560052816730181, -0.2450941791152904,
        0.2450941791152904, 0.7560052816730181,
        1.3439092791423422, 2.151945705166112,
    ],
    4: [
        -2.732896755154294, -2.069364258154187,
        -1.618400443227723, -1.2565648452462146,
        -0.9426291036999694, -0.6569817464411519,
        -0.38818871416000605, -0.12844300124876415,
        0.12844300124876415, 0.38818871416000605,
        0.6569817464411519, 0.9426291036999694,
        1.2565648452462146, 1.618400443227723,
        2.069364258154187, 2.732896755154294,
    ],
}


def _compute_boundaries(centroids: list[float]) -> list[float]:
    """Compute decision boundaries as midpoints between consecutive centroids."""
    return [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(len(centroids) - 1)]


def get_codebook_unscaled(bits: int) -> tuple[mx.array, mx.array]:
    """Returns (centroids, boundaries) for N(0,1) without scaling.

    Args:
        bits: Number of bits per coordinate (1, 2, 3, or 4)

    Returns:
        centroids: mx.array shape (2^bits,)
        boundaries: mx.array shape (2^bits - 1,) — decision boundaries
    """
    if bits not in _CENTROIDS:
        raise ValueError(f"Supported bits: 1, 2, 3, 4. Got: {bits}")

    c = _CENTROIDS[bits]
    b = _compute_boundaries(c)
    return mx.array(c, dtype=mx.float32), mx.array(b, dtype=mx.float32)


def get_codebook(bits: int, head_dim: int) -> tuple[mx.array, mx.array]:
    """Returns (centroids, boundaries) scaled by 1/sqrt(head_dim).

    After random rotation of unit vectors in R^d, each coordinate is
    approximately N(0, 1/d). Scaling the N(0,1) codebook by 1/sqrt(d)
    gives the optimal codebook for this distribution.

    Args:
        bits: Number of bits per coordinate (1, 2, 3, or 4)
        head_dim: Attention head dimension (e.g. 128)

    Returns:
        centroids: mx.array shape (2^bits,)
        boundaries: mx.array shape (2^bits - 1,)
    """
    centroids, boundaries = get_codebook_unscaled(bits)
    scale = 1.0 / math.sqrt(head_dim)
    return centroids * scale, boundaries * scale


def quantize_to_indices(values: mx.array, boundaries: mx.array) -> mx.array:
    """Quantize values to bucket indices using decision boundaries.

    Uses searchsorted for O(log n) per element.

    Args:
        values: (..., D) float32 values to quantize
        boundaries: (2^bits - 1,) sorted decision boundaries

    Returns:
        indices: (..., D) uint32 bucket indices in [0, 2^bits)
    """
    # mx.searchsorted isn't available in all MLX versions,
    # so we use a broadcast comparison approach
    # values[..., D, 1] > boundaries[1, n_boundaries] → sum gives index
    expanded_vals = mx.expand_dims(values, axis=-1)  # (..., D, 1)
    expanded_bounds = boundaries.reshape(*([1] * (values.ndim - 1)), 1, -1)  # (1,...,1, 1, n_bounds)
    # Count how many boundaries each value exceeds
    indices = mx.sum(expanded_vals > expanded_bounds, axis=-1).astype(mx.uint32)
    return indices
