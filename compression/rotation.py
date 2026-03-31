"""Structured random rotation via Walsh-Hadamard Transform + random sign flips.

Instead of a dense QR decomposition (O(d²) compute, O(d²) storage),
we use WHT + random signs which gives:
- O(d log d) compute for rotation
- O(d) storage (just the random sign vector)
- Same Gaussianization quality at d >= 64

The WHT uniformizes unit-sphere vectors just as well as a dense random
rotation because it's an orthogonal transform, and the random sign flips
break any structure in the input.

For the JL projection matrix (used in optional QJL), we still use a
dense Gaussian matrix since it's only needed once per layer.

Reference: WUSH paper confirms Hadamard is optimal among data-agnostic transforms.
"""

import math
import mlx.core as mx


def _hadamard_matrix(d: int) -> mx.array:
    """Build a normalized Hadamard matrix of size d×d.

    Uses the recursive Sylvester construction:
    H_1 = [1]
    H_2n = [[H_n, H_n], [H_n, -H_n]] / sqrt(2)

    d must be a power of 2.
    """
    if d == 1:
        return mx.array([[1.0]], dtype=mx.float32)

    if d & (d - 1) != 0:
        raise ValueError(f"Hadamard matrix requires power-of-2 dimension, got {d}")

    H_half = _hadamard_matrix(d // 2)
    scale = 1.0 / math.sqrt(2.0)
    top = mx.concatenate([H_half, H_half], axis=1) * scale
    bottom = mx.concatenate([H_half, -H_half], axis=1) * scale
    return mx.concatenate([top, bottom], axis=0)


class Rotation:
    """Structured random rotation: WHT + random sign flips.

    Stores only the random signs (d values) and the precomputed
    Hadamard matrix (shared across all instances of same dimension).
    """

    def __init__(self, head_dim: int, signs: mx.array, hadamard: mx.array):
        self.head_dim = head_dim
        self.signs = signs          # (d,) values in {-1, +1}
        self.hadamard = hadamard    # (d, d) normalized Hadamard matrix

    @property
    def matrix(self) -> mx.array:
        """Dense rotation matrix (for compatibility). Avoid in hot path."""
        return self.hadamard * self.signs[None, :]


# Cache Hadamard matrices by dimension
_hadamard_cache: dict[int, mx.array] = {}


def _get_hadamard(d: int) -> mx.array:
    if d not in _hadamard_cache:
        _hadamard_cache[d] = _hadamard_matrix(d)
        mx.eval(_hadamard_cache[d])
    return _hadamard_cache[d]


def generate_rotation(head_dim: int, seed: int = 42) -> Rotation:
    """Generate a structured random rotation for the given dimension.

    Args:
        head_dim: Dimension (must be power of 2, e.g. 64, 128, 256)
        seed: Random seed for reproducibility

    Returns:
        Rotation object with WHT + random signs
    """
    if head_dim & (head_dim - 1) != 0:
        raise ValueError(f"head_dim must be power of 2, got {head_dim}")

    mx.random.seed(seed)
    # Random ±1 signs
    signs = (2.0 * (mx.random.uniform(shape=(head_dim,)) > 0.5).astype(mx.float32) - 1.0)
    mx.eval(signs)

    hadamard = _get_hadamard(head_dim)
    return Rotation(head_dim, signs, hadamard)


def rotate_forward(x: mx.array, rotation: Rotation) -> mx.array:
    """Apply random rotation: y = H @ (signs ⊙ x).

    The sign flip is applied element-wise first (O(d)),
    then the WHT is applied as a matrix multiply (O(d²) via matmul,
    but could be O(d log d) with a dedicated WHT kernel).

    Args:
        x: (..., d) input vectors
        rotation: Rotation object

    Returns:
        y: (..., d) rotated vectors
    """
    # Element-wise sign flip
    flipped = x * rotation.signs
    # WHT (using matmul — MLX may fuse this efficiently on Metal)
    return flipped @ rotation.hadamard.T


def rotate_inverse(y: mx.array, rotation: Rotation) -> mx.array:
    """Apply inverse rotation: x = signs ⊙ (H^T @ y).

    Since H is orthogonal and symmetric (H = H^T for Hadamard),
    and sign flips are self-inverse, the inverse is:
    x = signs ⊙ (H @ y)

    Args:
        y: (..., d) rotated vectors
        rotation: Rotation object

    Returns:
        x: (..., d) vectors in original space
    """
    # Inverse WHT (H is symmetric for Hadamard: H^T = H)
    unrotated = y @ rotation.hadamard
    # Inverse sign flip (signs are self-inverse: s * s = 1)
    return unrotated * rotation.signs


def safe_normalize(x: mx.array, axis: int = -1, eps: float = 1e-8) -> tuple[mx.array, mx.array]:
    """Normalize vectors to unit length, safe for zero vectors.

    Args:
        x: Input tensor
        axis: Axis along which to normalize
        eps: Minimum norm (below this, norm is treated as 1.0)

    Returns:
        (normalized, norms) where norms shape has keepdims on the given axis
    """
    norms = mx.linalg.norm(x, axis=axis, keepdims=True)
    safe_norms = mx.where(norms < eps, mx.ones_like(norms), norms)
    return x / safe_norms, norms


def generate_jl_matrix(head_dim: int, seed: int = 137) -> mx.array:
    """Generate a dense Gaussian JL projection matrix.

    Used for optional QJL residual correction. Not needed for
    TurboQuant_mse (our primary path).

    Args:
        head_dim: Dimension
        seed: Random seed (different from rotation seed)

    Returns:
        S: (head_dim, head_dim) float32 matrix with i.i.d. N(0,1) entries
    """
    mx.random.seed(seed)
    S = mx.random.normal(shape=(head_dim, head_dim))
    mx.eval(S)
    return S
