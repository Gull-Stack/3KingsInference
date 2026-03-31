"""Mjolnir Expert Loader — mmap-based expert weight streaming.

Core principle from Flash-MoE: "Trust the OS."
The macOS page cache handles caching via standard LRU.
Every custom caching approach (Metal LRU, malloc cache, LZ4)
was tested and found slower due to GPU memory pressure or overhead.

Instead of loading all 512 experts per layer into RAM (~6.75MB each,
~3.4GB per layer, ~200GB total), we mmap the expert weight files
and let the OS page cache manage which pages are resident.

With 54GB free for page cache per machine (after model overhead),
and only 30 layers of experts per machine (half the model),
we expect ~95%+ cache hit rate after warmup.
"""

import os
import mmap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np


@dataclass
class ExpertConfig:
    """Configuration for expert weight files.

    Attributes:
        expert_dir: Directory containing packed expert weight files
        n_layers: Total number of MoE layers
        n_experts: Number of experts per layer (e.g. 512)
        k_active: Number of active experts per token (e.g. 4)
        expert_size_bytes: Size of each expert's packed weights in bytes
        hidden_dim: Model hidden dimension
        expert_intermediate: Expert FFN intermediate dimension
        bits: Quantization bits for expert weights (4 or 2)
        layer_offset: First layer index this machine handles (for sharding)
        n_local_layers: Number of layers on this machine
    """
    expert_dir: str = "./packed_experts"
    n_layers: int = 60
    n_experts: int = 512
    k_active: int = 4
    expert_size_bytes: int = 6_750_000  # ~6.75MB per expert at 4-bit
    hidden_dim: int = 4096
    expert_intermediate: int = 12288
    bits: int = 4
    layer_offset: int = 0
    n_local_layers: int = 60


class ExpertLoader:
    """Loads expert weights on demand from mmap'd files.

    Each layer's experts are stored in a single file:
      packed_experts/layer_XX.bin

    File layout: [expert_0_weights | expert_1_weights | ... | expert_511_weights]
    Each expert is expert_size_bytes contiguous.

    We mmap the entire file and slice into it. The OS page cache
    handles which experts are resident in RAM.
    """

    def __init__(self, config: ExpertConfig):
        self.config = config
        self._mmaps: dict[int, mmap.mmap] = {}
        self._files: dict[int, int] = {}  # fd by layer
        self._stats = {
            "loads": 0,
            "layers_accessed": set(),
        }

    def open(self):
        """Open and mmap all layer files for this machine's layers."""
        expert_dir = Path(self.config.expert_dir)
        if not expert_dir.exists():
            raise FileNotFoundError(f"Expert directory not found: {expert_dir}")

        start = self.config.layer_offset
        end = start + self.config.n_local_layers

        for layer_idx in range(start, end):
            filepath = expert_dir / f"layer_{layer_idx:02d}.bin"
            if not filepath.exists():
                raise FileNotFoundError(f"Expert file not found: {filepath}")

            fd = os.open(str(filepath), os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            expected_size = self.config.n_experts * self.config.expert_size_bytes

            if file_size < expected_size:
                os.close(fd)
                raise ValueError(
                    f"Layer {layer_idx} file too small: {file_size} < {expected_size}"
                )

            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            self._mmaps[layer_idx] = mm
            self._files[layer_idx] = fd

    def close(self):
        """Close all mmap'd files."""
        for mm in self._mmaps.values():
            mm.close()
        for fd in self._files.values():
            os.close(fd)
        self._mmaps.clear()
        self._files.clear()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def load_experts(
        self,
        layer_idx: int,
        expert_indices: list[int],
    ) -> list[mx.array]:
        """Load specific experts for a layer from mmap'd storage.

        This is the hot path. The OS page cache determines whether
        this is a memory read (~nanoseconds) or an SSD read (~microseconds).

        Args:
            layer_idx: Which transformer layer
            expert_indices: Which experts to load (typically K=4)

        Returns:
            List of mx.array expert weight tensors
        """
        if layer_idx not in self._mmaps:
            raise KeyError(
                f"Layer {layer_idx} not mmap'd. "
                f"This machine handles layers {self.config.layer_offset} to "
                f"{self.config.layer_offset + self.config.n_local_layers - 1}"
            )

        mm = self._mmaps[layer_idx]
        expert_size = self.config.expert_size_bytes
        experts = []

        for eidx in expert_indices:
            if eidx < 0 or eidx >= self.config.n_experts:
                raise IndexError(f"Expert index {eidx} out of range [0, {self.config.n_experts})")

            offset = eidx * expert_size
            # Read raw bytes from mmap (page cache handles caching)
            raw = mm[offset:offset + expert_size]

            # Convert to MLX array
            # 4-bit packed: 2 values per byte, uint8 storage
            if self.config.bits == 4:
                packed = np.frombuffer(raw, dtype=np.uint8)
                tensor = mx.array(packed, dtype=mx.uint8)
            elif self.config.bits == 2:
                packed = np.frombuffer(raw, dtype=np.uint8)
                tensor = mx.array(packed, dtype=mx.uint8)
            else:
                # Full precision fallback
                weights = np.frombuffer(raw, dtype=np.float32)
                tensor = mx.array(weights, dtype=mx.float32)

            experts.append(tensor)

        self._stats["loads"] += len(expert_indices)
        self._stats["layers_accessed"].add(layer_idx)

        return experts

    def load_expert_pair(
        self,
        layer_idx: int,
        expert_idx: int,
    ) -> tuple[mx.array, mx.array]:
        """Load gate + up projection weights for a single expert.

        In SwiGLU MoE, each expert has:
          gate_proj: (hidden_dim, expert_intermediate)
          up_proj:   (hidden_dim, expert_intermediate)
          down_proj: (expert_intermediate, hidden_dim)

        Returns gate_up (fused) and down projections.
        """
        [raw] = self.load_experts(layer_idx, [expert_idx])

        # Split the raw packed tensor into gate_up and down
        # Exact split depends on model architecture
        gate_up_size = self.config.hidden_dim * self.config.expert_intermediate * 2
        if self.config.bits == 4:
            gate_up_size //= 2  # 4-bit: 2 values per byte

        gate_up = raw[:gate_up_size]
        down = raw[gate_up_size:]

        return gate_up, down

    @property
    def stats(self) -> dict:
        """Return loading statistics."""
        return {
            "total_loads": self._stats["loads"],
            "layers_accessed": len(self._stats["layers_accessed"]),
            "estimated_bytes_read": self._stats["loads"] * self.config.expert_size_bytes,
        }


def estimate_page_cache_usage(
    total_ram_gb: float,
    model_overhead_gb: float,
    os_overhead_gb: float = 6.0,
) -> dict:
    """Estimate available page cache and expected hit rate.

    Args:
        total_ram_gb: Total machine RAM (e.g. 64)
        model_overhead_gb: RAM used by non-expert model components
        os_overhead_gb: RAM reserved for macOS

    Returns:
        Dict with page cache estimates
    """
    available_gb = total_ram_gb - model_overhead_gb - os_overhead_gb
    return {
        "available_page_cache_gb": available_gb,
        "expert_size_mb": 6.75,
        "experts_cacheable": int(available_gb * 1024 / 6.75),
        "total_experts_per_machine": 512 * 30,  # 30 layers × 512 experts
        "estimated_residency": min(1.0, available_gb * 1024 / (512 * 30 * 6.75 / 1024)),
        "note": "Temporal locality means active experts are much smaller than total"
    }
