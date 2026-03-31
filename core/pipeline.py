"""Main inference pipeline — orchestrates the three kings.

Pipeline flow per token:

  Machine A (layers 0-29):
    1. Embed token (or receive activations from Machine B for autoregressive)
    2. For each local layer:
       a. If full-attention layer:
          - Compute Q, K, V projections
          - Vision: compress K, V into TurboQuant cache
          - Compute attention using compressed cache
       b. If MoE layer:
          - Compute routing logits → top-K expert indices
          - Mjolnir: load K=4 expert weights from mmap'd SSD
          - Forward through experts + shared expert
          - Combine with gating weights
       c. If DeltaNet layer:
          - Standard linear attention (no KV cache)
    3. Odin: send activations to Machine B

  Machine B (layers 30-59):
    4. Receive activations from Machine A
    5. Process layers 30-59 (same as steps 2a-2c)
    6. Final norm + output projection → logits
    7. Odin: send logits back to Machine A for sampling
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from core.config import InferenceConfig
from sharding.shard import ShardConfig, LayerShard
from sharding.network import ActivationChannel
from streaming.expert_loader import ExpertLoader, ExpertConfig
from streaming.router import TopKRouter


class ThreeKingsPipeline:
    """Orchestrates distributed MoE inference with all three kings.

    Usage:
        config = InferenceConfig(machine_id=0, peer_addresses=["192.168.1.2"])
        pipeline = ThreeKingsPipeline(config)
        pipeline.setup()

        for token in generate(prompt):
            logits = pipeline.forward(token)
            next_token = sample(logits)
    """

    def __init__(self, config: InferenceConfig):
        self.config = config

        # Odin (sharding)
        self.shard_config = ShardConfig(
            n_layers=config.model.n_layers,
            n_machines=config.n_machines,
            machine_id=config.machine_id,
            peer_addresses=config.peer_addresses,
            port=config.shard_port,
        )
        self.shard = LayerShard(self.shard_config)
        self.channel: Optional[ActivationChannel] = None

        # Mjolnir (streaming)
        self.expert_config = ExpertConfig(
            expert_dir=config.expert_dir,
            n_layers=config.model.n_layers,
            n_experts=config.model.n_experts,
            k_active=config.model.k_active,
            hidden_dim=config.model.hidden_dim,
            expert_intermediate=config.model.expert_intermediate,
            bits=config.expert_bits,
            layer_offset=self.shard_config.layer_start,
            n_local_layers=len(list(self.shard_config.local_layers)),
        )
        self.expert_loader: Optional[ExpertLoader] = None
        self.router = TopKRouter(
            n_experts=config.model.n_experts,
            k_active=config.model.k_active,
            has_shared_expert=config.model.has_shared_expert,
        )

        # Vision (compression) — initialized per attention layer
        self.kv_caches: dict[int, object] = {}  # layer_idx → compressed KV cache

    def setup(self):
        """Initialize all three kings.

        Call this before inference. Sets up:
        1. Network connection to peer (Odin)
        2. mmap'd expert files (Mjolnir)
        3. KV cache compression configs (Vision)
        """
        # Odin: establish peer connection
        if self.config.n_machines > 1:
            is_server = self.shard_config.is_first
            peer = (
                self.config.peer_addresses[1] if is_server
                else self.config.peer_addresses[0]
            )
            self.channel = ActivationChannel(
                is_server=is_server,
                peer_address=peer,
                port=self.config.shard_port,
            )
            self.channel.connect()

        # Mjolnir: open expert files
        self.expert_loader = ExpertLoader(self.expert_config)
        self.expert_loader.open()

        # Vision: set up KV compression for attention layers
        if self.config.enable_kv_compression:
            from compression.quantizer import TurboQuantMSE
            # Create compressors for each full-attention layer on this machine
            for layer_idx in self.shard_config.local_layers:
                if self._is_attention_layer(layer_idx):
                    self.kv_caches[layer_idx] = {
                        "key_quantizer": TurboQuantMSE(
                            head_dim=self.config.model.head_dim,
                            bits=self.config.key_bits,
                        ),
                        "value_quantizer": TurboQuantMSE(
                            head_dim=self.config.model.head_dim,
                            bits=self.config.value_bits,
                        ),
                        "keys": [],
                        "values": [],
                    }

        print(f"3Kings: Pipeline ready on machine {self.config.machine_id}")
        print(f"  Odin:    Layers {self.shard_config.layer_start}-{self.shard_config.layer_end - 1}")
        print(f"  Mjolnir: {self.expert_config.n_local_layers} layers, "
              f"{self.config.model.n_experts} experts each")
        print(f"  Vision:  {len(self.kv_caches)} attention layers with "
              f"{self.config.key_bits}b keys / {self.config.value_bits}b values")

        mem = self.config.memory_estimate()
        print(f"  Memory:  ~{mem['total_fixed_gb']:.1f} GB fixed, "
              f"~{mem['available_page_cache_gb']:.0f} GB page cache")

    def _is_attention_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses full attention (vs DeltaNet)."""
        if self.config.model.n_delta_net_layers == 0:
            return True  # All layers are attention (dense model)
        # For Qwen3.5-397B: specific layers are full attention
        # This would be loaded from model config
        return layer_idx < self.config.model.n_full_attention_layers

    def teardown(self):
        """Clean up resources."""
        if self.channel:
            self.channel.close()
        if self.expert_loader:
            self.expert_loader.close()

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *args):
        self.teardown()

    def print_stats(self):
        """Print inference statistics."""
        if self.expert_loader:
            stats = self.expert_loader.stats
            print(f"\nMjolnir Stats:")
            print(f"  Expert loads: {stats['total_loads']}")
            print(f"  Layers accessed: {stats['layers_accessed']}")
            print(f"  Estimated bytes from SSD: "
                  f"{stats['estimated_bytes_read'] / 1024**2:.0f} MB")

        print(f"\nVision Stats:")
        print(f"  Compressed KV caches: {len(self.kv_caches)}")
        if self.kv_caches:
            total_entries = sum(
                len(cache["keys"]) for cache in self.kv_caches.values()
            )
            print(f"  Total cached tokens: {total_entries}")
