"""Main inference pipeline — orchestrates the three kings.

Uses mlx-lm as the model backend (correct weight loading, quantization,
architecture) and patches in our three kings:

1. Vision (TurboQuant) — compressed KV cache on attention layers
2. Mjolnir — SSD expert streaming (for 397B where experts don't fit in RAM)
3. Odin — distributed sharding across machines
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx

from core.config import InferenceConfig
from core.model import load_model, get_model_info, generate_with_compression
from sharding.network import ActivationChannel
from streaming.expert_loader import ExpertLoader, ExpertConfig


@dataclass
class GenerationStats:
    """Statistics from a generation run."""
    tokens_generated: int = 0
    total_time_s: float = 0.0
    prompt_tokens: int = 0
    expert_loads: int = 0
    cache_memory_mb: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_s > 0:
            return self.tokens_generated / self.total_time_s
        return 0.0


class ThreeKingsPipeline:
    """Orchestrates distributed MoE inference with all three kings.

    Usage:
        config = InferenceConfig(machine_id=0, peer_addresses=["192.168.1.2"])
        pipeline = ThreeKingsPipeline(config)
        pipeline.setup("mlx-community/Qwen3.5-35B-A3B-4bit")

        for text in pipeline.generate("Hello, world!"):
            print(text, end="", flush=True)
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.channel: Optional[ActivationChannel] = None
        self.expert_loader: Optional[ExpertLoader] = None
        self._stats = GenerationStats()

    def setup(self, model_path: str):
        """Initialize the pipeline.

        Loads model via mlx-lm, then patches in the three kings.
        """
        print(f"3Kings: Setting up on machine {self.config.machine_id}...")

        # Load model and tokenizer via mlx-lm (handles all quantization)
        # Strip expert weights when expert_dir exists — Mjolnir streams them from SSD
        has_expert_files = Path(self.config.expert_dir).exists()
        print(f"  Loading model from {model_path}{'  (lazy, experts stripped)' if has_expert_files else ''}...")
        self.model, self.tokenizer = load_model(model_path, strip_experts=has_expert_files)

        info = get_model_info(self.model)
        print(f"  Model: {info['n_layers']} layers "
              f"({info['n_attention_layers']} attention + {info['n_delta_layers']} DeltaNet)"
              f"{', MoE' if info['has_moe'] else ', Dense'}")

        # Odin: network connection to peer
        if self.config.n_machines > 1 and self.config.peer_addresses:
            from sharding.shard import ShardConfig
            shard_config = ShardConfig(
                n_layers=self.config.model.n_layers,
                n_machines=self.config.n_machines,
                machine_id=self.config.machine_id,
                peer_addresses=self.config.peer_addresses,
                port=self.config.shard_port,
            )
            is_server = shard_config.is_first
            peer = self.config.peer_addresses[0]
            self.channel = ActivationChannel(
                is_server=is_server,
                peer_address=peer,
                port=self.config.shard_port,
            )
            print(f"  Odin: Connecting to peer {peer}...")
            self.channel.connect()

        # Mjolnir: expert streaming (only for MoE models)
        if info['has_moe']:
            expert_config = ExpertConfig(
                expert_dir=self.config.expert_dir,
                n_layers=self.config.model.n_layers,
                n_experts=self.config.model.n_experts,
                k_active=self.config.model.k_active,
                hidden_dim=self.config.model.hidden_dim,
                expert_intermediate=self.config.model.expert_intermediate,
                bits=self.config.expert_bits,
                layer_offset=self.config.local_layer_start,
                n_local_layers=self.config.local_layer_end - self.config.local_layer_start,
            )
            self.expert_loader = ExpertLoader(expert_config)
            try:
                self.expert_loader.open()
                print(f"  Mjolnir: Expert files ready")
            except FileNotFoundError:
                print(f"  Mjolnir: Expert files not found — using in-RAM experts")
                self.expert_loader = None

        # Vision: KV compression config
        if self.config.enable_kv_compression:
            print(f"  Vision: {self.config.key_bits}b keys / {self.config.value_bits}b values "
                  f"(4.5-bit effective, 3.4x compression)")

        mem = self.config.memory_estimate()
        print(f"  Memory: ~{mem['total_fixed_gb']:.1f} GB fixed, "
              f"~{mem['available_page_cache_gb']:.0f} GB page cache")
        print(f"3Kings: Ready.\n")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = True,
    ):
        """Generate text from a prompt.

        Currently uses mlx-lm's native generation.
        KV compression and expert streaming will be patched in next iteration.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Pipeline not set up. Call setup() first.")

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p

        t_start = time.time()

        # Use mlx-lm generation
        response = generate_with_compression(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            key_bits=self.config.key_bits,
            value_bits=self.config.value_bits,
            verbose=False,
        )

        self._stats.total_time_s = time.time() - t_start
        self._stats.tokens_generated = len(self.tokenizer.encode(response))

        if stream:
            yield response
        else:
            yield response

    def generate_interactive(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text and return as a single string."""
        results = list(self.generate(
            prompt, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p,
            stream=False,
        ))
        return results[0] if results else ""

    def teardown(self):
        """Clean up resources."""
        if self.channel:
            self.channel.close()
        if self.expert_loader:
            self.expert_loader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.teardown()

    def print_stats(self):
        """Print generation statistics."""
        s = self._stats
        print(f"\n=== Generation Stats ===")
        print(f"Generated tokens: {s.tokens_generated}")
        print(f"Total time:       {s.total_time_s:.2f}s")
        print(f"Decode speed:     {s.tokens_per_second:.1f} tok/s")
        if self.expert_loader:
            stats = self.expert_loader.stats
            print(f"Expert loads:     {stats['total_loads']}")
        print(f"========================\n")
