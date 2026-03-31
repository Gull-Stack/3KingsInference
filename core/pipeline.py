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
          - Compute routing logits -> top-K expert indices
          - Mjolnir: load K=4 expert weights from mmap'd SSD
          - Forward through experts + shared expert
          - Combine with gating weights
       c. If DeltaNet layer:
          - Standard linear attention (no KV cache)
    3. Odin: send activations to Machine B

  Machine B (layers 30-59):
    4. Receive activations from Machine A
    5. Process layers 30-59 (same as steps 2a-2c)
    6. Final norm + output projection -> logits
    7. Odin: send logits back to Machine A for sampling
"""

import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np

from core.config import InferenceConfig
from core.model import Qwen3_5ForCausalLM, ModelArgs
from core.tokenizer import Tokenizer
from core.weights import load_model_from_path, config_to_model_args, load_model_config
from compression.kv_cache import CompressedKVCache, HybridCacheManager
from sharding.shard import ShardConfig, LayerShard
from sharding.network import ActivationChannel
from streaming.expert_loader import ExpertLoader, ExpertConfig
from streaming.router import TopKRouter


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
        pipeline.setup(model_path="mlx-community/Qwen3.5-397B-A17B-4bit")

        for token_text in pipeline.generate("Hello, world!"):
            print(token_text, end="", flush=True)
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

        # Model + tokenizer
        self.model: Optional[Qwen3_5ForCausalLM] = None
        self.tokenizer: Optional[Tokenizer] = None

        # Vision (compression) — hybrid cache manager
        self.cache_manager: Optional[HybridCacheManager] = None

        self._stats = GenerationStats()

    def setup(self, model_path: str):
        """Initialize all three kings.

        Args:
            model_path: Path to model weights (HF repo or local dir)
        """
        print(f"3Kings: Setting up on machine {self.config.machine_id}...")

        # Load tokenizer
        print("  Loading tokenizer...")
        self.tokenizer = Tokenizer.from_pretrained(model_path)

        # Build model and load non-expert weights from safetensors
        print("  Loading model (non-expert weights only)...")
        try:
            self.model, model_args = load_model_from_path(model_path)
        except FileNotFoundError:
            # Fallback: build model without loading weights (for testing)
            print("  WARNING: Could not load weights, using random initialization")
            args = ModelArgs.from_model_config(self.config.model)
            self.model = Qwen3_5ForCausalLM(args)
            model_args = args

        # Odin: establish peer connection
        if self.config.n_machines > 1 and self.config.peer_addresses:
            is_server = self.shard_config.is_first
            peer = (
                self.config.peer_addresses[1] if is_server and len(self.config.peer_addresses) > 1
                else self.config.peer_addresses[0]
            )
            self.channel = ActivationChannel(
                is_server=is_server,
                peer_address=peer,
                port=self.config.shard_port,
            )
            print(f"  Odin: Connecting to peer...")
            self.channel.connect()

        # Mjolnir: open expert files
        print(f"  Mjolnir: Opening expert files from {self.config.expert_dir}...")
        self.expert_loader = ExpertLoader(self.expert_config)
        try:
            self.expert_loader.open()
            self.model.set_expert_loaders(self.expert_loader)
            print(f"  Mjolnir: {self.expert_config.n_local_layers} layers ready")
        except FileNotFoundError as e:
            print(f"  Mjolnir: Expert files not found ({e}). MoE will fail until files are prepared.")

        # Vision: set up compressed KV cache
        if self.config.enable_kv_compression:
            self.cache_manager = HybridCacheManager(
                n_layers=self.config.model.n_layers,
                full_attention_interval=args.full_attention_interval,
                head_dim=args.head_dim,
                num_kv_heads=args.num_key_value_heads,
                key_bits=self.config.key_bits,
                value_bits=self.config.value_bits,
            )
            n_attn = sum(1 for c in self.cache_manager.caches if isinstance(c, CompressedKVCache))
            print(f"  Vision: {n_attn} attention layers with "
                  f"{self.config.key_bits}b keys / {self.config.value_bits}b values")

        mem = self.config.memory_estimate()
        print(f"  Memory: ~{mem['total_fixed_gb']:.1f} GB fixed, "
              f"~{mem['available_page_cache_gb']:.0f} GB page cache")
        print(f"3Kings: Pipeline ready.\n")

    def _build_causal_mask(self, seq_len: int, offset: int = 0) -> mx.array:
        """Build causal attention mask."""
        total_len = offset + seq_len
        mask = mx.full((seq_len, total_len), -1e9)
        for i in range(seq_len):
            mask = mask.at[i, :offset + i + 1].add(1e9)
        return mask[None, None, :, :]  # (1, 1, L, L_total)

    def forward(self, input_ids: mx.array) -> mx.array:
        """Run forward pass through the model.

        For distributed inference:
        - Machine 0: embed -> layers 0-29 -> send activations
        - Machine 1: receive activations -> layers 30-59 -> norm -> logits -> send back

        Args:
            input_ids: (B, L) token IDs

        Returns:
            logits: (B, L, vocab_size) — only on machine 0, or on single-machine mode
        """
        B, L = input_ids.shape

        # Determine cache offset for RoPE
        cache_offset = 0
        if self.cache_manager is not None:
            for c in self.cache_manager.caches:
                if isinstance(c, CompressedKVCache):
                    cache_offset = c.offset
                    break

        mask = self._build_causal_mask(L, offset=cache_offset)

        if self.config.n_machines <= 1:
            # Single machine: run everything
            return self.model(input_ids, mask=mask,
                            cache=self.cache_manager.caches if self.cache_manager else None)

        # Distributed mode
        if self.shard_config.is_first:
            # Machine 0: embed + local layers
            x = self.model.model.embed_tokens(input_ids)

            for i in range(self.shard_config.layer_start, self.shard_config.layer_end):
                layer = self.model.model.layers[i]
                cache = self.cache_manager.caches[i] if self.cache_manager else None
                x = layer(x, mask=mask, cache=cache)

            # Send activations to machine 1
            self.channel.send_activation(x)

            # Receive logits back from machine 1
            logits = self.channel.recv_activation()
            return logits

        else:
            # Machine 1: receive activations, run local layers, send logits back
            x = self.channel.recv_activation()

            for i in range(self.shard_config.layer_start, self.shard_config.layer_end):
                layer = self.model.model.layers[i]
                cache = self.cache_manager.caches[i] if self.cache_manager else None
                x = layer(x, mask=mask, cache=cache)

            # Final norm + lm_head
            x = self.model.model.norm(x)
            logits = self.model.lm_head(x)

            # Send logits back to machine 0
            self.channel.send_activation(logits)
            return logits

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = True,
    ):
        """Generate text from a prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stream: If True, yield tokens as they're generated

        Yields:
            Generated text tokens (if stream=True)

        Returns:
            Full generated text (if stream=False)
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Pipeline not set up. Call setup() first.")

        max_tokens = max_tokens or self.config.max_tokens

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        self._stats.prompt_tokens = len(input_ids)

        # Prefill: process entire prompt
        tokens = mx.array([input_ids])
        logits = self.forward(tokens)
        mx.eval(logits)

        generated_text = ""
        t_start = time.time()

        for step in range(max_tokens):
            # Sample next token from last position
            next_logits = logits[:, -1, :]

            if temperature > 0:
                next_logits = next_logits / temperature
                # Top-p sampling
                probs = mx.softmax(next_logits, axis=-1)
                sorted_indices = mx.argsort(-probs, axis=-1)
                sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
                cumulative = mx.cumsum(sorted_probs, axis=-1)
                mask = cumulative - sorted_probs > top_p
                sorted_probs = mx.where(mask, 0.0, sorted_probs)
                sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
                # Sample
                next_token = mx.random.categorical(mx.log(sorted_probs + 1e-10))
                next_token = mx.take_along_axis(sorted_indices, next_token[:, None], axis=-1)
                next_token = next_token.squeeze(-1)
            else:
                next_token = mx.argmax(next_logits, axis=-1)

            token_id = next_token.item()

            # Check EOS
            if token_id == self.tokenizer.eos_token_id:
                break

            # Decode token
            token_text = self.tokenizer.decode([token_id])
            generated_text += token_text
            self._stats.tokens_generated += 1

            if stream:
                yield token_text

            # Next step: single token forward
            logits = self.forward(next_token.reshape(1, 1))
            mx.eval(logits)

        self._stats.total_time_s = time.time() - t_start

        if self.expert_loader:
            self._stats.expert_loads = self.expert_loader.stats["total_loads"]
        if self.cache_manager:
            self._stats.cache_memory_mb = self.cache_manager.total_memory_bytes() / 1024**2

        if not stream:
            yield generated_text

    def teardown(self):
        """Clean up resources."""
        if self.channel:
            self.channel.close()
        if self.expert_loader:
            self.expert_loader.close()
        if self.cache_manager:
            self.cache_manager.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.teardown()

    def print_stats(self):
        """Print generation statistics."""
        s = self._stats
        print(f"\n=== Generation Stats ===")
        print(f"Prompt tokens:    {s.prompt_tokens}")
        print(f"Generated tokens: {s.tokens_generated}")
        print(f"Total time:       {s.total_time_s:.2f}s")
        print(f"Decode speed:     {s.tokens_per_second:.1f} tok/s")
        print(f"Expert loads:     {s.expert_loads}")
        print(f"KV cache memory:  {s.cache_memory_mb:.1f} MB")
        print(f"========================\n")
