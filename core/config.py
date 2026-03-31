"""Model and inference configuration for 3KingsInference.

Supports Qwen3.5-397B-A17B and Qwen3-235B-A22B architectures.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Defaults are for Qwen3.5-397B-A17B.
    """
    name: str = "Qwen3.5-397B-A17B"
    n_layers: int = 60
    n_full_attention_layers: int = 15
    n_delta_net_layers: int = 45
    hidden_dim: int = 4096
    n_heads: int = 64
    n_kv_heads: int = 8
    head_dim: int = 128
    n_experts: int = 512
    k_active: int = 4
    has_shared_expert: bool = True
    expert_intermediate: int = 12288
    vocab_size: int = 152064
    max_context: int = 131072
    rope_theta: float = 1000000.0

    @classmethod
    def qwen_397b(cls) -> "ModelConfig":
        """Qwen3.5-397B-A17B configuration."""
        return cls()

    @classmethod
    def qwen_235b(cls) -> "ModelConfig":
        """Qwen3-235B-A22B configuration."""
        return cls(
            name="Qwen3-235B-A22B",
            n_layers=94,
            n_full_attention_layers=94,  # Dense attention (all layers)
            n_delta_net_layers=0,
            hidden_dim=4096,
            n_heads=64,
            n_kv_heads=4,
            head_dim=128,
            n_experts=128,
            k_active=8,
            has_shared_expert=True,
            expert_intermediate=12288,
        )


@dataclass
class InferenceConfig:
    """Full inference configuration for 3Kings.

    Combines model, sharding, streaming, and compression settings.
    """
    model: ModelConfig = field(default_factory=ModelConfig)

    # Sharding (Odin)
    n_machines: int = 2
    machine_id: int = 0
    peer_addresses: list[str] = field(default_factory=list)
    shard_port: int = 5555

    # Streaming (Mjolnir)
    expert_dir: str = "./packed_experts"
    expert_bits: int = 4

    # Compression (Vision)
    enable_kv_compression: bool = True
    key_bits: int = 4
    value_bits: int = 5
    enable_sparse_v: bool = True
    sparse_v_percentile: float = 50.0

    # Inference
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

    @property
    def layers_per_machine(self) -> int:
        return self.model.n_layers // self.n_machines

    @property
    def local_layer_start(self) -> int:
        return self.machine_id * self.layers_per_machine

    @property
    def local_layer_end(self) -> int:
        if self.machine_id == self.n_machines - 1:
            return self.model.n_layers
        return (self.machine_id + 1) * self.layers_per_machine

    def memory_estimate(self) -> dict:
        """Estimate memory usage for this configuration."""
        # Non-expert weights (shared across machines)
        non_expert_gb = 5.5 / self.n_machines

        # Expert working set (mmap'd, OS-managed)
        expert_working_gb = 6.0

        # KV cache per machine at various context lengths
        bytes_per_kv = 2  # bfloat16
        if self.enable_kv_compression:
            effective_bits = (self.key_bits + self.value_bits) / 2
            bytes_per_kv = effective_bits / 8

        n_local_attn_layers = sum(
            1 for i in range(self.local_layer_start, self.local_layer_end)
            if i < self.model.n_full_attention_layers or self.model.n_delta_net_layers == 0
        )

        kv_per_token = (
            2 * n_local_attn_layers * self.model.n_kv_heads *
            self.model.head_dim * bytes_per_kv
        )

        return {
            "non_expert_gb": non_expert_gb,
            "expert_working_gb": expert_working_gb,
            "os_overhead_gb": 6.0,
            "kv_per_token_bytes": kv_per_token,
            "kv_at_8k_gb": kv_per_token * 8192 / 1024**3,
            "kv_at_32k_gb": kv_per_token * 32768 / 1024**3,
            "total_fixed_gb": non_expert_gb + expert_working_gb + 6.0,
            "available_page_cache_gb": 64.0 - (non_expert_gb + expert_working_gb + 6.0),
        }
