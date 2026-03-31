"""Layer sharding configuration and assignment.

Splits N transformer layers across M machines.
Each machine processes its assigned layers sequentially,
then sends activations to the next machine.
"""

from dataclasses import dataclass


@dataclass
class ShardConfig:
    """Configuration for layer sharding across machines.

    Attributes:
        n_layers: Total transformer layers in the model
        n_machines: Number of machines in the cluster
        machine_id: This machine's ID (0-indexed)
        peer_addresses: Network addresses of other machines
        port: Port for activation passing
    """
    n_layers: int = 60
    n_machines: int = 2
    machine_id: int = 0
    peer_addresses: list[str] = None
    port: int = 5555

    def __post_init__(self):
        if self.peer_addresses is None:
            self.peer_addresses = []

    @property
    def layers_per_machine(self) -> int:
        return self.n_layers // self.n_machines

    @property
    def layer_start(self) -> int:
        return self.machine_id * self.layers_per_machine

    @property
    def layer_end(self) -> int:
        if self.machine_id == self.n_machines - 1:
            return self.n_layers  # Last machine gets remainder
        return (self.machine_id + 1) * self.layers_per_machine

    @property
    def local_layers(self) -> range:
        return range(self.layer_start, self.layer_end)

    @property
    def is_first(self) -> bool:
        return self.machine_id == 0

    @property
    def is_last(self) -> bool:
        return self.machine_id == self.n_machines - 1

    @property
    def next_peer(self) -> str | None:
        if self.is_last:
            return self.peer_addresses[0] if self.peer_addresses else None
        next_id = self.machine_id + 1
        if next_id < len(self.peer_addresses):
            return self.peer_addresses[next_id]
        return None

    @property
    def prev_peer(self) -> str | None:
        if self.is_first:
            return None
        prev_id = self.machine_id - 1
        if prev_id < len(self.peer_addresses):
            return self.peer_addresses[prev_id]
        return None


class LayerShard:
    """Manages a contiguous shard of transformer layers.

    This machine processes layers [layer_start, layer_end) and
    passes activations to the next machine.

    The pipeline for each token:
    1. Receive activations from previous machine (or embed input if first)
    2. Process through local layers sequentially
    3. Send activations to next machine (or produce output if last)
    """

    def __init__(self, config: ShardConfig):
        self.config = config
        self.layers = []  # Populated during model loading

    def assign_layers(self, all_layers: list) -> list:
        """Extract this machine's layers from the full model.

        Args:
            all_layers: All transformer layers

        Returns:
            List of layers assigned to this machine
        """
        self.layers = all_layers[self.config.layer_start:self.config.layer_end]
        return self.layers

    def process(self, hidden_states, **kwargs):
        """Process hidden states through this shard's layers.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) input activations

        Returns:
            hidden_states after processing through local layers
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states

    def __repr__(self):
        return (
            f"LayerShard(machine={self.config.machine_id}, "
            f"layers={self.config.layer_start}-{self.config.layer_end - 1}, "
            f"n_layers={len(self.layers)})"
        )
