"""Odin — Distributed layer sharding across Apple Silicon machines.

Splits transformer layers across machines and handles
activation passing over the network.
"""

from sharding.shard import ShardConfig, LayerShard
from sharding.network import ActivationChannel
from sharding.coordinator import Coordinator

__all__ = ["ShardConfig", "LayerShard", "ActivationChannel", "Coordinator"]
