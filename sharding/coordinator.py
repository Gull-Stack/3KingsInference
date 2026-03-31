"""Multi-machine coordinator for 3KingsInference.

Handles:
- Model weight distribution: which layers go to which machine
- Startup sequencing: machine A waits for machine B
- Health checks: verify both machines are ready
- Expert file verification: ensure each machine has its layer files
"""

import json
import socket
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from core.config import InferenceConfig
from sharding.shard import ShardConfig


@dataclass
class MachineStatus:
    """Status of a single machine in the cluster."""
    machine_id: int
    address: str
    port: int
    ready: bool = False
    layers: tuple[int, int] = (0, 0)  # (start, end)
    expert_files_ok: bool = False
    memory_gb: float = 0.0
    error: Optional[str] = None


class Coordinator:
    """Coordinates startup and inference across two machines.

    Machine 0 (primary) acts as the coordinator/server.
    Machine 1 (secondary) connects to machine 0.

    Startup sequence:
    1. Both machines start and load non-expert weights
    2. Machine 1 connects to machine 0
    3. Both verify expert files for their layers
    4. Coordinator signals "ready" to both
    5. Inference begins with ping-pong activation passing
    """

    COORD_PORT = 5554  # Separate from activation passing port

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.shard_config = ShardConfig(
            n_layers=config.model.n_layers,
            n_machines=config.n_machines,
            machine_id=config.machine_id,
            peer_addresses=config.peer_addresses,
            port=config.shard_port,
        )
        self.machines: dict[int, MachineStatus] = {}

    def verify_expert_files(self) -> bool:
        """Check that all expert files for this machine's layers exist."""
        expert_dir = Path(self.config.expert_dir)
        if not expert_dir.exists():
            return False

        for layer_idx in self.shard_config.local_layers:
            filepath = expert_dir / f"layer_{layer_idx:02d}.bin"
            if not filepath.exists():
                return False
        return True

    def verify_non_expert_weights(self, model_dir: str) -> bool:
        """Check that non-expert model weights exist."""
        model_path = Path(model_dir)
        # Check for safetensors or bin files
        has_weights = (
            list(model_path.glob("*.safetensors")) or
            list(model_path.glob("*.bin")) or
            list(model_path.glob("*.npz"))
        )
        has_config = (model_path / "config.json").exists()
        return bool(has_weights) and has_config

    def get_local_status(self) -> MachineStatus:
        """Get this machine's status."""
        return MachineStatus(
            machine_id=self.config.machine_id,
            address="localhost",
            port=self.config.shard_port,
            ready=True,
            layers=(self.shard_config.layer_start, self.shard_config.layer_end),
            expert_files_ok=self.verify_expert_files(),
        )

    def wait_for_peer(self, timeout: float = 60.0) -> bool:
        """Wait for peer machine to connect (machine 0 only).

        Returns True if peer connected successfully.
        """
        if not self.shard_config.is_first:
            return self._connect_to_primary(timeout)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(timeout)

        try:
            sock.bind(("0.0.0.0", self.COORD_PORT))
            sock.listen(1)
            print(f"Coordinator: Waiting for peer on port {self.COORD_PORT}...")

            conn, addr = sock.accept()
            print(f"Coordinator: Peer connected from {addr}")

            # Exchange status
            local_status = self.get_local_status()
            status_json = json.dumps({
                "machine_id": local_status.machine_id,
                "layers": local_status.layers,
                "expert_files_ok": local_status.expert_files_ok,
                "ready": local_status.ready,
            })
            conn.sendall(status_json.encode() + b"\n")

            # Receive peer status
            peer_data = b""
            while b"\n" not in peer_data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                peer_data += chunk

            peer_status = json.loads(peer_data.strip())
            self.machines[peer_status["machine_id"]] = MachineStatus(
                machine_id=peer_status["machine_id"],
                address=addr[0],
                port=self.config.shard_port,
                ready=peer_status["ready"],
                layers=tuple(peer_status["layers"]),
                expert_files_ok=peer_status["expert_files_ok"],
            )

            # Send go signal
            conn.sendall(b"GO\n")
            conn.close()
            return True

        except socket.timeout:
            print("Coordinator: Timeout waiting for peer")
            return False
        finally:
            sock.close()

    def _connect_to_primary(self, timeout: float) -> bool:
        """Connect to the primary machine (machine 1 only)."""
        if not self.config.peer_addresses:
            print("Coordinator: No peer address configured")
            return False

        primary_addr = self.config.peer_addresses[0]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        retries = int(timeout / 2)
        for attempt in range(retries):
            try:
                sock.connect((primary_addr, self.COORD_PORT))
                print(f"Coordinator: Connected to primary at {primary_addr}")

                # Receive primary status
                data = b""
                while b"\n" not in data:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    data += chunk

                primary_status = json.loads(data.strip())
                self.machines[primary_status["machine_id"]] = MachineStatus(
                    machine_id=primary_status["machine_id"],
                    address=primary_addr,
                    port=self.config.shard_port,
                    ready=primary_status["ready"],
                    layers=tuple(primary_status["layers"]),
                    expert_files_ok=primary_status["expert_files_ok"],
                )

                # Send our status
                local_status = self.get_local_status()
                status_json = json.dumps({
                    "machine_id": local_status.machine_id,
                    "layers": local_status.layers,
                    "expert_files_ok": local_status.expert_files_ok,
                    "ready": local_status.ready,
                })
                sock.sendall(status_json.encode() + b"\n")

                # Wait for go signal
                go = sock.recv(64)
                sock.close()
                return b"GO" in go

            except (ConnectionRefusedError, socket.timeout):
                if attempt < retries - 1:
                    time.sleep(2)
                continue

        print("Coordinator: Failed to connect to primary")
        return False

    def print_cluster_info(self):
        """Print cluster status."""
        local = self.get_local_status()
        print(f"\n=== 3Kings Cluster ===")
        print(f"Machine {local.machine_id}: layers {local.layers[0]}-{local.layers[1]-1}, "
              f"experts {'OK' if local.expert_files_ok else 'MISSING'}")
        for mid, status in self.machines.items():
            print(f"Machine {mid}: layers {status.layers[0]}-{status.layers[1]-1}, "
                  f"experts {'OK' if status.expert_files_ok else 'MISSING'}, "
                  f"ready={status.ready}")
        print(f"======================\n")
