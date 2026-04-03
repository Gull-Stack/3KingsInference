"""Network activation passing between sharded machines.

Simple TCP-based activation transfer. For Apple Silicon Macs
connected via Thunderbolt bridge (~40Gbps) or 10GbE (~10Gbps),
activation tensors are small enough that network is not the bottleneck.

Per-token activation size: hidden_dim × sizeof(bfloat16) = 4096 × 2 = 8KB.
At 40Gbps (Thunderbolt): 8KB takes ~1.6 microseconds.
At 10Gbps (Ethernet): 8KB takes ~6.4 microseconds.

Compare to SSD expert read: 6.75MB takes ~400 microseconds at 17GB/s.
Network is 60-250x faster than SSD for activation passing.
"""

import socket
import struct
from typing import Optional

import mlx.core as mx
import numpy as np


class ActivationChannel:
    """Bidirectional activation passing between two machines.

    Machine A processes layers 0-29, sends activations to Machine B.
    Machine B processes layers 30-59, sends final output back to A.

    Uses raw TCP sockets for minimal overhead.
    """

    HEADER_SIZE = 16  # 4 ints: ndim, shape[0], shape[1], shape[2]

    def __init__(self, is_server: bool, peer_address: str, port: int = 5555):
        self.is_server = is_server
        self.peer_address = peer_address
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None

    def connect(self):
        """Establish connection with peer."""
        if self.is_server:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Disable Nagle's algorithm for low latency
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket.bind(("0.0.0.0", self.port))
            self._socket.listen(1)
            print(f"Odin: Waiting for peer on port {self.port}...")
            self._conn, addr = self._socket.accept()
            self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Odin: Peer connected from {addr}")
        else:
            self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Odin: Connecting to {self.peer_address}:{self.port}...")
            self._conn.connect((self.peer_address, self.port))
            print(f"Odin: Connected to peer")

    def send_activation(self, tensor: mx.array):
        """Send activation tensor to peer.

        Converts to numpy for serialization (MLX doesn't have
        built-in serialization for raw socket transfer).
        """
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        # Convert to numpy for serialization
        # MLX bfloat16 can't convert directly to numpy — cast to float32 first
        if tensor.dtype == mx.bfloat16:
            tensor = tensor.astype(mx.float32)
        data = np.array(tensor, dtype=np.float32)
        shape = data.shape

        # Send header: ndim + shape
        header = struct.pack("I" * (1 + len(shape)), len(shape), *shape)
        self._conn.sendall(header)

        # Send data
        self._conn.sendall(data.tobytes())

    def recv_activation(self) -> mx.array:
        """Receive activation tensor from peer."""
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        # Read ndim
        ndim_data = self._recv_exact(4)
        ndim = struct.unpack("I", ndim_data)[0]

        # Read shape
        shape_data = self._recv_exact(4 * ndim)
        shape = struct.unpack("I" * ndim, shape_data)

        # Read data
        n_bytes = int(np.prod(shape)) * 4  # float32
        raw = self._recv_exact(n_bytes)

        # Convert to MLX
        data = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        return mx.array(data)

    def _recv_exact(self, n_bytes: int) -> bytes:
        """Receive exactly n_bytes from socket."""
        chunks = []
        received = 0
        while received < n_bytes:
            chunk = self._conn.recv(min(n_bytes - received, 65536))
            if not chunk:
                raise ConnectionError("Peer disconnected")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def close(self):
        """Close connection."""
        if self._conn:
            self._conn.close()
        if self._socket:
            self._socket.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
