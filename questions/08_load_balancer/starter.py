"""
Question 08: Load Balancer for Inference

Build a load balancer that routes LLM inference requests to GPU servers
using weighted least-connections routing.

Implement the LoadBalancer class and verify it with the simulation harness.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LARGE_REQUEST_TOKEN_THRESHOLD = 4096
LARGE_REQUEST_MIN_GPU_MEMORY_GB = 80


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServerInfo:
    """Metadata and current state for a registered GPU server."""
    server_id: str
    capacity: int
    gpu_memory_gb: int
    active_requests: set = field(default_factory=set)
    is_draining: bool = False

    @property
    def current_load(self) -> int:
        """Number of currently active requests on this server."""
        return len(self.active_requests)

    @property
    def load_ratio(self) -> float:
        """Current load as a fraction of capacity."""
        return self.current_load / self.capacity

    @property
    def has_capacity(self) -> bool:
        """Whether this server can accept another request."""
        return self.current_load < self.capacity


# ---------------------------------------------------------------------------
# LoadBalancer — implement this
# ---------------------------------------------------------------------------

class LoadBalancer:
    """Routes LLM inference requests to GPU servers using weighted least-connections.

    Routing strategy:
        1. Select servers that are not draining and have available capacity.
        2. For large requests (estimated_tokens > 4096), only consider servers with
           >= 80 GB GPU memory.
        3. Among eligible servers, pick the one with the lowest load_ratio
           (current_load / capacity).
        4. Break ties by choosing the server with the most GPU memory.
    """

    def __init__(self) -> None:
        """Initialize the load balancer with empty server and request pools."""
        pass  # TODO: Initialize your data structures

    def add_server(
        self, server_id: str, capacity: int, gpu_memory_gb: int
    ) -> None:
        """Register a new GPU server with the load balancer.

        Args:
            server_id: Unique identifier for the server.
            capacity: Maximum concurrent requests this server can handle.
            gpu_memory_gb: GPU memory in gigabytes.

        Raises:
            ValueError: If a server with this ID already exists.
        """
        pass  # TODO

    def remove_server(self, server_id: str) -> None:
        """Remove a server from the pool (graceful drain).

        The server stops receiving new requests immediately. Once all in-flight
        requests complete, it is fully removed.

        Args:
            server_id: The server to remove.

        Raises:
            KeyError: If the server does not exist.
        """
        pass  # TODO

    def route_request(self, request_id: str, estimated_tokens: int) -> str:
        """Route an incoming request to the best available server.

        Args:
            request_id: Unique identifier for this request.
            estimated_tokens: Estimated number of tokens for this request.

        Returns:
            The server_id of the server that will handle this request.

        Raises:
            ValueError: If the request ID is already in use.
            RuntimeError: If no server has available capacity for this request.
        """
        pass  # TODO

    def complete_request(self, request_id: str) -> None:
        """Mark a request as completed, freeing server capacity.

        If the server was marked for removal and has no more in-flight requests,
        the server is fully removed.

        Args:
            request_id: The request to complete.

        Raises:
            KeyError: If the request ID is not found.
        """
        pass  # TODO

    def get_server_stats(self) -> dict[str, dict]:
        """Return current load statistics for all servers.

        Returns:
            A dict mapping server_id to a stats dict containing:
                - capacity (int)
                - current_load (int)
                - gpu_memory_gb (int)
                - load_ratio (float)
                - is_draining (bool)
                - active_requests (list[str])
        """
        pass  # TODO


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    lb = LoadBalancer()
    lb.add_server("gpu-01", capacity=4, gpu_memory_gb=80)
    lb.add_server("gpu-02", capacity=4, gpu_memory_gb=80)
    server = lb.route_request("req-1", estimated_tokens=100)
    print(f"Routed to: {server}")
    print(f"Stats: {lb.get_server_stats()}")
