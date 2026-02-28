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


# ---------------------------------------------------------------------------
# Simulation harness
# ---------------------------------------------------------------------------

def run_simulation() -> None:
    """Run a simulation of the load balancer with synthetic traffic.

    This generates a stream of requests with varying token counts, routes them
    through the load balancer, and prints statistics along the way.
    """
    random.seed(42)
    lb = LoadBalancer()

    # Register a heterogeneous server pool
    servers = [
        ("gpu-a100-01", 8, 80),
        ("gpu-a100-02", 8, 80),
        ("gpu-a10-01", 4, 24),
        ("gpu-a10-02", 4, 24),
        ("gpu-h100-01", 12, 80),
    ]
    for sid, cap, mem in servers:
        lb.add_server(sid, cap, mem)

    print("=== Load Balancer Simulation ===\n")
    print(f"Registered {len(servers)} servers.\n")

    # Generate a stream of requests
    active_requests: list[str] = []
    total_routed = 0
    total_completed = 0

    for step in range(1, 51):
        # Randomly complete some existing requests
        if active_requests and random.random() < 0.6:
            num_complete = random.randint(1, min(3, len(active_requests)))
            for _ in range(num_complete):
                req_id = active_requests.pop(random.randint(0, len(active_requests) - 1))
                lb.complete_request(req_id)
                total_completed += 1

        # Generate a new request
        req_id = f"req-{step:04d}"
        estimated_tokens = random.choice([
            random.randint(50, 500),      # short request
            random.randint(500, 2000),     # medium request
            random.randint(2000, 8000),    # long request (may exceed threshold)
        ])

        try:
            server = lb.route_request(req_id, estimated_tokens)
            active_requests.append(req_id)
            total_routed += 1
            if step <= 10 or step % 10 == 0:
                print(
                    f"Step {step:3d}: Routed {req_id} "
                    f"({estimated_tokens:5d} tokens) -> {server}"
                )
        except RuntimeError as e:
            if step <= 10 or step % 10 == 0:
                print(f"Step {step:3d}: REJECTED {req_id} — {e}")

        # At step 25, drain one server
        if step == 25:
            print(f"\n--- Draining gpu-a10-01 at step {step} ---\n")
            lb.remove_server("gpu-a10-01")

    # Print final stats
    print(f"\n=== Final Statistics ===")
    print(f"Total routed:    {total_routed}")
    print(f"Total completed: {total_completed}")
    print(f"Still active:    {len(active_requests)}")
    print(f"\nServer stats:")
    stats = lb.get_server_stats()
    for sid, info in sorted(stats.items()):
        print(
            f"  {sid}: load={info['current_load']}/{info['capacity']} "
            f"({info['load_ratio']:.1%}), "
            f"gpu={info['gpu_memory_gb']}GB, "
            f"draining={info['is_draining']}"
        )


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

def test_basic() -> None:
    """Run basic tests on the LoadBalancer."""
    print("Running basic tests...\n")

    lb = LoadBalancer()

    # Test: Add servers
    lb.add_server("gpu-01", capacity=4, gpu_memory_gb=40)
    lb.add_server("gpu-02", capacity=4, gpu_memory_gb=80)
    lb.add_server("gpu-03", capacity=2, gpu_memory_gb=80)

    # Test: Duplicate server
    try:
        lb.add_server("gpu-01", capacity=4, gpu_memory_gb=40)
        assert False, "Should have raised ValueError for duplicate server"
    except ValueError:
        pass

    # Test: Route a small request — should go to a server with lowest load ratio,
    # tie-broken by GPU memory
    server = lb.route_request("req-1", estimated_tokens=100)
    assert server in ("gpu-02", "gpu-03"), f"Expected gpu-02 or gpu-03, got {server}"
    print(f"  req-1 (small) -> {server}")

    # Test: Route a large request — only 80GB servers eligible
    server = lb.route_request("req-2", estimated_tokens=5000)
    assert server in ("gpu-02", "gpu-03"), f"Expected 80GB server, got {server}"
    print(f"  req-2 (large, >4096 tokens) -> {server}")

    # Test: Complete a request
    lb.complete_request("req-1")
    stats = lb.get_server_stats()
    load_summary = ", ".join(
        f"{s}={stats[s]['current_load']}" for s in sorted(stats)
    )
    print(f"  After completing req-1: loads = {load_summary}")

    # Test: Unknown request
    try:
        lb.complete_request("nonexistent")
        assert False, "Should have raised KeyError for unknown request"
    except KeyError:
        pass

    # Test: Server draining
    lb.add_server("gpu-04", capacity=2, gpu_memory_gb=40)
    lb.route_request("req-3", estimated_tokens=100)  # goes somewhere
    drain_target = "gpu-04"
    lb.route_request("req-drain", estimated_tokens=100)

    # Find which server req-drain went to (for this test, manually route to gpu-04)
    # Instead, let's do a controlled test:
    lb2 = LoadBalancer()
    lb2.add_server("s1", capacity=2, gpu_memory_gb=40)
    lb2.add_server("s2", capacity=2, gpu_memory_gb=40)
    s = lb2.route_request("r1", estimated_tokens=100)
    lb2.remove_server(s)
    stats = lb2.get_server_stats()
    assert stats[s]["is_draining"] is True, "Server should be draining"

    # New request should NOT go to draining server
    s2 = lb2.route_request("r2", estimated_tokens=100)
    assert s2 != s, f"Request should not be routed to draining server {s}"
    print(f"  Draining test: {s} is draining, new request went to {s2}")

    # Complete in-flight request on draining server -> server fully removed
    lb2.complete_request("r1")
    stats = lb2.get_server_stats()
    assert s not in stats, f"Server {s} should have been fully removed after drain"
    print(f"  Server {s} fully removed after drain complete")

    # Test: Remove unknown server
    try:
        lb2.remove_server("nonexistent")
        assert False, "Should have raised KeyError for unknown server"
    except KeyError:
        pass

    # Test: Capacity exhaustion
    lb3 = LoadBalancer()
    lb3.add_server("tiny", capacity=1, gpu_memory_gb=40)
    lb3.route_request("only-req", estimated_tokens=100)
    try:
        lb3.route_request("overflow-req", estimated_tokens=100)
        assert False, "Should have raised RuntimeError for no capacity"
    except RuntimeError:
        pass
    print("  Capacity exhaustion correctly raises RuntimeError")

    print("\nAll basic tests passed!\n")


if __name__ == "__main__":
    test_basic()
    run_simulation()
