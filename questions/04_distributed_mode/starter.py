"""
Question 04: Distributed Mode Finding

Find the mode of a dataset distributed across multiple nodes.
Minimize wall-clock time given that network communication is 10x slower
than local data reads.

See QUESTION.md for full problem description.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict


class Cluster:
    """
    Simulates a cluster of N nodes, each holding a portion of a distributed
    dataset. Tracks simulated wall-clock time for local reads and network sends.

    Timing model:
        - Local read:  10 bytes/sec throughput (fast)
        - Network send: 1 byte/sec throughput (slow, 10x worse than local)

    Usage:
        cluster = Cluster(num_nodes=10, seed=42)
        data = cluster.read_local_data(0)       # read node 0's data
        cluster.send(0, 1, payload_bytes)        # send data from node 0 to 1
        cluster.barrier()                        # synchronize all nodes
        received = cluster.get_received(1)       # get data sent to node 1
        print(cluster.total_time)                # simulated seconds elapsed
    """

    LOCAL_READ_SPEED = 10.0   # bytes per second for local reads
    NETWORK_SPEED = 1.0       # bytes per second for network sends

    def __init__(self, num_nodes: int = 10, total_values: int = 10_000, seed: int = 42) -> None:
        """
        Initialize the cluster with randomly distributed data.

        Args:
            num_nodes: Number of nodes in the cluster.
            total_values: Total number of integers across all nodes.
            seed: Random seed for reproducibility.
        """
        self.num_nodes = num_nodes
        self.total_time: float = 0.0
        self._bytes_read: int = 0
        self._bytes_sent: int = 0
        self._send_count: int = 0

        # Generate data: integers in [0, 1000) distributed across nodes
        rng = random.Random(seed)
        all_data = [rng.randint(0, 999) for _ in range(total_values)]
        rng.shuffle(all_data)

        # Split data across nodes (roughly equal)
        self._node_data: list[list[int]] = [[] for _ in range(num_nodes)]
        for i, val in enumerate(all_data):
            self._node_data[i % num_nodes].append(val)

        # Mailboxes for inter-node communication
        self._mailboxes: dict[int, list[bytes]] = defaultdict(list)

        # Tracking
        self._barrier_count = 0

    def read_local_data(self, node_id: int) -> list[int]:
        """
        Read the data stored locally on the given node.

        Simulated cost: len(data) * 4 bytes at 10 bytes/sec.

        Args:
            node_id: The node to read from (0-indexed).

        Returns:
            A list of integers stored on that node.

        Raises:
            ValueError: If node_id is out of range.
        """
        if not 0 <= node_id < self.num_nodes:
            raise ValueError(f"Invalid node_id {node_id}, must be in [0, {self.num_nodes})")

        data = self._node_data[node_id]
        num_bytes = len(data) * 4  # 4 bytes per int
        time_cost = num_bytes / self.LOCAL_READ_SPEED
        self.total_time += time_cost
        self._bytes_read += num_bytes
        return list(data)  # return a copy

    def send(self, from_node: int, to_node: int, data: bytes) -> None:
        """
        Send a byte payload from one node to another.

        Simulated cost: len(data) bytes at 1 byte/sec.

        Args:
            from_node: Source node ID.
            to_node: Destination node ID.
            data: The byte payload to send.

        Raises:
            ValueError: If node IDs are out of range.
            TypeError: If data is not bytes.
        """
        if not 0 <= from_node < self.num_nodes:
            raise ValueError(f"Invalid from_node {from_node}")
        if not 0 <= to_node < self.num_nodes:
            raise ValueError(f"Invalid to_node {to_node}")
        if not isinstance(data, bytes):
            raise TypeError(f"data must be bytes, got {type(data).__name__}")

        time_cost = len(data) / self.NETWORK_SPEED
        self.total_time += time_cost
        self._bytes_sent += len(data)
        self._send_count += 1
        self._mailboxes[to_node].append(data)

    def get_received(self, node_id: int) -> list[bytes]:
        """
        Retrieve all data that has been sent to the given node.

        This does NOT add to simulated time (data is already 'arrived').

        Args:
            node_id: The receiving node ID.

        Returns:
            A list of byte payloads sent to this node.
        """
        if not 0 <= node_id < self.num_nodes:
            raise ValueError(f"Invalid node_id {node_id}")
        received = self._mailboxes[node_id]
        self._mailboxes[node_id] = []  # clear after reading
        return received

    def barrier(self) -> None:
        """
        Synchronize all nodes. In a real system, this blocks until all nodes
        reach this point. In simulation, it's a no-op for timing but tracks
        that synchronization occurred.
        """
        self._barrier_count += 1

    def stats(self) -> dict:
        """Return performance statistics for the simulation."""
        return {
            "total_simulated_time": round(self.total_time, 2),
            "bytes_read_locally": self._bytes_read,
            "bytes_sent_over_network": self._bytes_sent,
            "network_send_count": self._send_count,
            "barrier_count": self._barrier_count,
        }

    def get_global_mode(self) -> int:
        """
        Compute the actual mode by examining all data directly.
        Used for verification only -- candidates should NOT call this.

        Returns:
            The true mode (smallest value if tied).
        """
        freq: dict[int, int] = defaultdict(int)
        for node_data in self._node_data:
            for val in node_data:
                freq[val] += 1
        max_count = max(freq.values())
        return min(val for val, count in freq.items() if count == max_count)


# ---------------------------------------------------------------------------
# YOUR IMPLEMENTATION
# ---------------------------------------------------------------------------

def find_mode(cluster: Cluster) -> int:
    """
    Find the mode (most frequent value) across all nodes in the cluster.

    You should use the cluster's primitives:
        - cluster.read_local_data(node_id) -> list[int]
        - cluster.send(from_node, to_node, data)
        - cluster.barrier()
        - cluster.get_received(node_id) -> list[bytes]
        - cluster.num_nodes

    Minimize cluster.total_time (simulated wall-clock time).
    If there is a tie for most frequent, return the smallest value.

    Args:
        cluster: A Cluster instance with distributed data.

    Returns:
        The mode of the entire dataset.
    """
    # TODO: Implement this function
    pass


# ---------------------------------------------------------------------------
# TEST HARNESS
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Run test cases to verify the find_mode implementation."""

    print("=" * 60)
    print("Test 1: Default cluster (10 nodes, 10,000 values)")
    print("=" * 60)
    cluster = Cluster(num_nodes=10, total_values=10_000, seed=42)
    expected = cluster.get_global_mode()

    result = find_mode(cluster)
    stats = cluster.stats()

    print(f"  Expected mode: {expected}")
    print(f"  Your result:   {result}")
    print(f"  Correct:       {result == expected}")
    print(f"  Stats:         {stats}")
    print()

    # Naive baseline: sending all raw data to node 0
    naive_network_bytes = 10_000 * 4  # 40,000 bytes
    naive_time = naive_network_bytes / Cluster.NETWORK_SPEED
    print(f"  Naive approach would send {naive_network_bytes} bytes "
          f"({naive_time:.0f}s simulated)")
    if stats["bytes_sent_over_network"] < naive_network_bytes:
        savings = (1 - stats["bytes_sent_over_network"] / naive_network_bytes) * 100
        print(f"  Your approach saves {savings:.1f}% network bandwidth!")
    print()

    print("=" * 60)
    print("Test 2: Small cluster (3 nodes, 15 values)")
    print("=" * 60)
    cluster2 = Cluster(num_nodes=3, total_values=15, seed=99)
    expected2 = cluster2.get_global_mode()

    result2 = find_mode(cluster2)
    stats2 = cluster2.stats()

    print(f"  Expected mode: {expected2}")
    print(f"  Your result:   {result2}")
    print(f"  Correct:       {result2 == expected2}")
    print(f"  Stats:         {stats2}")
    print()

    print("=" * 60)
    print("Test 3: Single node")
    print("=" * 60)
    cluster3 = Cluster(num_nodes=1, total_values=100, seed=7)
    expected3 = cluster3.get_global_mode()

    result3 = find_mode(cluster3)
    stats3 = cluster3.stats()

    print(f"  Expected mode: {expected3}")
    print(f"  Your result:   {result3}")
    print(f"  Correct:       {result3 == expected3}")
    print(f"  Network bytes: {stats3['bytes_sent_over_network']} (should be 0)")
    print()

    print("=" * 60)
    print("Test 4: All same value")
    print("=" * 60)
    cluster4 = Cluster(num_nodes=5, total_values=50, seed=0)
    # Override data to be all the same value
    for i in range(cluster4.num_nodes):
        cluster4._node_data[i] = [42] * len(cluster4._node_data[i])
    expected4 = 42

    result4 = find_mode(cluster4)
    print(f"  Expected mode: {expected4}")
    print(f"  Your result:   {result4}")
    print(f"  Correct:       {result4 == expected4}")
    print()

    # Summary
    all_correct = (
        result == expected
        and result2 == expected2
        and result3 == expected3
        and result4 == expected4
    )
    print("=" * 60)
    print(f"ALL TESTS PASSED: {all_correct}")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
