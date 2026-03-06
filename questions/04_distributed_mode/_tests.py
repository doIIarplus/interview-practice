"""Hidden tests for Question 04: Distributed Mode Finding
Run: python questions/04_distributed_mode/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import Cluster, find_mode


def test_default_cluster():
    """Test with default cluster (10 nodes, 10,000 values)."""
    cluster = Cluster(num_nodes=10, total_values=10_000, seed=42)
    expected = cluster.get_global_mode()
    result = find_mode(cluster)
    stats = cluster.stats()

    assert result == expected, f"Expected mode {expected}, got {result}"

    # Check that solution is more efficient than naive
    naive_network_bytes = 10_000 * 4
    assert stats["bytes_sent_over_network"] < naive_network_bytes, (
        f"Solution sent {stats['bytes_sent_over_network']} bytes, "
        f"naive would send {naive_network_bytes}"
    )
    print("[PASS] test_default_cluster")


def test_small_cluster():
    """Test with small cluster (3 nodes, 15 values)."""
    cluster = Cluster(num_nodes=3, total_values=15, seed=99)
    expected = cluster.get_global_mode()
    result = find_mode(cluster)
    assert result == expected, f"Expected mode {expected}, got {result}"
    print("[PASS] test_small_cluster")


def test_single_node():
    """Test with single node (no network needed)."""
    cluster = Cluster(num_nodes=1, total_values=100, seed=7)
    expected = cluster.get_global_mode()
    result = find_mode(cluster)
    stats = cluster.stats()
    assert result == expected, f"Expected mode {expected}, got {result}"
    assert stats["bytes_sent_over_network"] == 0, "Single node should send 0 bytes"
    print("[PASS] test_single_node")


def test_all_same_value():
    """Test when all values are identical."""
    cluster = Cluster(num_nodes=5, total_values=50, seed=0)
    for i in range(cluster.num_nodes):
        cluster._node_data[i] = [42] * len(cluster._node_data[i])
    result = find_mode(cluster)
    assert result == 42, f"Expected mode 42, got {result}"
    print("[PASS] test_all_same_value")


def run_tests():
    print("=" * 60)
    print("Distributed Mode Finding — Hidden Tests")
    print("=" * 60 + "\n")

    test_default_cluster()
    test_small_cluster()
    test_single_node()
    test_all_same_value()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
