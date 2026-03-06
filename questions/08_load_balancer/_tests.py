"""Hidden tests for Question 08: Load Balancer
Run: python questions/08_load_balancer/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import random
from starter import LoadBalancer, ServerInfo


def test_basic():
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

    # Test: Route a small request
    server = lb.route_request("req-1", estimated_tokens=100)
    assert server in ("gpu-02", "gpu-03"), f"Expected gpu-02 or gpu-03, got {server}"

    # Test: Route a large request — only 80GB servers eligible
    server = lb.route_request("req-2", estimated_tokens=5000)
    assert server in ("gpu-02", "gpu-03"), f"Expected 80GB server, got {server}"

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

    print("  [PASS] basic routing and completion\n")


def test_draining():
    """Test server draining behavior."""
    lb = LoadBalancer()
    lb.add_server("s1", capacity=2, gpu_memory_gb=40)
    lb.add_server("s2", capacity=2, gpu_memory_gb=40)
    s = lb.route_request("r1", estimated_tokens=100)
    lb.remove_server(s)
    stats = lb.get_server_stats()
    assert stats[s]["is_draining"] is True, "Server should be draining"

    # New request should NOT go to draining server
    s2 = lb.route_request("r2", estimated_tokens=100)
    assert s2 != s, f"Request should not be routed to draining server {s}"

    # Complete in-flight request on draining server -> server fully removed
    lb.complete_request("r1")
    stats = lb.get_server_stats()
    assert s not in stats, f"Server {s} should have been fully removed after drain"
    print("  [PASS] draining\n")


def test_capacity_exhaustion():
    """Test that capacity exhaustion raises RuntimeError."""
    lb = LoadBalancer()
    lb.add_server("tiny", capacity=1, gpu_memory_gb=40)
    lb.route_request("only-req", estimated_tokens=100)
    try:
        lb.route_request("overflow-req", estimated_tokens=100)
        assert False, "Should have raised RuntimeError for no capacity"
    except RuntimeError:
        pass
    print("  [PASS] capacity exhaustion\n")


def test_remove_unknown_server():
    """Test removing a server that doesn't exist."""
    lb = LoadBalancer()
    lb.add_server("s1", capacity=2, gpu_memory_gb=40)
    try:
        lb.remove_server("nonexistent")
        assert False, "Should have raised KeyError for unknown server"
    except KeyError:
        pass
    print("  [PASS] remove unknown server\n")


def test_simulation():
    """Run a simulation of the load balancer with synthetic traffic."""
    random.seed(42)
    lb = LoadBalancer()

    servers = [
        ("gpu-a100-01", 8, 80),
        ("gpu-a100-02", 8, 80),
        ("gpu-a10-01", 4, 24),
        ("gpu-a10-02", 4, 24),
        ("gpu-h100-01", 12, 80),
    ]
    for sid, cap, mem in servers:
        lb.add_server(sid, cap, mem)

    active_requests = []
    total_routed = 0
    total_completed = 0

    for step in range(1, 51):
        if active_requests and random.random() < 0.6:
            num_complete = random.randint(1, min(3, len(active_requests)))
            for _ in range(num_complete):
                req_id = active_requests.pop(random.randint(0, len(active_requests) - 1))
                lb.complete_request(req_id)
                total_completed += 1

        req_id = f"req-{step:04d}"
        estimated_tokens = random.choice([
            random.randint(50, 500),
            random.randint(500, 2000),
            random.randint(2000, 8000),
        ])

        try:
            lb.route_request(req_id, estimated_tokens)
            active_requests.append(req_id)
            total_routed += 1
        except RuntimeError:
            pass

        if step == 25:
            lb.remove_server("gpu-a10-01")

    assert total_routed > 0, "Should have routed at least some requests"
    stats = lb.get_server_stats()
    assert len(stats) > 0, "Should have at least some servers"
    print(f"  [PASS] simulation (routed {total_routed}, completed {total_completed})\n")


def run_tests():
    print("=" * 60)
    print("Load Balancer — Hidden Tests")
    print("=" * 60 + "\n")

    test_basic()
    test_draining()
    test_capacity_exhaustion()
    test_remove_unknown_server()
    test_simulation()

    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
