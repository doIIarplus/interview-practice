"""Hidden tests for Question 15: Collective Communication Simulator
Run: python questions/15_collective_communication/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import (
    ring_allreduce, ring_allgather, ring_reduce_scatter,
    simulate_data_parallel_step, simulate_pipeline_parallel,
    naive_allreduce, vectors_equal, make_sample_stages,
)


def test_ring_allreduce_sum():
    """Test ring allreduce with sum operation."""
    gpu_data = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ]
    expected = naive_allreduce(gpu_data, op="sum")
    result, stats = ring_allreduce(gpu_data, op="sum")
    for i, gpu_result in enumerate(result):
        assert vectors_equal(gpu_result, expected), (
            f"GPU {i}: expected {expected}, got {gpu_result}"
        )
    n = len(gpu_data)
    assert stats["steps"] == 2 * (n - 1), f"Expected {2*(n-1)} steps, got {stats['steps']}"
    print("  [PASS] test_ring_allreduce_sum")


def test_ring_allreduce_max():
    """Test ring allreduce with max operation."""
    gpu_data = [
        [1.0, 5.0, 2.0, 8.0],
        [3.0, 1.0, 7.0, 2.0],
        [2.0, 6.0, 1.0, 9.0],
    ]
    expected = naive_allreduce(gpu_data, op="max")
    result, stats = ring_allreduce(gpu_data, op="max")
    for i, gpu_result in enumerate(result):
        assert vectors_equal(gpu_result, expected), (
            f"GPU {i}: expected {expected}, got {gpu_result}"
        )
    print("  [PASS] test_ring_allreduce_max")


def test_ring_allreduce_avg():
    """Test ring allreduce with average operation."""
    gpu_data = [
        [4.0, 8.0],
        [2.0, 6.0],
        [6.0, 4.0],
        [0.0, 2.0],
    ]
    expected = naive_allreduce(gpu_data, op="avg")
    result, stats = ring_allreduce(gpu_data, op="avg")
    for i, gpu_result in enumerate(result):
        assert vectors_equal(gpu_result, expected), (
            f"GPU {i}: expected {expected}, got {gpu_result}"
        )
    print("  [PASS] test_ring_allreduce_avg")


def test_ring_allreduce_single_gpu():
    """Test allreduce with a single GPU (edge case)."""
    gpu_data = [[1.0, 2.0, 3.0]]
    result, stats = ring_allreduce(gpu_data, op="sum")
    assert vectors_equal(result[0], [1.0, 2.0, 3.0]), (
        f"Single GPU should return unchanged data, got {result[0]}"
    )
    assert stats["steps"] == 0, f"Single GPU: expected 0 steps, got {stats['steps']}"
    print("  [PASS] test_ring_allreduce_single_gpu")


def test_ring_allreduce_non_divisible():
    """Test allreduce when data length is not divisible by num GPUs."""
    gpu_data = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    expected = naive_allreduce(gpu_data, op="sum")
    result, stats = ring_allreduce(gpu_data, op="sum")
    for i, gpu_result in enumerate(result):
        assert vectors_equal(gpu_result, expected), (
            f"GPU {i}: expected {expected}, got {gpu_result}"
        )
    print("  [PASS] test_ring_allreduce_non_divisible")


def test_ring_allgather():
    """Test ring allgather."""
    gpu_data = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
    expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    result, stats = ring_allgather(gpu_data)
    for i, gpu_result in enumerate(result):
        assert vectors_equal(gpu_result, expected), (
            f"GPU {i}: expected {expected}, got {gpu_result}"
        )
    assert stats["steps"] == 2, f"Expected 2 steps, got {stats['steps']}"
    print("  [PASS] test_ring_allgather")


def test_ring_allgather_unequal_chunks():
    """Test allgather with unequal chunk sizes."""
    gpu_data = [
        [1.0],
        [2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    result, stats = ring_allgather(gpu_data)
    for i, gpu_result in enumerate(result):
        assert vectors_equal(gpu_result, expected), (
            f"GPU {i}: expected {expected}, got {gpu_result}"
        )
    print("  [PASS] test_ring_allgather_unequal_chunks")


def test_ring_reduce_scatter():
    """Test ring reduce-scatter."""
    gpu_data = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]
    full_reduced = naive_allreduce(gpu_data, op="sum")
    result, stats = ring_reduce_scatter(gpu_data, op="sum")
    n = len(gpu_data)
    chunk_size = len(gpu_data[0]) // n
    for i in range(n):
        expected_chunk = full_reduced[i * chunk_size : (i + 1) * chunk_size]
        assert vectors_equal(result[i], expected_chunk), (
            f"GPU {i}: expected chunk {expected_chunk}, got {result[i]}"
        )
    assert stats["steps"] == n - 1
    print("  [PASS] test_ring_reduce_scatter")


def test_data_parallel_step():
    """Test data parallel gradient synchronization."""
    gradients = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5, 1.6],
    ]
    result, stats = simulate_data_parallel_step(gradients, num_gpus=4)
    expected = naive_allreduce(gradients, op="avg")
    for i, gpu_grads in enumerate(result):
        assert vectors_equal(gpu_grads, expected, tol=1e-6), (
            f"GPU {i}: expected {expected}, got {gpu_grads}"
        )
    print("  [PASS] test_data_parallel_step")


def test_pipeline_parallel_basic():
    """Test pipeline parallelism with simple stages."""
    stages = make_sample_stages()
    data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    result, stats = simulate_pipeline_parallel(stages, data, num_microbatches=3)
    expected = [[6.0 * x[0] + 3.0] for x in data]
    assert len(result) == len(data), f"Expected {len(data)} results, got {len(result)}"
    for i, (res, exp) in enumerate(zip(result, expected)):
        assert vectors_equal(res, exp, tol=1e-6), (
            f"Sample {i}: expected {exp}, got {res}"
        )
    S = len(stages)
    M = 3
    assert stats["total_steps"] == S + M - 1
    assert stats["num_stages"] == S
    assert stats["num_microbatches"] == M
    expected_bubble = (S - 1) / (S * (S + M - 1))
    assert abs(stats["bubble_ratio"] - expected_bubble) < 1e-6
    print(f"    Total steps: {stats['total_steps']}")
    print(f"    Bubble ratio: {stats['bubble_ratio']:.4f}")
    print("  [PASS] test_pipeline_parallel_basic")


def test_pipeline_parallel_single_stage():
    """Test pipeline with a single stage."""
    stages = [lambda x: [v ** 2 for v in x]]
    data = [[1.0, 2.0], [3.0, 4.0]]
    result, stats = simulate_pipeline_parallel(stages, data, num_microbatches=2)
    expected = [[1.0, 4.0], [9.0, 16.0]]
    for i, (res, exp) in enumerate(zip(result, expected)):
        assert vectors_equal(res, exp, tol=1e-6), (
            f"Sample {i}: expected {exp}, got {res}"
        )
    assert stats["total_steps"] == 2
    assert stats["bubble_ratio"] == 0.0
    print("  [PASS] test_pipeline_parallel_single_stage")


def test_pipeline_parallel_many_microbatches():
    """Test that more microbatches reduce bubble ratio."""
    stages = [
        lambda x: [v + 1 for v in x],
        lambda x: [v * 2 for v in x],
        lambda x: [v - 0.5 for v in x],
        lambda x: [v / 3 for v in x],
    ]
    data = [[float(i)] for i in range(12)]
    _, stats_3 = simulate_pipeline_parallel(stages, data, num_microbatches=3)
    _, stats_6 = simulate_pipeline_parallel(stages, data, num_microbatches=6)
    _, stats_12 = simulate_pipeline_parallel(stages, data, num_microbatches=12)
    print(f"    Bubble ratio (M=3):  {stats_3['bubble_ratio']:.4f}")
    print(f"    Bubble ratio (M=6):  {stats_6['bubble_ratio']:.4f}")
    print(f"    Bubble ratio (M=12): {stats_12['bubble_ratio']:.4f}")
    assert stats_6["bubble_ratio"] < stats_3["bubble_ratio"]
    assert stats_12["bubble_ratio"] < stats_6["bubble_ratio"]
    print("  [PASS] test_pipeline_parallel_many_microbatches")


def test_allreduce_large_scale():
    """Test allreduce with more GPUs and larger data."""
    import random
    rng = random.Random(42)
    num_gpus = 8
    vec_len = 1024
    gpu_data = [[rng.gauss(0, 1) for _ in range(vec_len)] for _ in range(num_gpus)]
    expected = naive_allreduce(gpu_data, op="sum")
    result, stats = ring_allreduce(gpu_data, op="sum")
    for i in range(num_gpus):
        assert vectors_equal(result[i], expected, tol=1e-6), (
            f"GPU {i} mismatch in large-scale allreduce"
        )
    assert stats["steps"] == 2 * (num_gpus - 1)
    print(f"    8 GPUs, 1024-element vectors: {stats['steps']} steps")
    print(f"    Bytes transferred per GPU: {stats['bytes_transferred_per_gpu']}")
    print("  [PASS] test_allreduce_large_scale")


def run_tests():
    print("Running Collective Communication Simulator tests...\n")
    test_ring_allreduce_sum()
    test_ring_allreduce_max()
    test_ring_allreduce_avg()
    test_ring_allreduce_single_gpu()
    test_ring_allreduce_non_divisible()
    test_ring_allgather()
    test_ring_allgather_unequal_chunks()
    test_ring_reduce_scatter()
    test_data_parallel_step()
    test_pipeline_parallel_basic()
    test_pipeline_parallel_single_stage()
    test_pipeline_parallel_many_microbatches()
    test_allreduce_large_scale()
    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
