"""
Question 15: Collective Communication Simulator

Simulate distributed GPU collective communication operations: ring allreduce,
allgather, reduce-scatter, data-parallel gradient sync, and pipeline
parallelism with micro-batching.

Run this file directly to execute the test cases:
    python starter.py
"""

from __future__ import annotations

import copy
import math
from typing import Callable


# --------------------------------------------------------------------------- #
#  Collective Operations                                                      #
# --------------------------------------------------------------------------- #

def ring_allreduce(
    gpu_data: list[list[float]], op: str = "sum"
) -> tuple[list[list[float]], dict]:
    """Perform allreduce using the ring algorithm.

    Every GPU starts with a data vector of the same length. After the
    operation, every GPU holds the element-wise reduction of all vectors.

    The ring algorithm proceeds in two phases:
      Phase 1 (Reduce-Scatter): N-1 steps. Data is divided into N chunks.
        In each step, GPU i sends chunk c to GPU (i+1)%N and receives chunk
        c' from GPU (i-1)%N, accumulating the received chunk with its own.
        After this phase, GPU i holds the fully reduced chunk i.
      Phase 2 (AllGather): N-1 steps. GPUs propagate reduced chunks around
        the ring until every GPU has all N reduced chunks.

    Args:
        gpu_data: List of N data vectors, one per GPU. All have the same length.
        op: Reduction operation - "sum", "max", or "avg".

    Returns:
        Tuple of:
        - Final state: list of N vectors (all identical after allreduce).
        - Stats dict with:
          "steps": total number of ring steps (2 * (N-1)),
          "bytes_transferred_per_gpu": total bytes sent by one GPU (floats * 4).
    """
    # TODO: Implement this function
    pass


def ring_allgather(
    gpu_data: list[list[float]],
) -> tuple[list[list[float]], dict]:
    """Perform allgather using the ring algorithm.

    Each GPU starts with a local chunk. After N-1 ring steps, every GPU
    has the concatenation of all chunks in GPU order.

    In each step, every GPU sends its most recently received chunk to its
    right neighbor and receives a new chunk from its left neighbor.

    Args:
        gpu_data: List of N chunks, one per GPU. May have different lengths.

    Returns:
        Tuple of:
        - Final state: list of N vectors, each being the concatenation of
          all original chunks [chunk_0 + chunk_1 + ... + chunk_{N-1}].
        - Stats dict with:
          "steps": N-1,
          "bytes_transferred_per_gpu": total bytes sent by one GPU.
    """
    # TODO: Implement this function
    pass


def ring_reduce_scatter(
    gpu_data: list[list[float]], op: str = "sum"
) -> tuple[list[list[float]], dict]:
    """Perform reduce-scatter using the ring algorithm.

    Every GPU starts with a full data vector. After the operation, GPU i
    holds only the i-th chunk of the element-wise reduction.

    This is Phase 1 of allreduce (the reduce-scatter phase), running
    standalone.

    Args:
        gpu_data: List of N data vectors, all the same length.
        op: Reduction operation - "sum", "max", or "avg".

    Returns:
        Tuple of:
        - Final state: GPU i holds only its reduced chunk (as a list[float]).
          Return format: list of N lists, where gpu[i] contains only chunk i.
        - Stats dict with:
          "steps": N-1,
          "bytes_transferred_per_gpu": total bytes sent by one GPU.
    """
    # TODO: Implement this function
    pass


def simulate_data_parallel_step(
    gradients_per_gpu: list[list[float]], num_gpus: int
) -> tuple[list[list[float]], dict]:
    """Simulate one gradient synchronization step in data-parallel training.

    Each GPU has independently computed gradients. This function uses
    allreduce to average gradients across all GPUs.

    Args:
        gradients_per_gpu: List of N gradient vectors, one per GPU.
        num_gpus: Number of GPUs (must equal len(gradients_per_gpu)).

    Returns:
        Tuple of:
        - Averaged gradients: list of N identical vectors.
        - Communication stats from the allreduce.
    """
    # TODO: Implement this function
    pass


def simulate_pipeline_parallel(
    stages: list[Callable],
    data: list[list[float]],
    num_microbatches: int,
) -> tuple[list[list[float]], dict]:
    """Simulate pipeline parallelism with micro-batching.

    The model is split into S stages (one per GPU). Input data is split
    into M micro-batches. Execution follows a pipeline schedule:

    Timeline (S=3 stages, M=4 microbatches):
        Step:  0    1    2    3    4    5
        GPU 0: mb0  mb1  mb2  mb3  ---  ---
        GPU 1: ---  mb0  mb1  mb2  mb3  ---
        GPU 2: ---  ---  mb0  mb1  mb2  mb3

    Total steps = S + M - 1
    Active slots = S * M (each micro-batch passes through each stage)
    Total slots = S * total_steps
    Bubble slots = Total slots - Active slots
    Bubble ratio = Bubble slots / Total slots

    Args:
        stages: List of S callable functions, one per stage.
            Each takes list[float] and returns list[float].
        data: List of input samples (list of float vectors).
        num_microbatches: Number of micro-batches to split data into.

    Returns:
        Tuple of:
        - Output for each input sample, in original order.
        - Stats dict:
          "total_steps": S + M - 1,
          "bubble_ratio": fraction of GPU-time wasted in bubbles,
          "num_stages": S,
          "num_microbatches": M.
    """
    # TODO: Implement this function
    pass


# --------------------------------------------------------------------------- #
#  Helper Utilities                                                           #
# --------------------------------------------------------------------------- #

def elementwise_op(
    a: list[float], b: list[float], op: str
) -> list[float]:
    """Apply an element-wise binary operation on two vectors.

    Args:
        a: First vector.
        b: Second vector (same length as a).
        op: One of "sum", "max", "avg" (for avg, this just sums; divide later).

    Returns:
        Result vector.
    """
    if op == "sum" or op == "avg":
        return [x + y for x, y in zip(a, b)]
    elif op == "max":
        return [max(x, y) for x, y in zip(a, b)]
    else:
        raise ValueError(f"Unknown op: {op}")


def naive_allreduce(
    gpu_data: list[list[float]], op: str = "sum"
) -> list[float]:
    """Reference allreduce implementation (not ring-based).

    Simply reduces all vectors element-wise. Used to verify ring results.

    Args:
        gpu_data: List of N equal-length vectors.
        op: Reduction operation.

    Returns:
        Single reduced vector.
    """
    n = len(gpu_data)
    vec_len = len(gpu_data[0])
    result = [0.0] * vec_len

    if op == "sum" or op == "avg":
        for i in range(vec_len):
            result[i] = sum(gpu_data[g][i] for g in range(n))
        if op == "avg":
            result = [x / n for x in result]
    elif op == "max":
        for i in range(vec_len):
            result[i] = max(gpu_data[g][i] for g in range(n))

    return result


def vectors_equal(
    a: list[float], b: list[float], tol: float = 1e-9
) -> bool:
    """Check if two float vectors are equal within tolerance."""
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def make_sample_stages() -> list[Callable]:
    """Create sample pipeline stage functions for testing.

    Returns a list of 3 stages:
    - Stage 0: multiply by 2
    - Stage 1: add 1
    - Stage 2: multiply by 3

    So the pipeline computes: ((x * 2) + 1) * 3 = 6x + 3
    """
    return [
        lambda x: [v * 2 for v in x],
        lambda x: [v + 1 for v in x],
        lambda x: [v * 3 for v in x],
    ]


# --------------------------------------------------------------------------- #
#  Test Cases                                                                 #
# --------------------------------------------------------------------------- #

def test_ring_allreduce_sum() -> None:
    """Test ring allreduce with sum operation."""
    gpu_data = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ]

    expected = naive_allreduce(gpu_data, op="sum")
    result, stats = ring_allreduce(gpu_data, op="sum")

    # All GPUs should have the same result
    for i, gpu_result in enumerate(result):
        assert vectors_equal(gpu_result, expected), (
            f"GPU {i}: expected {expected}, got {gpu_result}"
        )

    # Verify stats
    n = len(gpu_data)
    assert stats["steps"] == 2 * (n - 1), (
        f"Expected {2*(n-1)} steps, got {stats['steps']}"
    )

    print("  [PASS] test_ring_allreduce_sum")


def test_ring_allreduce_max() -> None:
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


def test_ring_allreduce_avg() -> None:
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


def test_ring_allreduce_single_gpu() -> None:
    """Test allreduce with a single GPU (edge case)."""
    gpu_data = [[1.0, 2.0, 3.0]]
    result, stats = ring_allreduce(gpu_data, op="sum")

    assert vectors_equal(result[0], [1.0, 2.0, 3.0]), (
        f"Single GPU should return unchanged data, got {result[0]}"
    )
    assert stats["steps"] == 0, f"Single GPU: expected 0 steps, got {stats['steps']}"

    print("  [PASS] test_ring_allreduce_single_gpu")


def test_ring_allreduce_non_divisible() -> None:
    """Test allreduce when data length is not divisible by num GPUs."""
    # 3 GPUs, data length 7 (not divisible by 3)
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


def test_ring_allgather() -> None:
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


def test_ring_allgather_unequal_chunks() -> None:
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


def test_ring_reduce_scatter() -> None:
    """Test ring reduce-scatter."""
    gpu_data = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]

    # Full reduction: [8, 8, 8, 8, 8, 8]
    # Chunk 0 (indices 0-1): [8, 8] -> GPU 0
    # Chunk 1 (indices 2-3): [8, 8] -> GPU 1
    # Chunk 2 (indices 4-5): [8, 8] -> GPU 2

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


def test_data_parallel_step() -> None:
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


def test_pipeline_parallel_basic() -> None:
    """Test pipeline parallelism with simple stages."""
    stages = make_sample_stages()  # *2, +1, *3 -> result = (2x + 1) * 3 = 6x + 3

    data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    result, stats = simulate_pipeline_parallel(stages, data, num_microbatches=3)

    # Expected: each x -> 6x + 3
    expected = [[6.0 * x[0] + 3.0] for x in data]

    assert len(result) == len(data), f"Expected {len(data)} results, got {len(result)}"
    for i, (res, exp) in enumerate(zip(result, expected)):
        assert vectors_equal(res, exp, tol=1e-6), (
            f"Sample {i}: expected {exp}, got {res}"
        )

    # Stats checks
    S = len(stages)
    M = 3
    assert stats["total_steps"] == S + M - 1, (
        f"Expected {S + M - 1} total steps, got {stats['total_steps']}"
    )
    assert stats["num_stages"] == S
    assert stats["num_microbatches"] == M

    expected_bubble = (S - 1) / (S * (S + M - 1))
    assert abs(stats["bubble_ratio"] - expected_bubble) < 1e-6, (
        f"Expected bubble_ratio {expected_bubble:.4f}, got {stats['bubble_ratio']:.4f}"
    )

    print(f"    Total steps: {stats['total_steps']}")
    print(f"    Bubble ratio: {stats['bubble_ratio']:.4f}")
    print("  [PASS] test_pipeline_parallel_basic")


def test_pipeline_parallel_single_stage() -> None:
    """Test pipeline with a single stage (no communication)."""
    stages = [lambda x: [v ** 2 for v in x]]
    data = [[1.0, 2.0], [3.0, 4.0]]

    result, stats = simulate_pipeline_parallel(stages, data, num_microbatches=2)

    expected = [[1.0, 4.0], [9.0, 16.0]]
    for i, (res, exp) in enumerate(zip(result, expected)):
        assert vectors_equal(res, exp, tol=1e-6), (
            f"Sample {i}: expected {exp}, got {res}"
        )

    assert stats["total_steps"] == 2  # 1 + 2 - 1 = 2
    assert stats["bubble_ratio"] == 0.0, (
        f"Single stage should have 0 bubbles, got {stats['bubble_ratio']}"
    )

    print("  [PASS] test_pipeline_parallel_single_stage")


def test_pipeline_parallel_many_microbatches() -> None:
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

    assert stats_6["bubble_ratio"] < stats_3["bubble_ratio"], (
        "More microbatches should reduce bubble ratio"
    )
    assert stats_12["bubble_ratio"] < stats_6["bubble_ratio"], (
        "More microbatches should reduce bubble ratio"
    )

    print("  [PASS] test_pipeline_parallel_many_microbatches")


def test_allreduce_large_scale() -> None:
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


def run_all_tests() -> None:
    """Run all test cases."""
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
    run_all_tests()
