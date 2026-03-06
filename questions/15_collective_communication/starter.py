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


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    gpu_data = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    result, stats = ring_allreduce(gpu_data, op="sum")
    print(f"Allreduce result: {result}")
    print(f"Stats: {stats}")
