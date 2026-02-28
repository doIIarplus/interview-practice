"""
Question 07: Matrix Tiling Optimizer

Optimize matrix multiplication through cache-friendly tiling (blocking).

Matrices are represented as list[list[float]] — no NumPy allowed.
Implement tiled matrix multiplication and benchmark it against the naive approach.
"""

import time
import random
from typing import Optional


def generate_matrix(n: int, seed: Optional[int] = None) -> list[list[float]]:
    """Generate an NxN matrix filled with random floats in [0, 1).

    Args:
        n: Matrix dimension (creates an n x n matrix).
        seed: Optional random seed for reproducibility.

    Returns:
        An n x n matrix as a list of lists of floats.
    """
    if seed is not None:
        random.seed(seed)
    return [[random.random() for _ in range(n)] for _ in range(n)]


def matrices_equal(
    A: list[list[float]], B: list[list[float]], tol: float = 1e-9
) -> bool:
    """Check if two matrices are equal within a floating-point tolerance.

    Args:
        A: First matrix.
        B: Second matrix.
        tol: Maximum allowed absolute difference per element.

    Returns:
        True if all corresponding elements differ by at most `tol`.
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False
    for i in range(len(A)):
        for j in range(len(A[0])):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


def print_matrix(M: list[list[float]], label: str = "Matrix") -> None:
    """Pretty-print a small matrix for debugging.

    Args:
        M: The matrix to print.
        label: A label to print above the matrix.
    """
    print(f"\n{label} ({len(M)}x{len(M[0])}):")
    for row in M:
        print("  [" + ", ".join(f"{x:8.4f}" for x in row) + "]")


def time_function(fn, *args, **kwargs) -> tuple[float, any]:
    """Time the execution of a function and return (elapsed_seconds, result).

    Args:
        fn: The function to time.
        *args: Positional arguments passed to fn.
        **kwargs: Keyword arguments passed to fn.

    Returns:
        A tuple of (elapsed_time_in_seconds, function_return_value).
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


# ---------------------------------------------------------------------------
# Naive matrix multiplication (provided as baseline)
# ---------------------------------------------------------------------------

def matmul_naive(
    A: list[list[float]], B: list[list[float]]
) -> list[list[float]]:
    """Naive O(N^3) matrix multiplication.

    Computes C = A * B using the standard triple-loop algorithm.

    This is your baseline. It accesses B column-by-column in the innermost
    loop, which causes poor cache utilization because the matrix is stored
    in row-major order.

    Args:
        A: An N x N matrix (list of lists).
        B: An N x N matrix (list of lists).

    Returns:
        The N x N product matrix C = A * B.
    """
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


# ---------------------------------------------------------------------------
# TODO: Implement tiled matrix multiplication
# ---------------------------------------------------------------------------

def matmul_tiled(
    A: list[list[float]],
    B: list[list[float]],
    tile_size: int = 32,
) -> list[list[float]]:
    """Tiled (blocked) matrix multiplication for better cache performance.

    Break the computation into tiles of size `tile_size x tile_size` so that
    the working set fits in CPU cache. This should produce results identical
    to matmul_naive within floating-point tolerance.

    Hint: You need six nested loops:
        - Three outer loops (ii, jj, kk) that iterate over tile boundaries
          in steps of tile_size.
        - Three inner loops (i, j, k) that iterate within each tile.
        - Use min() to handle the case where N is not divisible by tile_size.

    Args:
        A: An N x N matrix (list of lists).
        B: An N x N matrix (list of lists).
        tile_size: The side length of each square tile. Default 32.

    Returns:
        The N x N product matrix C = A * B.
    """
    pass  # TODO: Implement this


# ---------------------------------------------------------------------------
# TODO: Implement benchmarking
# ---------------------------------------------------------------------------

def benchmark(
    sizes: list[int] = [128, 256, 512, 1024],
    tile_sizes: list[int] = [16, 32, 64],
) -> None:
    """Compare naive vs. tiled multiplication across different matrix sizes.

    For each matrix size:
    1. Generate two random NxN matrices (using the same seed for reproducibility).
    2. Time the naive implementation.
    3. Time the tiled implementation for each tile_size.
    4. Verify that all results match.
    5. Print a summary with times and speedup ratios.

    Args:
        sizes: List of matrix dimensions to benchmark.
        tile_sizes: List of tile sizes to test for the tiled implementation.
    """
    pass  # TODO: Implement this


# ---------------------------------------------------------------------------
# Quick correctness check
# ---------------------------------------------------------------------------

def test_correctness() -> None:
    """Run basic correctness checks on both implementations."""
    print("Running correctness checks...")

    # 2x2 test
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = [[19, 22], [43, 50]]

    result_naive = matmul_naive(A, B)
    assert result_naive == expected, f"Naive 2x2 failed: {result_naive}"

    result_tiled = matmul_tiled(A, B, tile_size=1)
    assert result_tiled is not None, "matmul_tiled returned None — not yet implemented?"
    assert matrices_equal(result_tiled, expected), f"Tiled (ts=1) 2x2 failed: {result_tiled}"

    result_tiled = matmul_tiled(A, B, tile_size=2)
    assert matrices_equal(result_tiled, expected), f"Tiled (ts=2) 2x2 failed: {result_tiled}"

    # 3x3 test — not divisible by tile_size=2
    A = [[1, 0, 2], [0, 1, 0], [3, 0, 1]]
    B = [[1, 2, 0], [0, 1, 1], [2, 0, 1]]
    expected = [[5, 2, 2], [0, 1, 1], [5, 6, 1]]

    result_naive = matmul_naive(A, B)
    assert result_naive == expected, f"Naive 3x3 failed: {result_naive}"

    result_tiled = matmul_tiled(A, B, tile_size=2)
    assert matrices_equal(result_tiled, expected), f"Tiled (ts=2) 3x3 failed: {result_tiled}"

    # Random matrix test
    random.seed(42)
    n = 50
    A = generate_matrix(n, seed=100)
    B = generate_matrix(n, seed=200)
    result_naive = matmul_naive(A, B)
    for ts in [1, 7, 16, 25, 50, 64]:
        result_tiled = matmul_tiled(A, B, tile_size=ts)
        assert matrices_equal(result_naive, result_tiled), (
            f"Tiled (ts={ts}) does not match naive for {n}x{n}"
        )

    print("All correctness checks passed!\n")


if __name__ == "__main__":
    test_correctness()
    benchmark()
