"""Hidden tests for Question 07: Matrix Tiling Optimizer
Run: python questions/07_matrix_tiling_optimizer/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import random
from starter import matmul_naive, matmul_tiled, generate_matrix, matrices_equal, time_function


def test_correctness():
    """Run correctness checks on both implementations."""
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

    print("[PASS] test_correctness\n")


def test_benchmark():
    """Basic benchmark to verify tiled is at least as correct as naive."""
    n = 64
    A = generate_matrix(n, seed=300)
    B = generate_matrix(n, seed=400)

    result_naive = matmul_naive(A, B)
    result_tiled = matmul_tiled(A, B, tile_size=16)
    assert matrices_equal(result_naive, result_tiled), "Tiled should match naive"
    print("[PASS] test_benchmark\n")


def run_tests():
    print("=" * 60)
    print("Matrix Tiling Optimizer — Hidden Tests")
    print("=" * 60 + "\n")

    test_correctness()
    test_benchmark()

    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
