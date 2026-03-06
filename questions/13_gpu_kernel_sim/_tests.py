"""Hidden tests for Question 13: GPU Kernel Simulator
Run: python questions/13_gpu_kernel_sim/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import GPUSimulator, generate_matrix, matrices_equal, naive_transpose


def test_bank_conflicts():
    """Test bank conflict counting logic."""
    sim = GPUSimulator()

    # Sequential access: no conflicts
    addresses = [i * 4 for i in range(32)]
    result = sim.bank_conflict_count(addresses)
    assert result == 0, f"Sequential access: expected 0 conflicts, got {result}"

    # All threads access same bank but different addresses
    addresses = [i * 128 for i in range(32)]
    result = sim.bank_conflict_count(addresses)
    assert result == 31, f"All-same-bank: expected 31 conflicts, got {result}"

    # Broadcast: all threads access exact same address
    addresses = [0] * 32
    result = sim.bank_conflict_count(addresses)
    assert result == 0, f"Broadcast: expected 0 conflicts, got {result}"

    # Stride-2: 2 threads per bank
    addresses = [i * 8 for i in range(32)]
    result = sim.bank_conflict_count(addresses)
    assert result == 16, f"Stride-2: expected 16 conflicts, got {result}"

    # Mixed: some broadcast, some conflict
    addresses = [0] * 16 + [128 * (i + 1) for i in range(16)]
    result = sim.bank_conflict_count(addresses)
    assert result == 16, f"Mixed broadcast+conflict: expected 16 conflicts, got {result}"

    print("  [PASS] test_bank_conflicts")


def test_coalescing():
    """Test memory coalescing analysis."""
    sim = GPUSimulator()

    # Perfectly coalesced
    addresses = [i * 4 for i in range(32)]
    coalesced, lines = sim.is_coalesced(addresses)
    assert coalesced is True, f"Sequential: expected coalesced=True, got {coalesced}"
    assert lines == 1, f"Sequential: expected 1 cache line, got {lines}"

    # Strided access
    addresses = [i * 512 for i in range(32)]
    coalesced, lines = sim.is_coalesced(addresses)
    assert coalesced is False, f"Strided: expected coalesced=False, got {coalesced}"
    assert lines == 128, f"Strided: expected 128 cache lines, got {lines}"

    # Two cache lines
    addresses = [64 + i * 4 for i in range(32)]
    coalesced, lines = sim.is_coalesced(addresses)
    assert coalesced is False, f"Offset: expected coalesced=False, got {coalesced}"
    assert lines == 2, f"Offset: expected 2 cache lines, got {lines}"

    print("  [PASS] test_coalescing")


def test_transpose_correctness():
    """Test that both transpose methods produce correct results."""
    sim = GPUSimulator()

    matrix = [[float(r * 4 + c) for c in range(4)] for r in range(4)]
    expected = naive_transpose(matrix)

    result_naive, stats_naive = sim.simulate_transpose(matrix, block_dim=(4, 4))
    assert matrices_equal(result_naive, expected), (
        f"Naive transpose incorrect.\nExpected: {expected}\nGot: {result_naive}"
    )
    assert stats_naive["tiles_processed"] == 1

    result_padded, stats_padded = sim.simulate_transpose_padded(matrix, block_dim=(4, 4))
    assert matrices_equal(result_padded, expected), (
        f"Padded transpose incorrect.\nExpected: {expected}\nGot: {result_padded}"
    )

    print("  [PASS] test_transpose_correctness (small)")


def test_transpose_non_square():
    """Test transpose with a non-square, non-tile-aligned matrix."""
    sim = GPUSimulator()

    matrix = generate_matrix(10, 7, seed=123)
    expected = naive_transpose(matrix)

    result, stats = sim.simulate_transpose(matrix, block_dim=(4, 4))
    assert matrices_equal(result, expected), "Naive transpose failed on 10x7 matrix"

    result_p, stats_p = sim.simulate_transpose_padded(matrix, block_dim=(4, 4))
    assert matrices_equal(result_p, expected), "Padded transpose failed on 10x7 matrix"

    print("  [PASS] test_transpose_non_square (10x7)")


def test_padding_reduces_conflicts():
    """Test that padded transpose has fewer bank conflicts."""
    sim = GPUSimulator()

    matrix = generate_matrix(32, 32, seed=999)

    _, stats_naive = sim.simulate_transpose(matrix, block_dim=(32, 32))
    _, stats_padded = sim.simulate_transpose_padded(matrix, block_dim=(32, 32))

    print(f"    Naive bank conflicts:  {stats_naive['bank_conflicts']}")
    print(f"    Padded bank conflicts: {stats_padded['bank_conflicts']}")

    assert stats_padded["bank_conflicts"] < stats_naive["bank_conflicts"], (
        "Padded transpose should have fewer bank conflicts than naive"
    )
    assert stats_padded["bank_conflicts"] == 0, (
        f"Padded transpose should have 0 conflicts, got {stats_padded['bank_conflicts']}"
    )

    print("  [PASS] test_padding_reduces_conflicts")


def test_large_transpose():
    """Test transpose on a larger matrix for correctness."""
    sim = GPUSimulator()

    matrix = generate_matrix(100, 64, seed=7)
    expected = naive_transpose(matrix)

    result, stats = sim.simulate_transpose(matrix, block_dim=(32, 32))
    assert matrices_equal(result, expected), "Naive transpose failed on 100x64 matrix"

    result_p, stats_p = sim.simulate_transpose_padded(matrix, block_dim=(32, 32))
    assert matrices_equal(result_p, expected), "Padded transpose failed on 100x64 matrix"

    print(f"    Tiles processed: {stats['tiles_processed']}")
    print(f"    Naive conflicts: {stats['bank_conflicts']}")
    print(f"    Padded conflicts: {stats_p['bank_conflicts']}")
    print("  [PASS] test_large_transpose (100x64)")


def run_tests():
    print("Running GPU Kernel Simulator tests...\n")

    test_bank_conflicts()
    test_coalescing()
    test_transpose_correctness()
    test_transpose_non_square()
    test_padding_reduces_conflicts()
    test_large_transpose()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
