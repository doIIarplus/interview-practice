"""
Question 13: GPU Kernel Simulator

Simulate a single Streaming Multiprocessor (SM) executing a kernel to reason
about GPU performance characteristics: bank conflicts, memory coalescing,
and shared memory tiling optimizations.

Run this file directly to execute the test cases:
    python starter.py
"""

from __future__ import annotations

import math
import random
from collections import defaultdict


class GPUSimulator:
    """Simulates a single GPU Streaming Multiprocessor (SM).

    Models shared memory bank conflicts, global memory coalescing,
    and tiled matrix transpose with and without padding optimization.
    """

    def __init__(
        self,
        shared_mem_kb: int = 48,
        num_banks: int = 32,
        warp_size: int = 32,
    ) -> None:
        """Initialize the GPU simulator.

        Args:
            shared_mem_kb: Shared memory size in kilobytes.
            num_banks: Number of shared memory banks.
            warp_size: Number of threads per warp.
        """
        self.shared_mem_bytes = shared_mem_kb * 1024
        self.num_banks = num_banks
        self.warp_size = warp_size

    def bank_conflict_count(self, addresses: list[int]) -> int:
        """Count bank conflicts for a warp's shared memory access.

        Given one address per thread (32 addresses for a warp), compute how
        many extra serialized cycles are caused by bank conflicts.

        Bank mapping: bank = (address // 4) % num_banks

        Rules:
        - Multiple threads accessing *different* addresses in the same bank
          cause bank conflicts (serialized).
        - Multiple threads accessing the *exact same* address get a broadcast
          (no conflict).
        - If a bank has K distinct addresses, it requires K serialized accesses,
          contributing K - 1 extra cycles.

        Args:
            addresses: List of 32 byte-addresses, one per thread in the warp.

        Returns:
            Total number of extra serialized cycles (0 means no conflicts).
        """
        # TODO: Implement this method
        pass

    def is_coalesced(
        self, addresses: list[int], cache_line_bytes: int = 128
    ) -> tuple[bool, int]:
        """Analyze whether a warp's global memory access is coalesced.

        Coalesced access means threads access a contiguous region that maps to
        a minimal number of cache lines.

        Args:
            addresses: List of 32 byte-addresses for global memory access.
            cache_line_bytes: Size of a cache line in bytes.

        Returns:
            Tuple of (is_perfectly_coalesced, num_cache_lines).
            Perfectly coalesced: 32 threads * 4 bytes = 128 bytes fits in 1 cache line.
        """
        # TODO: Implement this method
        pass

    def simulate_transpose(
        self,
        matrix: list[list[float]],
        block_dim: tuple[int, int] = (32, 32),
    ) -> tuple[list[list[float]], dict]:
        """Simulate matrix transpose using shared memory tiling (naive).

        For each tile:
        1. Load tile from global memory into shared memory (row-major).
           Thread (tx, ty) writes to shared[ty][tx].
        2. Read tile from shared memory in transposed order.
           Thread (tx, ty) reads shared[tx][ty] -- this causes bank conflicts
           because consecutive threads (varying tx) read from the same column,
           hitting the same bank.
        3. Write transposed tile to output in global memory.

        Args:
            matrix: 2D list of floats (M rows x N cols).
            block_dim: Tile dimensions (tile_rows, tile_cols).

        Returns:
            Tuple of:
            - Transposed matrix (N rows x M cols).
            - Stats dict with keys:
              "bank_conflicts": total bank conflicts across all tiles,
              "global_mem_transactions": total global memory transactions,
              "tiles_processed": number of tiles processed.
        """
        # TODO: Implement this method
        pass

    def simulate_transpose_padded(
        self,
        matrix: list[list[float]],
        block_dim: tuple[int, int] = (32, 32),
    ) -> tuple[list[list[float]], dict]:
        """Simulate matrix transpose with padded shared memory tiles.

        Same as simulate_transpose, but the shared memory tile has an extra
        column: dimensions are block_dim[0] x (block_dim[1] + 1).

        This changes the bank mapping on column access:
        - Without padding, shared[row][col] at byte offset (row * TILE_DIM + col) * 4
          -> bank = ((row * TILE_DIM + col) * 4 // 4) % 32 = (row * TILE_DIM + col) % 32
          When TILE_DIM = 32 and col is fixed, all rows map to the same bank.
        - With padding, shared[row][col] at byte offset (row * (TILE_DIM + 1) + col) * 4
          -> bank = (row * (TILE_DIM + 1) + col) % 32 = (row * 33 + col) % 32
          Since 33 and 32 are coprime, consecutive rows map to different banks.

        Args:
            matrix: 2D list of floats (M rows x N cols).
            block_dim: Tile dimensions (tile_rows, tile_cols).

        Returns:
            Tuple of:
            - Transposed matrix (N rows x M cols).
            - Stats dict (same format as simulate_transpose, but bank_conflicts
              should be 0 or near-0).
        """
        # TODO: Implement this method
        pass

    # ---- Helper: simulate shared memory bank mapping ----

    def _get_bank(self, address: int) -> int:
        """Return the bank number for a given byte address."""
        return (address // 4) % self.num_banks

    def _shared_mem_offset(
        self, row: int, col: int, row_stride: int
    ) -> int:
        """Return the byte offset in shared memory for element [row][col].

        Args:
            row: Row index in the shared memory tile.
            col: Column index in the shared memory tile.
            row_stride: Number of elements per row (may include padding).

        Returns:
            Byte offset (each element is 4 bytes).
        """
        return (row * row_stride + col) * 4


# --------------------------------------------------------------------------- #
#  Helper utilities                                                           #
# --------------------------------------------------------------------------- #

def generate_matrix(rows: int, cols: int, seed: int = 42) -> list[list[float]]:
    """Generate a matrix with predictable random values.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        seed: Random seed for reproducibility.

    Returns:
        2D list of floats.
    """
    rng = random.Random(seed)
    return [[rng.uniform(-10.0, 10.0) for _ in range(cols)] for _ in range(rows)]


def matrices_equal(
    a: list[list[float]], b: list[list[float]], tol: float = 1e-9
) -> bool:
    """Check if two matrices are equal within tolerance.

    Args:
        a: First matrix.
        b: Second matrix.
        tol: Absolute tolerance for float comparison.

    Returns:
        True if matrices have the same shape and all elements are within tol.
    """
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return False
        for va, vb in zip(row_a, row_b):
            if abs(va - vb) > tol:
                return False
    return True


def naive_transpose(matrix: list[list[float]]) -> list[list[float]]:
    """Compute transpose of a matrix (reference implementation).

    Args:
        matrix: 2D list of floats (M x N).

    Returns:
        Transposed matrix (N x M).
    """
    if not matrix or not matrix[0]:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]


def print_matrix(matrix: list[list[float]], label: str = "") -> None:
    """Pretty-print a matrix for debugging.

    Args:
        matrix: 2D list to print.
        label: Optional label printed above the matrix.
    """
    if label:
        print(f"{label}:")
    for row in matrix:
        print("  [" + ", ".join(f"{v:8.2f}" for v in row) + "]")
    print()


# --------------------------------------------------------------------------- #
#  Test cases                                                                 #
# --------------------------------------------------------------------------- #

def test_bank_conflicts() -> None:
    """Test bank conflict counting logic."""
    sim = GPUSimulator()

    # --- Sequential access: thread i reads word i (stride=4 bytes) ---
    # Each thread hits a unique bank -> no conflicts
    addresses = [i * 4 for i in range(32)]
    result = sim.bank_conflict_count(addresses)
    assert result == 0, f"Sequential access: expected 0 conflicts, got {result}"

    # --- All threads access same bank but different addresses ---
    # Stride = 128 = 32 banks * 4 bytes, so all addresses map to bank 0
    addresses = [i * 128 for i in range(32)]
    result = sim.bank_conflict_count(addresses)
    assert result == 31, f"All-same-bank: expected 31 conflicts, got {result}"

    # --- Broadcast: all threads access exact same address ---
    addresses = [0] * 32
    result = sim.bank_conflict_count(addresses)
    assert result == 0, f"Broadcast: expected 0 conflicts, got {result}"

    # --- Stride-2 (8 bytes): 2 threads per bank for 16 banks ---
    addresses = [i * 8 for i in range(32)]
    result = sim.bank_conflict_count(addresses)
    assert result == 16, f"Stride-2: expected 16 conflicts, got {result}"

    # --- Mixed: some broadcast, some conflict ---
    # 16 threads read address 0 (broadcast), 16 threads read different
    # addresses all in bank 0
    addresses = [0] * 16 + [128 * (i + 1) for i in range(16)]
    result = sim.bank_conflict_count(addresses)
    # Bank 0: address 0 (broadcast -> 1 access) + 16 distinct addresses = 17 distinct -> 16 conflicts
    assert result == 16, f"Mixed broadcast+conflict: expected 16 conflicts, got {result}"

    print("  [PASS] test_bank_conflicts")


def test_coalescing() -> None:
    """Test memory coalescing analysis."""
    sim = GPUSimulator()

    # Perfectly coalesced: sequential 4-byte accesses starting at aligned addr
    addresses = [i * 4 for i in range(32)]
    coalesced, lines = sim.is_coalesced(addresses)
    assert coalesced is True, f"Sequential: expected coalesced=True, got {coalesced}"
    assert lines == 1, f"Sequential: expected 1 cache line, got {lines}"

    # Strided access: each thread 512 bytes apart -> many cache lines
    addresses = [i * 512 for i in range(32)]
    coalesced, lines = sim.is_coalesced(addresses)
    assert coalesced is False, f"Strided: expected coalesced=False, got {coalesced}"
    assert lines == 128, f"Strided: expected 128 cache lines, got {lines}"

    # Two cache lines: sequential but starting at non-aligned address
    # 32 accesses * 4 bytes = 128 bytes; if start is at offset 64, spans 2 lines
    addresses = [64 + i * 4 for i in range(32)]
    coalesced, lines = sim.is_coalesced(addresses)
    assert coalesced is False, f"Offset: expected coalesced=False, got {coalesced}"
    assert lines == 2, f"Offset: expected 2 cache lines, got {lines}"

    print("  [PASS] test_coalescing")


def test_transpose_correctness() -> None:
    """Test that both transpose methods produce correct results."""
    sim = GPUSimulator()

    # Small square matrix
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


def test_transpose_non_square() -> None:
    """Test transpose with a non-square, non-tile-aligned matrix."""
    sim = GPUSimulator()

    # 10 x 7 matrix -- not divisible by any typical tile size
    matrix = generate_matrix(10, 7, seed=123)
    expected = naive_transpose(matrix)

    result, stats = sim.simulate_transpose(matrix, block_dim=(4, 4))
    assert matrices_equal(result, expected), "Naive transpose failed on 10x7 matrix"

    result_p, stats_p = sim.simulate_transpose_padded(matrix, block_dim=(4, 4))
    assert matrices_equal(result_p, expected), "Padded transpose failed on 10x7 matrix"

    print("  [PASS] test_transpose_non_square (10x7)")


def test_padding_reduces_conflicts() -> None:
    """Test that padded transpose has fewer (ideally zero) bank conflicts."""
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


def test_large_transpose() -> None:
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


def run_all_tests() -> None:
    """Run all test cases."""
    print("Running GPU Kernel Simulator tests...\n")

    test_bank_conflicts()
    test_coalescing()
    test_transpose_correctness()
    test_transpose_non_square()
    test_padding_reduces_conflicts()
    test_large_transpose()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_all_tests()
