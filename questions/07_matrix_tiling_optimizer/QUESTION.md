# Question 07: Matrix Tiling Optimizer

## Difficulty: Medium-Hard
## Topics: Cache Optimization, Memory Hierarchy, Tiled Algorithms, Performance Engineering
## Estimated Time: 45-60 minutes

---

## Background

In naive matrix multiplication, accessing elements of matrix **B** column-by-column causes
poor cache utilization. This is because matrices stored in row-major order (as Python lists of
lists) have their rows laid out contiguously in memory. When you iterate down a column of B,
each access jumps to a completely different row, evicting useful cache lines and causing
frequent cache misses.

**Tiled (blocked) matrix multiplication** addresses this by breaking the matrices into smaller
sub-blocks (tiles) that fit in the CPU cache. By operating on one tile at a time, you maximize
both spatial and temporal locality, keeping frequently accessed data in cache for as long as
possible.

This technique is foundational in high-performance computing and is a core concept behind
GPU kernel optimization, where thread blocks operate on shared-memory tiles.

---

## Task

You are given a naive matrix multiplication implementation that multiplies two NxN matrices.
Your task is to optimize it by implementing tiled (blocked) matrix multiplication to improve
cache performance.

All matrices are represented as `list[list[float]]` — no NumPy. We want to see the raw
algorithm and your understanding of why it performs better.

### Part 1: Verify the Naive Implementation

The starter code includes a naive matrix multiplication:

```python
def matmul_naive(A: list[list[float]], B: list[list[float]]) -> list[list[float]]
```

Verify you understand the standard O(N^3) triple-loop algorithm before proceeding.

### Part 2: Implement Tiled Matrix Multiplication

```python
def matmul_tiled(A: list[list[float]], B: list[list[float]], tile_size: int = 32) -> list[list[float]]
```

Requirements:
- Use a 6-loop tiling structure: three outer loops iterate over tile boundaries (ii, jj, kk),
  and three inner loops iterate within each tile (i, j, k).
- Must handle matrix sizes that are **not** evenly divisible by `tile_size`. For example,
  a 100x100 matrix with `tile_size=32` must still produce correct results.
- Must produce results **identical** to the naive implementation (within floating-point tolerance).

### Part 3: Benchmark

Write a benchmarking function that:
- Compares naive vs. tiled multiplication across matrix sizes: 128, 256, 512, 1024
- Reports execution time for each implementation at each size
- Verifies correctness by checking that both implementations produce the same result
- Experiments with different tile sizes (e.g., 16, 32, 64) to find the best performer

---

## Example

```python
>>> A = [[1, 2], [3, 4]]
>>> B = [[5, 6], [7, 8]]
>>> matmul_naive(A, B)
[[19, 22], [43, 50]]
>>> matmul_tiled(A, B, tile_size=1)
[[19, 22], [43, 50]]
>>> matmul_tiled(A, B, tile_size=2)
[[19, 22], [43, 50]]
```

For a 3x3 example:
```python
>>> A = [[1, 0, 2], [0, 1, 0], [3, 0, 1]]
>>> B = [[1, 2, 0], [0, 1, 1], [2, 0, 1]]
>>> matmul_naive(A, B)
[[5, 2, 2], [0, 1, 1], [5, 6, 1]]
>>> matmul_tiled(A, B, tile_size=2)  # Note: 3 is not divisible by 2
[[5, 2, 2], [0, 1, 1], [5, 6, 1]]
```

### Expected Benchmark Output (approximate)

```
Matrix Tiling Optimizer Benchmark
==================================

Size: 128x128
  Naive:  0.482s
  Tiled (tile_size=16): 0.391s  (1.23x speedup)
  Tiled (tile_size=32): 0.365s  (1.32x speedup)
  Tiled (tile_size=64): 0.378s  (1.28x speedup)
  Correctness: PASS

Size: 256x256
  Naive:  4.012s
  Tiled (tile_size=16): 2.987s  (1.34x speedup)
  Tiled (tile_size=32): 2.654s  (1.51x speedup)
  Tiled (tile_size=64): 2.801s  (1.43x speedup)
  Correctness: PASS

...
```

Note: Actual speedups in pure Python will be modest compared to C/C++ because Python's
interpreter overhead dominates. The algorithmic improvement is still measurable and the
concepts translate directly to systems-level code and GPU kernels.

---

## Constraints

- Do not use NumPy, ctypes, or any C extensions
- Matrices are always square (NxN)
- Matrix elements are floating-point numbers
- N will be at least 1 and at most 2048
- `tile_size` will be at least 1

---

## What We're Looking For

1. A correct tiled implementation with the proper 6-loop structure
2. Correct handling of edge cases (non-aligned sizes, tile_size=1, tile_size >= N)
3. Clear understanding of **why** tiling improves cache performance
4. Thoughtful benchmarking methodology
5. Ability to reason about memory hierarchy and its impact on performance
