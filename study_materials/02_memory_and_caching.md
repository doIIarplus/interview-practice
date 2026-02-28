# Memory Hierarchy & Cache Optimization — From First Principles

This guide explains why some memory access patterns are fast and others are slow,
how CPU caches work, and how to write cache-friendly code. These concepts are
directly tested in Q07 (Matrix Tiling), Q11 (Memory Pool), and Q13 (GPU Kernel
Sim), and are fundamental to performance engineering at every level.

---

## Table of Contents

1. [The Memory Wall](#the-memory-wall)
2. [CPU Cache Hierarchy](#cpu-cache-hierarchy)
3. [Cache Lines: The Fundamental Unit](#cache-lines-the-fundamental-unit)
4. [Spatial and Temporal Locality](#spatial-and-temporal-locality)
5. [Why Naive Matrix Multiply Is Slow](#why-naive-matrix-multiply-is-slow)
6. [Tiled / Blocked Algorithms](#tiled--blocked-algorithms)
7. [Memory Alignment and Padding](#memory-alignment-and-padding)
8. [False Sharing](#false-sharing)
9. [Mapping to GPU Memory](#mapping-to-gpu-memory)
10. [Practical Python Perspective](#practical-python-perspective)
11. [Key Formulas and Numbers](#key-formulas-and-numbers)

---

## The Memory Wall

Modern CPUs can execute billions of operations per second, but memory cannot
deliver data nearly that fast. This gap — the "memory wall" — is the central
challenge of performance engineering.

```
                    Latency (approximate)
                    =====================
  CPU Register      ~0.3 ns    (1 cycle)
  L1 Cache          ~1 ns      (3-4 cycles)
  L2 Cache          ~4 ns      (12-14 cycles)
  L3 Cache          ~12 ns     (40 cycles)
  DRAM (Main Mem)   ~60 ns     (200 cycles)
  SSD               ~100 us    (100,000 ns)
  Network (DC)      ~500 us    (500,000 ns)
  HDD               ~5 ms      (5,000,000 ns)

  Ratio: DRAM is ~200x slower than L1 cache.
  Ratio: SSD is ~100,000x slower than L1 cache.
```

**Key insight:** A program that accesses main memory on every operation can be
200x slower than one that keeps its data in L1 cache. Caches exist to bridge
this gap automatically — but only if your access patterns cooperate.

---

## CPU Cache Hierarchy

Modern CPUs have a hierarchy of caches between the registers and main memory:

```
┌──────────────────────────────────────────────────────────────────────┐
│  CPU Core 0                          CPU Core 1                     │
│  ┌─────────────┐                     ┌─────────────┐                │
│  │  Registers  │                     │  Registers  │                │
│  │  (fastest)  │                     │  (fastest)  │                │
│  └──────┬──────┘                     └──────┬──────┘                │
│         │                                   │                       │
│  ┌──────┴──────┐                     ┌──────┴──────┐                │
│  │  L1 Cache   │  ~32-64 KB         │  L1 Cache   │                │
│  │  (per-core) │  ~4 cycle latency   │  (per-core) │                │
│  └──────┬──────┘                     └──────┴──────┘                │
│         │                                   │                       │
│  ┌──────┴──────┐                     ┌──────┴──────┐                │
│  │  L2 Cache   │  ~256 KB - 1 MB     │  L2 Cache   │                │
│  │  (per-core) │  ~12 cycle latency  │  (per-core) │                │
│  └──────┬──────┘                     └──────┴──────┘                │
│         │                                   │                       │
│         └──────────────┬────────────────────┘                       │
│                        │                                            │
│                 ┌──────┴──────┐                                     │
│                 │  L3 Cache   │  ~4-32 MB                           │
│                 │  (shared)   │  ~40 cycle latency                  │
│                 └──────┬──────┘                                     │
│                        │                                            │
└────────────────────────┼────────────────────────────────────────────┘
                         │
                  ┌──────┴──────┐
                  │    DRAM     │  ~16-128 GB
                  │ (main mem)  │  ~200 cycle latency
                  └─────────────┘
```

### Key properties

- **L1 cache** is split into L1d (data) and L1i (instructions). Very fast, very
  small.
- **L2 cache** is unified (data + instructions). Larger, slightly slower.
- **L3 cache** is shared across all cores. Largest, slowest of the caches.
- Each level is **inclusive** — data in L1 is also in L2 and L3 (in most
  architectures).
- When data is not in any cache ("cache miss"), it must be fetched from DRAM.

---

## Cache Lines: The Fundamental Unit

Caches do not load individual bytes. They load **cache lines** — contiguous
blocks of memory, typically **64 bytes** on modern x86 and ARM processors.

```
Memory Address Space:
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ Line 0 │ Line 1 │ Line 2 │ Line 3 │ Line 4 │ Line 5 │  ...   │
│ 64 B   │ 64 B   │ 64 B   │ 64 B   │ 64 B   │ 64 B   │        │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┘
  addr     addr     addr
  0-63    64-127  128-191
```

### What this means

When you access a single `int32` (4 bytes) at address 100, the CPU loads the
entire 64-byte cache line containing addresses 64-127. This means:

- **Accessing addr 104 next is free** — it is already in the cache line.
- **Accessing addr 200 next requires a new cache line load** — potentially a
  cache miss.

### Example: Array iteration

```
Array of 16 int32 values (64 bytes total = 1 cache line):

┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ a0 │ a1 │ a2 │ a3 │ a4 │ a5 │ a6 │ a7 │ a8 │ a9 │a10 │a11 │a12 │a13 │a14 │a15 │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
  4B   4B   4B   4B   4B   4B   4B   4B   4B   4B   4B   4B   4B   4B   4B   4B
  └───────────────── one cache line (64 bytes) ─────────────────────────────────┘

Iterating a0, a1, a2, ..., a15:
  - a0: cache MISS (loads entire line)
  - a1 through a15: cache HIT (already loaded!)
  - Result: 1 miss per 16 elements = 93.75% hit rate
```

---

## Spatial and Temporal Locality

Caches exploit two types of **locality** in access patterns:

### Spatial Locality

**Nearby memory addresses are accessed close together in time.**

This is why sequential array iteration is fast: when you access `a[0]`, the
cache line containing `a[0]` through `a[15]` is loaded, and you access `a[1]`
next.

```
GOOD (spatial locality — sequential access):
  for i in range(N):
      sum += array[i]         # Each access is adjacent to the last

BAD (no spatial locality — random access):
  for i in random_indices:
      sum += array[i]         # Jumps all over memory
```

### Temporal Locality

**Recently accessed addresses are accessed again soon.**

This is why loops over small datasets are fast: the working set fits in cache
and stays there.

```
GOOD (temporal locality — small working set):
  for _ in range(1000):
      for i in range(64):     # Same 64 elements reused 1000 times
          sum += array[i]     # Stays in L1 cache

BAD (no temporal locality — large working set, single pass):
  for i in range(10_000_000):  # Huge array, each element used once
      sum += array[i]          # Elements evicted before reuse
```

---

## Why Naive Matrix Multiply Is Slow

Matrix multiplication is the canonical example of how access patterns affect
performance.

### Row-major storage (C/Python/NumPy default)

In row-major order, a 2D matrix is stored row by row in memory:

```
Matrix A (4x4):        Memory Layout:
┌─────┬─────┬─────┬─────┐
│ a00 │ a01 │ a02 │ a03 │    a00 a01 a02 a03 a10 a11 a12 a13 a20 ...
├─────┼─────┼─────┼─────┤    └─── row 0 ────┘└─── row 1 ────┘
│ a10 │ a11 │ a12 │ a13 │
├─────┼─────┼─────┼─────┤    Adjacent elements in the SAME ROW are
│ a20 │ a21 │ a22 │ a23 │    adjacent in memory (good spatial locality).
├─────┼─────┼─────┼─────┤
│ a30 │ a31 │ a32 │ a33 │    Adjacent elements in the SAME COLUMN are
└─────┴─────┴─────┴─────┘    N elements apart in memory (bad locality).
```

### Naive matrix multiply: C = A @ B

```python
# C[i][j] = sum(A[i][k] * B[k][j] for k in range(N))
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]
```

Access pattern analysis:
- `A[i][k]`: iterating over k = iterating across row i. **Sequential access.
  Good locality!**
- `B[k][j]`: iterating over k = iterating down column j. **Stride-N access.
  Bad locality!**

```
Accessing B column-wise:

B[0][j]  ←── cache line loaded
B[1][j]  ←── N elements away = different cache line (MISS!)
B[2][j]  ←── another cache line (MISS!)
B[3][j]  ←── another cache line (MISS!)
  ...

For an NxN matrix with N=1024 and 8-byte doubles:
  Row = 8192 bytes = 128 cache lines
  Each B[k][j] access misses (since the previous cache line was evicted
  by the time we come back to that row).
```

### The result

For large matrices (N > a few hundred), the naive algorithm is **dominated by
cache misses** on B. It might achieve only 1-2% of the CPU's theoretical peak
FLOPS.

---

## Tiled / Blocked Algorithms

The fix: process the matrices in small **tiles** (blocks) that fit in cache.

### The idea

Instead of computing the entire result one element at a time, divide A, B, and
C into TxT sub-matrices and process each sub-matrix product:

```
┌─────────┬─────────┐        ┌─────────┬─────────┐
│  A_00   │  A_01   │        │  B_00   │  B_01   │
│  (TxT)  │  (TxT)  │   @    │  (TxT)  │  (TxT)  │
├─────────┼─────────┤        ├─────────┼─────────┤
│  A_10   │  A_11   │        │  B_10   │  B_11   │
│  (TxT)  │  (TxT)  │        │  (TxT)  │  (TxT)  │
└─────────┴─────────┘        └─────────┴─────────┘

C_00 = A_00 @ B_00 + A_01 @ B_10
C_01 = A_00 @ B_01 + A_01 @ B_11
C_10 = A_10 @ B_00 + A_11 @ B_10
C_11 = A_10 @ B_01 + A_11 @ B_11
```

### Tiled matrix multiply

```python
def tiled_matmul(A, B, C, N, T):
    """Multiply NxN matrices A and B into C using TxT tiles."""
    for ii in range(0, N, T):         # Tile row of C
        for jj in range(0, N, T):     # Tile col of C
            for kk in range(0, N, T): # Tile of shared dimension
                # Multiply one TxT block
                for i in range(ii, min(ii + T, N)):
                    for j in range(jj, min(jj + T, N)):
                        for k in range(kk, min(kk + T, N)):
                            C[i][j] += A[i][k] * B[k][j]
```

### Why this works

Each tile of B (TxT elements) fits in L1 or L2 cache. While processing that
tile, we reuse it T times (once for each row of the A tile).

```
Choose T such that 3 * T * T * sizeof(element) fits in L1 cache.

L1 cache = 32 KB = 32768 bytes
Element = 8 bytes (float64)
3 * T^2 * 8 <= 32768
T^2 <= 1365
T <= 36

So T = 32 is a good tile size for L1.
T = 64 is good for L2 (256 KB).
```

### Performance impact

| Approach | Cache Misses | Relative Speed |
|----------|-------------|----------------|
| Naive    | O(N^3 / L)  | 1x (baseline)  |
| Tiled    | O(N^3 / (L * T)) | ~T/L faster |

For N=1024 with T=32: tiled can be 5-10x faster than naive. With SIMD and other
optimizations, optimized BLAS achieves 90%+ of peak FLOPS.

---

## Memory Alignment and Padding

### Alignment

CPUs access memory most efficiently when data is aligned to its natural size:
- `int32` should be at an address divisible by 4
- `int64` / `float64` should be at an address divisible by 8
- SIMD vectors (128-bit, 256-bit) should be at 16-byte or 32-byte boundaries

Misaligned access may:
- Require two cache line reads instead of one (if the data straddles a cache
  line boundary)
- Be silently slower on x86 (hardware handles it but with a penalty)
- Cause a fault on some architectures (ARM, historically)

### Struct padding

C compilers add padding to structs for alignment. This affects memory layout:

```
struct Example {        Memory layout:
    char a;     // 1B   [a][pad][pad][pad][b  b  b  b][ c ][ pad  pad  pad ]
    int b;      // 4B    0    1    2    3   4  5  6  7   8    9   10   11
    char c;     // 1B
};                      Total: 12 bytes (not 6!)
// sizeof = 12

struct Packed {         Reordered:
    int b;      // 4B   [b  b  b  b][a][c][pad][pad]
    char a;     // 1B    0  1  2  3   4  5   6    7
    char c;     // 1B
};                      Total: 8 bytes
// sizeof = 8
```

**Tip:** Order struct fields from largest to smallest to minimize padding.

---

## False Sharing

**False sharing** occurs when two threads modify different variables that happen
to be on the same cache line. The cache coherence protocol forces the cache line
to bounce between cores, serializing access.

```
Cache Line (64 bytes):
┌────────────────────────────────┬────────────────────────────────┐
│  counter_thread_0 (8 bytes)   │  counter_thread_1 (8 bytes)   │
└────────────────────────────────┴────────────────────────────────┘
        Thread 0 writes here            Thread 1 writes here

Even though they modify DIFFERENT variables, the cache line is shared.
Each write invalidates the other core's copy of the line.

Core 0: write counter_0 → invalidates Core 1's cache line
Core 1: write counter_1 → must reload line from Core 0 → invalidates Core 0
Core 0: write counter_0 → must reload line from Core 1 → ...
```

**Fix:** Pad variables so each thread's data is on a separate cache line:

```c
struct PaddedCounter {
    long value;
    char padding[56];  // 64 - 8 = 56 bytes of padding
};
// Each counter is now on its own cache line
```

---

## Mapping to GPU Memory

GPU memory hierarchy follows similar principles but with different terminology
and explicit programmer control:

```
CPU World                        GPU World
═══════════                      ═════════
Registers                        Registers (per-thread, fastest)
    │                                │
L1 Cache (automatic)            Shared Memory (per-block, PROGRAMMABLE)
    │                                │
L2 Cache (automatic)            L1/L2 Cache (automatic)
    │                                │
DRAM (main memory)              Global Memory (HBM, large, slow)
```

### Key differences

| Aspect | CPU | GPU |
|--------|-----|-----|
| L1 role | Automatic, hardware-managed | Shared memory is programmer-managed |
| Size per core | ~32 KB L1 | ~100 KB shared memory per SM |
| Latency (main memory) | ~200 cycles | ~400-800 cycles |
| Bandwidth (main memory) | ~50 GB/s (DDR5) | ~2-3 TB/s (HBM3) |
| Cache line / transaction size | 64 bytes | 32-128 bytes |

### Bank conflicts (GPU-specific)

GPU shared memory is divided into **32 banks** (one per thread in a warp). If
two threads in the same warp access the same bank (but different addresses),
the accesses are **serialized**.

```
Shared Memory Banks (32 banks):
Bank:  0    1    2    3    4    5   ...  31
       │    │    │    │    │    │         │
     addr0 addr1 addr2 addr3 addr4 ...  addr31
     addr32 addr33 ...

Thread 0 accesses Bank 0  ──┐
Thread 1 accesses Bank 1  ──┤  No conflict: all different banks
Thread 2 accesses Bank 2  ──┤  Full bandwidth!
  ...                       │
Thread 31 accesses Bank 31 ─┘

Thread 0 accesses Bank 0  ──┐
Thread 1 accesses Bank 0  ──┤  CONFLICT: 2 threads, same bank
  Result: serialized (2x slower)
```

### Memory coalescing (GPU-specific)

When threads in a warp access **adjacent** global memory addresses, the hardware
combines them into a single wide memory transaction. This is analogous to
spatial locality on CPUs but more explicit.

```
GOOD (coalesced): Thread i accesses address base + i * sizeof(element)
  Thread 0 → addr 0     ┐
  Thread 1 → addr 4     ├── Combined into ONE 128-byte transaction
  Thread 2 → addr 8     │
  ...                    │
  Thread 31 → addr 124  ┘

BAD (uncoalesced): Thread i accesses address base + i * stride
  Thread 0 → addr 0       ─── Transaction 1
  Thread 1 → addr 4096    ─── Transaction 2
  Thread 2 → addr 8192    ─── Transaction 3
  ...                          (32 separate transactions!)
```

---

## Practical Python Perspective

Python objects have significant overhead compared to C/C++ data. However, NumPy
arrays store data contiguously and benefit from cache effects.

### Demonstration: row vs column iteration in NumPy

```python
import numpy as np
import time

N = 4096
A = np.random.randn(N, N)

# Row-major iteration (good locality)
start = time.perf_counter()
row_sum = 0.0
for i in range(N):
    for j in range(N):
        row_sum += A[i, j]  # Sequential in memory
row_time = time.perf_counter() - start

# Column-major iteration (bad locality)
start = time.perf_counter()
col_sum = 0.0
for i in range(N):
    for j in range(N):
        col_sum += A[j, i]  # Stride-N access
col_time = time.perf_counter() - start

print(f"Row-major: {row_time:.3f}s")
print(f"Col-major: {col_time:.3f}s")
print(f"Ratio:     {col_time/row_time:.1f}x slower")
# Typical result: column-major is 2-5x slower
```

### Practical tips for Python

1. **Use NumPy vectorized operations** — they operate on contiguous memory in C.
2. **Prefer row-wise operations** on row-major arrays (NumPy default).
3. **Use `np.ascontiguousarray()`** if you need to ensure contiguous layout.
4. **Profile before optimizing** — Python's object overhead often dominates
   cache effects. Use `numpy` or `ctypes` for cache-sensitive code.

---

## Key Formulas and Numbers

### Cache performance

```
Hit Rate = cache hits / total accesses
Miss Rate = 1 - Hit Rate
Average Access Time = Hit_Time + Miss_Rate * Miss_Penalty

Example:
  L1 hit time = 4 cycles, L1 miss rate = 5%, L2 hit time = 12 cycles
  L2 miss rate = 10% (of L1 misses), DRAM = 200 cycles

  Avg time = 4 + 0.05 * (12 + 0.10 * 200)
           = 4 + 0.05 * 32
           = 5.6 cycles    (if working set fits mostly in L1)
```

### Working set size

```
If your working set fits in L1 (32 KB):  ~4 cycle access
If your working set fits in L2 (256 KB): ~12 cycle access
If your working set fits in L3 (8 MB):   ~40 cycle access
If your working set exceeds L3:          ~200 cycle access (DRAM)
```

### Optimal tile size

```
For tiled matrix multiply with 3 matrices (A tile, B tile, C tile):
  3 * T^2 * element_size <= Cache_Size

L1 (32 KB), float32 (4 bytes):  T <= 52  →  use T = 32 or 48
L1 (32 KB), float64 (8 bytes):  T <= 36  →  use T = 32
L2 (256 KB), float64 (8 bytes): T <= 103 →  use T = 64 or 96
```

### Bandwidth

```
Memory Bandwidth Utilization:
  Useful bytes transferred / Total time = Achieved Bandwidth
  Achieved Bandwidth / Peak Bandwidth = Bandwidth Utilization

Example (CPU):
  Peak DRAM bandwidth: 50 GB/s
  Matrix size: 1024 x 1024 x 8 bytes = 8 MB
  Time to read entire matrix: 8 MB / 50 GB/s = 0.16 ms

Example (GPU):
  Peak HBM bandwidth: 2 TB/s (A100)
  Matrix size: 4096 x 4096 x 2 bytes (FP16) = 32 MB
  Time to read entire matrix: 32 MB / 2 TB/s = 0.016 ms
```

### Arithmetic intensity

```
Arithmetic Intensity = FLOPs / Bytes transferred

Matrix multiply (NxN):
  FLOPs = 2 * N^3
  Bytes = 3 * N^2 * element_size (read A, B, write C)
  AI = 2 * N^3 / (3 * N^2 * elem_size) = 2N / (3 * elem_size)

  For N=1024, float32: AI = 2048 / 12 ≈ 170 FLOPs/byte
  → Matrix multiply is COMPUTE-bound (high AI)

Vector add (N elements):
  FLOPs = N
  Bytes = 3 * N * element_size (read A, B, write C)
  AI = N / (3 * N * elem_size) = 1 / (3 * elem_size)

  For float32: AI = 1/12 ≈ 0.08 FLOPs/byte
  → Vector add is MEMORY-bound (low AI)
```

---

## Summary: What to Remember for Interviews

1. **Cache lines are 64 bytes.** Accessing one element loads 64 bytes. Use
   spatial locality to make those extra bytes useful.

2. **Sequential access is fast, random access is slow.** Row-major iteration on
   a row-major array is sequential. Column-major iteration is stride-N.

3. **Tiling keeps data in cache.** Choose tile sizes so the working set fits in
   L1 or L2.

4. **Know the numbers.** L1 ~4 cycles, L2 ~12, L3 ~40, DRAM ~200. These ratios
   explain why cache optimization gives 5-50x speedups.

5. **GPU shared memory is programmable L1.** You explicitly load tiles from
   global memory into shared memory, analogous to blocking for CPU L1.

6. **Coalescing is spatial locality for GPUs.** Adjacent threads should access
   adjacent memory addresses.

7. **Arithmetic intensity determines the bottleneck.** High AI = compute-bound.
   Low AI = memory-bound. Optimize the bottleneck.
