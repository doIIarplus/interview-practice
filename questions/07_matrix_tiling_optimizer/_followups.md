# Follow-Up Questions: Question 07 — Matrix Tiling Optimizer

---

## 1. Why does tiling improve performance? Explain in terms of cache lines and memory hierarchy.

**Expected Answer:**
- CPUs load memory in cache lines (typically 64 bytes). In naive matmul, accessing
  `B[k][j]` with incrementing `k` jumps across rows, meaning each access may load a new
  cache line but only use one element from it.
- Tiling ensures that once a block of B is loaded into cache, all its elements are used
  before being evicted. This exploits both spatial locality (using nearby elements within
  a cache line) and temporal locality (reusing the same data multiple times).
- The working set of three tile-sized blocks should fit in L1 cache. For tile_size=32
  with 8-byte floats: `3 * 32 * 32 * 8 = 24,576 bytes = 24 KB`, well within a typical
  32-64 KB L1 cache.

---

## 2. What tile size would be optimal and why? How does it relate to L1 cache size?

**Expected Answer:**
- The optimal tile size is the largest T such that three T x T blocks of the matrix fit in
  L1 cache: `3 * T^2 * element_size <= L1_cache_size`.
- For a 32 KB L1 with 8-byte doubles: `T <= sqrt(32768 / 24) ~= 37`, so T=32 is a
  common choice.
- In practice, you want to leave room for other data in the cache, so slightly smaller
  tiles may perform better.
- The optimal tile size depends on the specific CPU architecture and should be tuned
  empirically.
- In pure Python, each "float" is a 28-byte object with pointer indirection, so the
  cache analysis differs from C. The improvement still exists but is less dramatic.

---

## 3. How would you extend this to GPU execution? What maps to CUDA thread blocks?

**Expected Answer:**
- In CUDA, tiles map directly to **thread blocks**. Each thread block loads a tile of A
  and a tile of B into **shared memory** (fast, on-chip SRAM).
- Within a thread block, each thread computes one or more elements of the output tile.
- The shared memory acts like a programmer-managed L1 cache — you explicitly control
  what data is loaded and when.
- The tile size is bounded by the shared memory per block (typically 48-96 KB).
- Multiple thread blocks execute concurrently across SMs (streaming multiprocessors),
  covering different output tiles.

---

## 4. What is memory coalescing in GPU programming and how does it relate to this problem?

**Expected Answer:**
- Memory coalescing means that adjacent threads in a warp access adjacent memory locations,
  allowing the GPU to combine multiple memory requests into a single wide transaction.
- In matrix multiplication, coalescing requires that threads in a warp access consecutive
  columns of B (or consecutive rows, depending on the layout).
- Poor coalescing is the GPU equivalent of poor cache line utilization on CPUs.
- Tiling helps because shared memory has no coalescing requirement — once data is in
  shared memory, it can be accessed in any pattern efficiently.

---

## 5. How would you handle non-square matrices?

**Expected Answer:**
- For A (M x K) * B (K x N) = C (M x N), you tile along all three dimensions independently:
  - ii steps through rows of A/C (0 to M)
  - jj steps through columns of B/C (0 to N)
  - kk steps through the shared dimension (0 to K)
- The `min()` bounds handle the edges just as with square matrices.
- The tile doesn't need to be the same size in all dimensions — you could use different
  tile sizes for M, N, and K dimensions based on the matrix shape.

---

## 6. What other optimizations exist beyond tiling?

**Expected Answer:**
- **Loop interchange (ikj ordering)**: Even without tiling, changing the naive loop order
  from ijk to ikj makes B access row-wise, improving locality. This is simpler than
  tiling and sometimes nearly as effective.
- **Loop unrolling**: Reduces loop overhead and enables instruction-level parallelism.
- **SIMD/vectorization**: Use SSE/AVX instructions to process 4-8 floats per instruction.
- **Strassen's algorithm**: Reduces complexity from O(N^3) to O(N^2.807) by trading
  multiplications for additions. Practical for large matrices.
- **Copy optimization (packing)**: Copy tiles into contiguous memory buffers to eliminate
  stride-related cache misses.
- **Prefetching**: Issue prefetch instructions to bring the next tile into cache before
  it's needed.
- **Multi-threading**: Parallelize over tiles using threads (each thread handles different
  output tiles).

---

## 7. In a real system, you'd use NumPy/BLAS. Why are those so much faster than pure Python?

**Expected Answer:**
- **Compiled code**: BLAS routines (e.g., OpenBLAS, MKL, ATLAS) are written in highly
  optimized C/Fortran/assembly, eliminating Python interpreter overhead.
- **SIMD**: They use AVX-512 or similar to process 8+ doubles per instruction.
- **Cache optimization**: They implement multi-level tiling, packing, and prefetching
  strategies tuned for specific CPU architectures.
- **Multi-threaded**: They parallelize across CPU cores automatically.
- **Python overhead**: Each Python float is a heap-allocated 28-byte object. A C double
  is just 8 bytes on the stack/in a register. Python's interpreter loop adds overhead
  to every arithmetic operation.
- A well-tuned BLAS implementation can approach the theoretical peak FLOP/s of the CPU,
  while pure Python may achieve less than 1% of that.
