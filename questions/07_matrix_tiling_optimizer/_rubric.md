# Rubric: Question 07 — Matrix Tiling Optimizer

**Total: 100 points**

---

## 1. Correct Tiled Implementation (30 points)

### Full Credit (30 pts)
- `matmul_tiled` produces results identical to `matmul_naive` for all tested matrix sizes
  and tile sizes (within floating-point tolerance).
- Works for edge cases: `tile_size=1`, `tile_size >= N`, `N=1`.

### Partial Credit (15-25 pts)
- Works for tile-aligned sizes but fails for non-aligned sizes.
- Minor off-by-one errors in loop bounds.

### Minimal Credit (5-10 pts)
- Demonstrates understanding of the concept but implementation has significant bugs.

### No Credit (0 pts)
- Not implemented or produces incorrect results for basic cases.

---

## 2. Proper 6-Loop Tiling Structure (20 points)

### Full Credit (20 pts)
The candidate uses the canonical 6-loop structure:

```python
for ii in range(0, n, tile_size):          # tile row
    for jj in range(0, n, tile_size):      # tile column
        for kk in range(0, n, tile_size):  # tile k-dimension
            for i in range(ii, min(ii + tile_size, n)):
                for j in range(jj, min(jj + tile_size, n)):
                    for k in range(kk, min(kk + tile_size, n)):
                        C[i][j] += A[i][k] * B[k][j]
```

Key observations:
- The outer three loops step by `tile_size`.
- The inner three loops iterate within the current tile.
- The kk loop is the **reduction** dimension — it can be outer or middle, but it must
  accumulate partial sums across tiles.

### Partial Credit (10-15 pts)
- Uses tiling but with a non-standard loop ordering that still produces correct results.
- Uses only 4-5 loops (partially tiled).

### No Credit (0 pts)
- No tiling structure; just rearranged the naive loops.

---

## 3. Handles Non-Tile-Aligned Matrix Sizes (15 points)

### Full Credit (15 pts)
- Uses `min(ii + tile_size, n)` (or equivalent) for inner loop bounds.
- Correctly handles cases where `N % tile_size != 0`.
- Tested with non-aligned sizes (e.g., N=100, tile_size=32).

### Partial Credit (7-10 pts)
- Handles it but with an awkward separate pass for the "remainder" tiles.

### No Credit (0 pts)
- Only works when `N` is divisible by `tile_size`.
- Crashes or produces wrong results for non-aligned sizes.

---

## 4. Meaningful Benchmarking (10 points)

### Full Credit (10 pts)
- Times both implementations across multiple matrix sizes.
- Reports wall-clock times and calculates speedup ratios.
- Verifies correctness for each benchmark case.
- Tests multiple tile sizes.
- Clean, readable output format.

### Partial Credit (5-7 pts)
- Basic timing but missing correctness verification or speedup calculation.

### No Credit (0 pts)
- No benchmarking implemented.

---

## 5. Discussion of Why Tiling Helps (15 points)

### Full Credit (15 pts)
The candidate can clearly explain:
- **Row-major storage**: In `B[k][j]`, incrementing `k` jumps to a different row, which
  may be in a different cache line or even a different page.
- **Cache lines**: CPU caches load data in fixed-size blocks (typically 64 bytes). Accessing
  B column-wise wastes most of each loaded cache line.
- **Spatial locality**: Tiling ensures that when a cache line is loaded, most/all of its
  elements are used before eviction.
- **Temporal locality**: Tiling ensures that elements of A and B are reused multiple times
  while still in cache, rather than being evicted and reloaded.
- **Working set**: Each tile should fit in L1 or L2 cache. Three tiles of size T*T with
  8-byte doubles require `3 * T^2 * 8` bytes. For T=32: 24 KB, fits in most L1 caches.

### Partial Credit (7-10 pts)
- Mentions "cache" or "locality" but cannot give a detailed explanation.

### No Credit (0 pts)
- Cannot explain why tiling helps.

---

## 6. Bonus: Tile Size Experimentation (10 points)

### Full Credit (10 pts)
- Experiments with multiple tile sizes (e.g., 8, 16, 32, 64, 128).
- Discusses the tradeoff: too small = too much loop overhead; too large = doesn't fit in cache.
- Relates optimal tile size to L1 cache size (typically 32-64 KB).
- Notes that the optimal tile size for pure Python may differ from C/Fortran due to
  interpreter overhead and Python object representation.

### Partial Credit (5-7 pts)
- Tests a few tile sizes but limited discussion.

### No Credit (0 pts)
- Only uses default tile_size=32 with no experimentation.

---

## Red Flags
- Using NumPy or other libraries for the multiplication itself.
- Not understanding why the inner loop order matters.
- Claiming tiling changes the algorithmic complexity (it's still O(N^3)).
- Unable to connect this to real-world GPU programming concepts.

## Green Flags
- Mentions loop interchange optimization (ikj vs ijk ordering for the naive case).
- Discusses how this maps to CUDA shared memory tiling.
- Mentions that Python's overhead makes the speedup less dramatic than in C.
- Discusses Strassen's or other sub-cubic algorithms as further optimizations.
- Considers prefetching or memory alignment.
