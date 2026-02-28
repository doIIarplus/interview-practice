# Rubric: GPU Kernel Simulator

**Total: 100 points**

---

## 1. Bank Conflict Computation with Broadcast Handling (25 points)

### Full marks (25):
- Correctly maps addresses to banks via `(address // 4) % num_banks`
- Groups accesses by bank
- Within each bank, identifies **distinct addresses** (deduplicates for broadcast)
- Computes conflicts per bank as `num_distinct_addresses - 1`
- Sums across all banks for total conflicts
- Handles edge cases:
  - All threads accessing same address (broadcast) -> 0 conflicts
  - All threads accessing same bank, different addresses -> 31 conflicts
  - Mixed broadcast + distinct addresses in the same bank

### Partial credit:
- (15) Correct bank mapping and conflict counting but misses broadcast special case
- (10) Correct bank mapping but wrong conflict formula (e.g., counting total accesses instead of distinct - 1)
- (5) Understands the concept but implementation has bugs

### Common mistakes to watch for:
- Not handling broadcast (treating same-address accesses as conflicts)
- Off-by-one: returning N instead of N-1 for a bank with N distinct addresses
- Confusing byte addresses with word addresses
- Not handling the case where addresses list length != warp_size

---

## 2. Memory Coalescing Analysis (15 points)

### Full marks (15):
- Correctly computes cache line index for each address: `address // cache_line_bytes`
- Counts unique cache lines touched
- Defines "perfectly coalesced" as all 32 threads fitting in exactly 1 cache line (32 * 4 = 128 bytes)
- Returns correct tuple `(bool, int)`

### Partial credit:
- (10) Correct cache line counting but wrong definition of "perfectly coalesced"
- (5) Understands concept but implementation has issues

### Key insight the candidate should demonstrate:
- Understanding that coalescing is about spatial locality in global memory
- The relationship between cache line size and access pattern

---

## 3. Correct Naive Transpose with Shared Memory Simulation (20 points)

### Full marks (20):
- Correctly divides matrix into tiles based on `block_dim`
- Handles non-tile-aligned matrices (partial tiles at edges)
- Correctly loads tile from global memory into simulated shared memory (row-major)
- Correctly reads from shared memory in transposed order (column-major)
- Simulates bank conflicts during the column-major read:
  - For a 32-wide tile, thread (tx, ty) reads `shared[tx][ty]`
  - Address in shared memory: `(tx * tile_cols + ty) * 4`
  - Bank: `(tx * tile_cols + ty) % 32`
  - When `tile_cols = 32`, bank = `(tx * 32 + ty) % 32 = ty` -- so all threads in a row of tx values with fixed ty hit the same bank
- Produces the correct transposed matrix
- Correct stats dict

### Partial credit:
- (15) Correct transpose but bank conflict simulation is wrong or missing
- (10) Correct for square tile-aligned matrices but breaks on non-aligned
- (5) Transpose logic is incorrect

### Architecture note:
- The candidate should explain WHY column-major access from shared memory causes conflicts: when `tile_cols` is a multiple of 32, `shared[row][fixed_col]` maps all rows to the same bank.

---

## 4. Padded Transpose Eliminating Bank Conflicts (20 points)

### Full marks (20):
- Uses `tile_cols + 1` as the row stride for shared memory (the padding column)
- Correctly computes bank mapping with padding:
  - Address: `(row * (tile_cols + 1) + col) * 4`
  - Bank: `(row * (tile_cols + 1) + col) % 32`
  - When `tile_cols = 32`: bank = `(row * 33 + col) % 32`
  - Since `gcd(33, 32) = 1`, consecutive rows map to different banks for the same column
- Reports 0 (or near-0) bank conflicts
- Produces the correct transposed matrix (identical to naive)
- Handles non-aligned matrices

### Partial credit:
- (15) Correct transpose result but bank conflict counting with padding is wrong
- (10) Padding is applied but not correctly simulated
- (5) Understands the concept but implementation fails

---

## 5. Understanding of WHY Padding Eliminates Bank Conflicts (10 points)

*Assessed through code comments, variable naming, or verbal explanation.*

### Full marks (10):
- Can explain: without padding, `shared[row][col]` with stride 32 means bank = `(row * 32 + col) % 32 = col % 32`. For a fixed column, all rows map to the same bank.
- With padding (stride 33): bank = `(row * 33 + col) % 32`. Since 33 is coprime to 32, incrementing row by 1 shifts the bank by 1. All 32 rows of a column map to 32 different banks.
- Can generalize: the trick works whenever `stride` and `num_banks` are coprime.

### Partial credit:
- (5) Vague understanding ("padding shifts the banks") without precise mathematical reasoning
- (0) Cannot explain why padding helps

---

## 6. Code Clarity and Edge Cases (10 points)

### Full marks (10):
- Clean, readable code with meaningful variable names
- Handles edge cases:
  - Non-square matrices
  - Matrices smaller than a single tile
  - Single-row or single-column matrices
  - Empty address list (defensive programming)
- Uses appropriate data structures (defaultdict, sets for deduplication)
- Helper methods are well-factored

### Partial credit:
- (7) Generally clean but misses some edge cases
- (4) Works but hard to follow or poorly structured
- (2) Messy code that mostly works

---

## Bonus Observations (not scored, but positive signals):

- Mentions that real GPUs have warp shuffle instructions that can avoid shared memory entirely for some operations
- Discusses how occupancy (number of concurrent warps) is affected by shared memory usage per block
- Notes that padding wastes shared memory and could reduce occupancy
- Mentions that CUDA 11+ has async copy instructions (cp.async) that bypass shared memory for some patterns
- Discusses how bank width differs across GPU architectures (4 bytes on most, 8 bytes on some configurations)
- Mentions that memory coalescing rules differ between compute capabilities (CC 2.x vs 3.x+)

---

## Red Flags:

- Cannot explain what a bank conflict is conceptually
- Does not understand the difference between shared memory and global memory
- Confuses byte addresses with element indices
- Cannot explain why the naive transpose has conflicts (does not understand the column-major access pattern)
- Uses NumPy or external libraries (question specifies stdlib only)
- Transpose produces incorrect results for non-square matrices
