# Follow-Up Questions: GPU Kernel Simulator

---

## 1. Why does the padding trick work? Draw out the bank mapping with and without padding.

**Expected answer:**

Without padding (stride = 32, which equals num_banks):
```
shared[0][col] -> bank = (0 * 32 + col) % 32 = col
shared[1][col] -> bank = (1 * 32 + col) % 32 = col
shared[2][col] -> bank = (2 * 32 + col) % 32 = col
...
```
All rows in the same column map to the SAME bank -> 32-way bank conflict on column access.

With padding (stride = 33, coprime to 32):
```
shared[0][col] -> bank = (0 * 33 + col) % 32 = col % 32
shared[1][col] -> bank = (1 * 33 + col) % 32 = (col + 1) % 32
shared[2][col] -> bank = (2 * 33 + col) % 32 = (col + 2) % 32
...
```
Each row maps to a DIFFERENT bank -> no bank conflicts.

The key mathematical insight: if `stride` and `num_banks` are coprime (gcd = 1), then `(row * stride) % num_banks` cycles through all bank values as row varies. This is a consequence of modular arithmetic -- multiplying by a number coprime to the modulus generates all residues.

---

## 2. What is warp divergence and how would you model it in this simulator?

**Expected answer:**

Warp divergence occurs when threads in the same warp take different branches in an if/else statement. Since all 32 threads must execute in lockstep (SIMT), the warp must execute BOTH branches, masking off threads that don't take each path. This effectively doubles (or more) the execution time for divergent code.

To model it in the simulator:
- Track an "active mask" per warp (32-bit bitmask of active threads)
- At branch points, split the warp: execute the taken path with one mask, then the not-taken path with the complement mask
- Count total instruction slots executed (including masked-off slots as wasted cycles)
- The penalty = (instructions on path A) + (instructions on path B) instead of max(A, B)
- On newer architectures (Volta+), independent thread scheduling allows some mitigation, but divergence is still costly

---

## 3. How does the memory coalescing pattern differ between NVIDIA's older and newer architectures?

**Expected answer:**

- **Compute Capability 1.x (Tesla)**: Very strict coalescing rules. Threads had to access consecutive, aligned addresses in a specific pattern. Misaligned or non-sequential accesses fell back to per-thread transactions.

- **Compute Capability 2.x (Fermi)**: Introduced L1/L2 cache. Coalescing became more forgiving -- the hardware issues 128-byte cache line requests and any access within a line is served. Scattered accesses within the same line are fine.

- **Compute Capability 3.x+ (Kepler, Maxwell, Pascal)**: Further relaxed. 32-byte sectors within 128-byte lines. Only the sectors actually accessed are fetched, reducing wasted bandwidth for partial line usage.

- **Compute Capability 7.x+ (Volta, Ampere, Hopper)**: Global memory access goes through L2 cache (configurable L1). Sector-based access (32 bytes). Hardware handles complex access patterns more gracefully. Additionally, async copy (cp.async) can bypass L1 for shared memory loads.

The trend: newer architectures are more forgiving of non-ideal access patterns, but coalesced access is STILL significantly faster because it minimizes the number of memory transactions.

---

## 4. In Flash Attention, how is shared memory used to avoid materializing the full attention matrix?

**Expected answer:**

Flash Attention (Dao et al., 2022) computes attention without ever materializing the full N x N attention matrix in HBM (global memory). The key idea:

1. **Tiled computation**: The Q, K, V matrices are loaded in tiles into shared memory (SRAM).
2. **Online softmax**: Instead of computing the full softmax over all keys, FlashAttention uses an online softmax algorithm that maintains running statistics (max and sum of exponentials) as it processes tiles of K.
3. **Memory hierarchy exploitation**:
   - Shared memory (SRAM): ~20 TB/s bandwidth, ~100 KB per SM
   - HBM: ~2 TB/s bandwidth, ~80 GB total
   - By keeping intermediate results in shared memory and only writing final results to HBM, Flash Attention reduces memory I/O from O(N^2) to O(N^2 / SRAM_size) in terms of HBM reads.
4. **No materialization**: The N x N attention score matrix and the N x N softmax output are never fully stored -- only tile-sized chunks exist in shared memory at any time.
5. **Backward pass**: Uses recomputation instead of storing intermediate results, trading compute for memory.

This achieves both reduced memory usage (O(N) instead of O(N^2)) and faster wall-clock time due to reduced HBM traffic.

---

## 5. What is occupancy and how does shared memory usage affect it?

**Expected answer:**

**Occupancy** = (number of active warps on an SM) / (maximum warps the SM supports)

An SM has limited resources that are partitioned among resident thread blocks:
- **Shared memory**: Total shared memory per SM is fixed (e.g., 48-228 KB depending on GPU and configuration). Each block claims its shared memory allocation. More shared memory per block -> fewer blocks can be resident.
- **Registers**: Each thread uses registers; total registers per SM are limited. More registers per thread -> fewer warps.
- **Thread block slots**: Maximum number of blocks per SM (e.g., 16-32).
- **Warp slots**: Maximum warps per SM (e.g., 48-64).

Impact of shared memory on occupancy:
- If a block uses 48 KB of shared memory and the SM has 96 KB, only 2 blocks can be resident.
- If a block uses 16 KB, 6 blocks could fit (subject to other limits).
- The padding trick for transpose INCREASES shared memory usage by ~3% (33 vs 32 elements per row), which is negligible, but in tight situations it could reduce occupancy.
- Higher occupancy helps hide latency (more warps available to execute while others wait for memory), but is not always necessary -- sometimes a kernel is compute-bound and lower occupancy is fine.

---

## 6. How would you extend this simulator to model register pressure and spilling?

**Expected answer:**

Register pressure occurs when a kernel needs more registers per thread than available, causing "register spilling" -- some variables are stored in local memory (actually global memory, very slow).

Extensions to the simulator:
1. **Register file model**: Track total registers per SM (e.g., 65536 for modern GPUs) and registers per thread.
2. **Occupancy calculator**: `max_warps = min(max_warps_per_sm, total_registers / (regs_per_thread * warp_size))`. This interacts with shared memory limits.
3. **Spill detection**: If `regs_per_thread > max_registers_per_thread` (typically 255), mark excess variables as "spilled" to local memory.
4. **Spill cost model**: Each spilled access adds a local memory transaction (same latency as global memory, but typically cached in L1/L2). Model this as additional memory traffic.
5. **Trade-off analysis**: Allow the user to vary registers per thread and see the impact on occupancy vs spill overhead. Sometimes accepting lower occupancy (more registers) is better than spilling.
6. **Launch bounds**: Model `__launch_bounds__` hint that tells the compiler the max threads per block, allowing it to use more registers per thread without worrying about occupancy.
