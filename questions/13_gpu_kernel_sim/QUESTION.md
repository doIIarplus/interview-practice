# Question 13: GPU Kernel Simulator

## Overview

You are building a simplified simulator for a GPU execution model to reason about performance characteristics. The simulator models a single **Streaming Multiprocessor (SM)** executing a kernel launch.

Understanding the GPU execution model -- thread blocks, warps, shared memory bank conflicts, and coalesced memory access -- is fundamental to writing high-performance GPU code.

## Background

### GPU Execution Model

A **Streaming Multiprocessor (SM)** is the fundamental compute unit on a GPU. When a kernel is launched, **thread blocks** are assigned to SMs for execution.

- **Thread blocks**: Groups of threads (up to 1024) assigned to an SM. All threads in a block can cooperate via shared memory and synchronization barriers.
- **Warps**: Groups of 32 threads that execute in lockstep (SIMT -- Single Instruction, Multiple Threads). A thread block is divided into warps. Warp 0 contains threads 0-31, warp 1 contains threads 32-63, etc.
- **Shared memory**: Fast on-chip memory (48 KB typical) shared by all threads in a block. Organized into **banks** to enable parallel access.

### Shared Memory Banks

Shared memory is divided into 32 banks, each 4 bytes wide. The bank for a given byte address is:

```
bank = (address // 4) % num_banks
```

When multiple threads in a warp access **different addresses** in the **same bank**, these accesses are serialized -- this is a **bank conflict**. If N threads hit the same bank (at different addresses), it takes N serialized accesses instead of 1.

**Special case -- Broadcast**: If multiple threads access the **exact same address**, the hardware broadcasts the value to all requesting threads with no conflict.

### Memory Coalescing

Global memory (DRAM) is accessed in large cache-line-sized transactions (typically 128 bytes). When threads in a warp access a contiguous aligned region, the hardware **coalesces** these into minimal transactions. Scattered accesses waste bandwidth.

---

## Task

Implement a `GPUSimulator` class with the following methods:

### 1. Constructor

```python
def __init__(self, shared_mem_kb: int = 48, num_banks: int = 32, warp_size: int = 32)
```

### 2. Bank Conflict Analysis

```python
def bank_conflict_count(self, addresses: list[int]) -> int
```

Given a list of 32 memory addresses (one per thread in a warp) for a shared memory access, compute the number of **bank conflicts** (extra serialized cycles).

- Determine which bank each address maps to: `bank = (address // 4) % num_banks`
- Group accesses by bank.
- Within each bank group, accesses to the **exact same address** are broadcast (free -- counted as 1 access).
- If a bank has K distinct addresses accessed by threads, it requires K serialized accesses. The extra cycles (conflicts) for that bank = K - 1.
- Return the **total** extra serialized cycles across all banks.

### 3. Memory Coalescing Analysis

```python
def is_coalesced(self, addresses: list[int], cache_line_bytes: int = 128) -> tuple[bool, int]
```

Given 32 memory addresses for a warp's global memory access:
- Determine how many cache lines are touched. A cache line starting at address `A` covers bytes `[A, A + cache_line_bytes)`, where `A` is aligned to `cache_line_bytes`.
- Return `(is_perfectly_coalesced, num_cache_lines)`.
- **Perfectly coalesced** means all 32 threads access consecutive 4-byte elements fitting within the minimal number of cache lines. For 32 threads x 4 bytes = 128 bytes, this is exactly 1 cache line.

### 4. Naive Matrix Transpose with Shared Memory

```python
def simulate_transpose(self, matrix: list[list[float]], block_dim: tuple[int, int] = (32, 32)) -> tuple[list[list[float]], dict]
```

Simulate a matrix transpose using shared memory tiling (a classic GPU optimization pattern):

1. Divide the matrix into tiles of size `block_dim`.
2. For each tile:
   - **Load** from global memory row-by-row into a shared memory tile (coalesced reads).
   - **Store** from shared memory column-by-column to the output matrix at the transposed position (coalesced writes).
3. Simulate the shared memory access patterns and count bank conflicts.
   - When loading into shared memory: thread `(tx, ty)` writes to `shared[ty][tx]` -- row-major write, no conflicts.
   - When reading from shared memory for the transpose: thread `(tx, ty)` reads `shared[tx][ty]` -- column-major read, which causes bank conflicts because consecutive threads access the same bank.

Return the transposed matrix and a stats dictionary:
```python
{"bank_conflicts": int, "global_mem_transactions": int, "tiles_processed": int}
```

Handle non-tile-aligned matrices by processing partial tiles at the edges.

### 5. Padded Transpose (Optimization)

```python
def simulate_transpose_padded(self, matrix: list[list[float]], block_dim: tuple[int, int] = (32, 32)) -> tuple[list[list[float]], dict]
```

Same as above, but use the **padding trick**: allocate shared memory tiles with an extra column, i.e., dimensions `block_dim[0] x (block_dim[1] + 1)`.

When reading column-major from the padded layout, consecutive threads no longer map to the same bank, eliminating bank conflicts.

Return the transposed matrix and stats. Bank conflicts should be **0** (or near-0).

---

## Examples

```python
sim = GPUSimulator()

# --- Bank Conflict Examples ---

# Sequential word access: thread i accesses word i -> each bank accessed once
addresses = [i * 4 for i in range(32)]
print(sim.bank_conflict_count(addresses))  # 0

# All threads access addresses that map to bank 0 (stride of 128 bytes = 32 banks * 4)
addresses = [i * 128 for i in range(32)]
print(sim.bank_conflict_count(addresses))  # 31

# Broadcast: all threads access the exact same address -> free
addresses = [0] * 32
print(sim.bank_conflict_count(addresses))  # 0

# Stride-2 access: hits only 16 of 32 banks, 2 threads per bank
addresses = [i * 8 for i in range(32)]
print(sim.bank_conflict_count(addresses))  # 16 (16 banks each with 2 accesses -> 16 extra)

# --- Coalescing Examples ---

# Perfectly coalesced: 32 consecutive 4-byte accesses
addresses = [i * 4 for i in range(32)]
coalesced, lines = sim.is_coalesced(addresses)
print(coalesced, lines)  # True 1

# Strided access touching many cache lines
addresses = [i * 512 for i in range(32)]
coalesced, lines = sim.is_coalesced(addresses)
print(coalesced, lines)  # False 128

# --- Transpose Example ---
matrix = [[float(r * 4 + c) for c in range(4)] for r in range(4)]
# [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

transposed, stats = sim.simulate_transpose(matrix, block_dim=(4, 4))
# transposed = [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
print(stats["tiles_processed"])  # 1
```

## Constraints

- All addresses are non-negative integers representing byte addresses.
- Matrix elements are floats.
- `block_dim` elements are positive integers.
- You may assume the matrix is non-empty.
- Handle matrices that are not evenly divisible by `block_dim`.

## Notes

- Focus on correctness of the simulation, not on actual GPU execution.
- Use only the Python standard library (no NumPy or other external packages).
- Think carefully about the bank mapping for the padded case and convince yourself (and explain) why the padding eliminates conflicts.
