# GPU Architecture — A Primer for CPU Programmers

This guide explains GPU architecture from the ground up, assuming you understand
CPU programming but have never written GPU code. It covers the CUDA programming
model, memory hierarchy, and key performance concepts needed for questions Q00,
Q07, Q13, Q16, and Q18.

---

## Table of Contents

1. [CPU vs GPU: The Big Picture](#cpu-vs-gpu-the-big-picture)
2. [The CUDA Programming Model](#the-cuda-programming-model)
3. [Thread Hierarchy: Grid, Blocks, Threads](#thread-hierarchy-grid-blocks-threads)
4. [Warps: The Real Unit of Execution](#warps-the-real-unit-of-execution)
5. [GPU Memory Hierarchy](#gpu-memory-hierarchy)
6. [Memory Coalescing](#memory-coalescing)
7. [Shared Memory and Bank Conflicts](#shared-memory-and-bank-conflicts)
8. [Occupancy](#occupancy)
9. [Warp Divergence](#warp-divergence)
10. [Putting It All Together: Matrix Multiply on GPU](#putting-it-all-together-matrix-multiply-on-gpu)
11. [Modern GPU Specs (A100 / H100)](#modern-gpu-specs-a100--h100)
12. [Key Takeaways for Interviews](#key-takeaways-for-interviews)

---

## CPU vs GPU: The Big Picture

A CPU has a few powerful cores optimized for **sequential performance**:
complex branch prediction, out-of-order execution, large caches.

A GPU has thousands of simple cores optimized for **throughput**: simple
in-order execution, small caches, but massive parallelism.

```
CPU (e.g., 16-core server CPU):
┌────────────────────────────────────────────┐
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │
│  │ Core │  │ Core │  │ Core │  │ Core │   │   16 powerful cores
│  │ OoO  │  │ OoO  │  │ OoO  │  │ OoO  │   │   Optimized for latency
│  │ +big │  │ +big │  │ +big │  │ +big │   │   Large caches
│  │cache │  │cache │  │cache │  │cache │   │   Complex control logic
│  └──────┘  └──────┘  └──────┘  └──────┘   │
│            ... (16 total) ...              │
│         ┌──────────────────┐               │
│         │  Large L3 Cache  │  ~32 MB       │
│         └──────────────────┘               │
└────────────────────────────────────────────┘

GPU (e.g., A100):
┌─────────────────────────────────────────────────────────┐
│ SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM │
│ SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM │
│ SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM │
│ SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM SM │
│  ... 108 Streaming Multiprocessors (SMs) total ...      │
│                                                         │
│  Each SM has:                                           │
│    - 64 FP32 cores (simple ALUs)                        │
│    - 32 FP64 cores                                      │
│    - 4 Tensor Cores (matrix multiply units)             │
│    - 256 KB register file                               │
│    - 192 KB configurable shared memory / L1 cache       │
│    - Up to 2048 concurrent threads (64 warps)           │
│                                                         │
│              ┌──────────────────┐                        │
│              │   40 MB L2 Cache │                        │
│              └──────────────────┘                        │
│              ┌──────────────────┐                        │
│              │  80 GB HBM2e     │  2 TB/s bandwidth     │
│              └──────────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

### The key trade-off

| Property | CPU | GPU |
|----------|-----|-----|
| Cores | 8-64 | 1000s of simple cores |
| Clock speed | 3-5 GHz | 1-2 GHz |
| Per-thread performance | Very high | Low |
| Throughput (parallel work) | Moderate | Very high |
| Memory bandwidth | ~50-100 GB/s | ~2-3 TB/s |
| Cache per core | Large (32KB L1) | Small, shared |
| Best for | Sequential, branchy code | Massively parallel, uniform work |

**Key insight:** GPUs are not faster per thread. They are faster in aggregate
because they run thousands of threads simultaneously and hide memory latency
through massive parallelism.

---

## The CUDA Programming Model

CUDA is NVIDIA's programming model for GPUs. You write a **kernel** (a function
that runs on the GPU), and launch it with thousands of threads.

### A simple kernel (conceptual pseudocode)

```c
// This function runs on the GPU, once per thread
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Unique thread ID
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // Each thread adds one element
    }
}

// Launch: 256 threads per block, enough blocks to cover N elements
int blocks = (N + 255) / 256;
vector_add<<<blocks, 256>>>(A, B, C, N);
```

### Python equivalent (for understanding)

```python
def vector_add_cpu(A, B, C, N):
    """CPU version: sequential."""
    for idx in range(N):
        C[idx] = A[idx] + B[idx]

def vector_add_gpu_simulated(A, B, C, N, block_size=256):
    """Simulating what the GPU does: all threads run 'simultaneously'."""
    num_blocks = (N + block_size - 1) // block_size
    for block_id in range(num_blocks):          # Blocks run on different SMs
        for thread_id in range(block_size):      # Threads in block run together
            idx = block_id * block_size + thread_id
            if idx < N:
                C[idx] = A[idx] + B[idx]
```

The GPU runs all these iterations **in parallel** (not sequentially in a loop).

---

## Thread Hierarchy: Grid, Blocks, Threads

CUDA organizes threads in a three-level hierarchy:

```
                         Grid
            ┌──────────────────────────────┐
            │                              │
            │   Block(0,0)    Block(1,0)   │    Grid = all blocks
            │  ┌──────────┐ ┌──────────┐   │    launched by one
            │  │ Thread    │ │ Thread   │   │    kernel call
            │  │ (0,0)    │ │ (0,0)    │   │
            │  │ (0,1)    │ │ (0,1)    │   │
            │  │ (1,0)    │ │ (1,0)    │   │
            │  │ (1,1)    │ │ (1,1)    │   │
            │  │ ...       │ │ ...      │   │
            │  └──────────┘ └──────────┘   │
            │                              │
            │   Block(0,1)    Block(1,1)   │
            │  ┌──────────┐ ┌──────────┐   │
            │  │ ...       │ │ ...      │   │
            │  └──────────┘ └──────────┘   │
            │                              │
            └──────────────────────────────┘

Grid dimensions:  gridDim.x, gridDim.y, gridDim.z   (up to 3D)
Block dimensions: blockDim.x, blockDim.y, blockDim.z (up to 3D)
```

### Key rules

1. **Threads in the same block** can synchronize and share data via shared
   memory. Threads in different blocks cannot (easily) communicate.

2. **Blocks are independent.** The GPU can execute blocks in any order, on any
   SM. This is how GPUs scale — more SMs = more blocks running simultaneously.

3. **Block size** is typically 128, 256, or 512 threads. Maximum is usually
   1024 threads per block.

4. **A unique thread ID** is computed from block and thread indices:

```
1D: global_id = blockIdx.x * blockDim.x + threadIdx.x

2D: global_x = blockIdx.x * blockDim.x + threadIdx.x
    global_y = blockIdx.y * blockDim.y + threadIdx.y
```

### Mapping to hardware

- Each **block** is assigned to one **SM** (Streaming Multiprocessor).
- Multiple blocks can run on the same SM simultaneously (if resources allow).
- The SM schedules threads from its assigned blocks.

---

## Warps: The Real Unit of Execution

While you program in terms of threads and blocks, the GPU actually executes
threads in groups of **32** called **warps**.

```
Block of 256 threads:
┌───────────────────────────────────────────────────────────────┐
│ Warp 0     Warp 1     Warp 2     Warp 3     ... Warp 7      │
│ T0-T31    T32-T63    T64-T95    T96-T127       T224-T255    │
│                                                              │
│ All 32 threads in a warp execute the SAME instruction        │
│ at the SAME time (SIMT = Single Instruction, Multiple       │
│ Threads).                                                    │
└───────────────────────────────────────────────────────────────┘
```

### SIMT execution

**SIMT** (Single Instruction, Multiple Threads) means that all 32 threads in a
warp execute the same instruction simultaneously, but each thread operates on
its own data (its own registers, its own memory addresses).

This is similar to SIMD (Single Instruction, Multiple Data) on CPUs, but more
flexible:
- CPU SIMD: one instruction processes a fixed-size vector (e.g., 8 floats)
- GPU SIMT: 32 threads can have different addresses, and can even diverge (see
  warp divergence below)

### Why warps matter

1. **Memory access:** All 32 threads in a warp access memory simultaneously.
   If they access adjacent addresses, the hardware combines them into one
   transaction (coalescing).

2. **Branching:** If threads in a warp take different branches, the warp must
   execute both paths (warp divergence).

3. **Synchronization:** Threads within a warp are implicitly synchronized (they
   execute in lockstep). Threads across warps need explicit synchronization.

---

## GPU Memory Hierarchy

```
Per-Thread:
┌─────────────────────┐
│     Registers       │  ~256 KB per SM, split among threads
│  Fastest, private   │  Latency: 0 cycles (operand)
│  to each thread     │  Bandwidth: highest
└─────────┬───────────┘
          │
Per-Block:
┌─────────┴───────────┐
│   Shared Memory     │  Up to ~100-164 KB per SM (configurable)
│  Programmer-managed │  Latency: ~20-30 cycles
│  Shared by all      │  Bandwidth: ~1.5 TB/s per SM
│  threads in a block │  Divided into 32 banks
└─────────┬───────────┘
          │
Per-SM:
┌─────────┴───────────┐
│    L1 Cache         │  Often shares physical SRAM with shared memory
│  Hardware-managed   │  Latency: ~30 cycles
└─────────┬───────────┘
          │
Per-GPU:
┌─────────┴───────────┐
│    L2 Cache         │  ~40 MB (A100)
│  Hardware-managed   │  Latency: ~200 cycles
└─────────┬───────────┘
          │
┌─────────┴───────────┐
│   Global Memory     │  40-80 GB HBM
│  (HBM / VRAM)       │  Latency: ~400-800 cycles
│  Accessible by all  │  Bandwidth: ~2 TB/s (A100)
│  threads            │  This is the "main memory" of the GPU
└─────────────────────┘
```

### Memory types summary

| Memory | Scope | Lifetime | Size | Speed | Programmer Control |
|--------|-------|----------|------|-------|--------------------|
| Registers | Per-thread | Thread | ~255 per thread | Fastest | Automatic |
| Shared Memory | Per-block | Block | ~100 KB / SM | Very fast | Explicit (__shared__) |
| L1 Cache | Per-SM | Automatic | Configurable | Fast | Indirect |
| L2 Cache | Per-GPU | Automatic | ~40 MB | Medium | Indirect |
| Global Memory | Per-GPU | Application | 40-80 GB | Slow | Explicit (malloc) |
| Constant Memory | Per-GPU | Application | 64 KB | Fast (cached) | Explicit |

### The key pattern: load from global, compute in shared

The most common optimization pattern in GPU programming:

```
1. Load a tile of data from global memory into shared memory
   (cooperative: each thread loads a few elements)
2. __syncthreads()  — barrier to ensure all data is loaded
3. Compute using data in shared memory (fast access, no global memory traffic)
4. __syncthreads()  — barrier to ensure computation is done
5. Write results back to global memory
```

This is directly analogous to tiling/blocking for CPU caches, but the
programmer must manage it explicitly.

---

## Memory Coalescing

**Coalescing** means combining memory accesses from threads in a warp into a
minimum number of memory transactions.

### Coalesced access (good)

```
Thread:     T0    T1    T2    T3    ...   T31
Address:   [0]   [1]   [2]   [3]   ...  [31]

Each thread accesses the next consecutive address.
Hardware combines into ONE 128-byte transaction.
                    ┌──────────────────────────────────┐
Transaction:        │  addresses 0-31 (128 bytes)      │
                    └──────────────────────────────────┘
Bandwidth utilization: 100%
```

### Strided access (bad)

```
Thread:     T0    T1    T2    T3    ...   T31
Address:   [0]  [128] [256] [384]  ... [3968]

Each thread accesses every 128th element (stride = 128).
Each access is in a different 128-byte cache sector.
Hardware issues 32 SEPARATE transactions.

Transaction 1: │ addr 0-31     │  (only 4 bytes used out of 128!)
Transaction 2: │ addr 128-159  │  (only 4 bytes used!)
Transaction 3: │ addr 256-287  │  (only 4 bytes used!)
...
Bandwidth utilization: 4/128 = 3.1%  ← terrible!
```

### Practical rule

**Adjacent threads should access adjacent memory addresses.**

```
GOOD: data[threadIdx.x]              — stride 1, coalesced
GOOD: data[blockIdx.x * blockDim.x + threadIdx.x]  — still stride 1
BAD:  data[threadIdx.x * N]          — stride N, uncoalesced
BAD:  data[threadIdx.x * 32]         — stride 32, partially uncoalesced
```

### Structure of Arrays vs Array of Structures

```
Array of Structures (AoS) — BAD for GPU:
  particles[i].x, particles[i].y, particles[i].z
  Memory: [x0 y0 z0] [x1 y1 z1] [x2 y2 z2] ...

  If all threads read .x: strided access (stride = 3)

Structure of Arrays (SoA) — GOOD for GPU:
  x[i], y[i], z[i]
  Memory: [x0 x1 x2 ...] [y0 y1 y2 ...] [z0 z1 z2 ...]

  If all threads read x: coalesced access (stride = 1)
```

---

## Shared Memory and Bank Conflicts

Shared memory is divided into **32 banks** (same as the number of threads in a
warp). Each bank can service one request per cycle.

### Bank assignment

Addresses are assigned to banks in an interleaved fashion:

```
Bank:     0     1     2     3    ...   31    0     1    ...
Address: [0-3] [4-7] [8-11][12-15]... [124-127][128-131] ...

Rule: bank = (address / 4) % 32
  (4 bytes = one 32-bit word per bank slot)
```

### No conflict: each thread accesses a different bank

```
Thread 0 → Bank 0     ┐
Thread 1 → Bank 1     │  No conflict!
Thread 2 → Bank 2     ├  All 32 accesses served simultaneously
...                    │  in ONE cycle
Thread 31 → Bank 31   ┘

Example: shared_mem[threadIdx.x]  — stride 1, all different banks
```

### 2-way bank conflict: two threads access the same bank

```
Thread 0 → Bank 0     ┐
Thread 1 → Bank 2     │  Thread 0 and Thread 16 both access Bank 0!
...                    │  These two accesses are SERIALIZED.
Thread 16 → Bank 0    │  Takes 2 cycles instead of 1.
...                    │
Thread 31 → Bank 30   ┘

Example: shared_mem[threadIdx.x * 2]  — stride 2, every other bank used,
         threads 0 and 16 both hit bank 0
```

### Worst case: 32-way bank conflict

```
All 32 threads → Bank 0   — serialized into 32 cycles!

Example: shared_mem[threadIdx.x * 32]  — all threads access bank 0
```

### Avoiding bank conflicts

- **Stride-1 access:** `shared_mem[threadIdx.x]` — no conflicts
- **Padding:** Add an extra element per row to shift bank assignments
  ```
  // Without padding: row width = 32, stride = 32 → all same bank
  __shared__ float tile[32][32];    // Column access has 32-way conflict

  // With padding: row width = 33, stride = 33 → different banks
  __shared__ float tile[32][33];    // Column access has no conflict!
  ```

---

## Occupancy

**Occupancy** = (active warps per SM) / (maximum warps per SM)

Each SM can support a maximum number of concurrent warps (e.g., 64 warps = 2048
threads on A100). Occupancy measures how many of those slots are actually used.

### Why occupancy matters

GPUs hide memory latency by switching between warps. When one warp is waiting
for a memory access, the SM executes another warp. More active warps = more
opportunities to hide latency.

```
Low occupancy (few active warps):
  Warp A: [compute][----wait for memory----][compute]
  Warp B: .........[compute][----wait for memory----]
  SM:     [  busy  ][    IDLE    ][  busy  ][  IDLE  ]

High occupancy (many active warps):
  Warp A: [compute][----wait----][compute]
  Warp B: ........[compute][----wait----][compute]
  Warp C: ................[compute][----wait----]
  Warp D: ........................[compute][----wait----]
  SM:     [  busy  ][  busy  ][  busy  ][  busy  ][ busy ]
                     Always a warp ready to run!
```

### What limits occupancy

Occupancy is limited by **resource usage per block**:

1. **Registers per thread:** Each SM has a fixed register file (~65536
   registers). If each thread uses 64 registers, you can have 65536/64 = 1024
   threads = 32 warps (out of 64 max → 50% occupancy).

2. **Shared memory per block:** Each SM has ~100 KB. If each block uses 48 KB,
   you can fit 2 blocks per SM.

3. **Threads per block:** Maximum 1024. If your block has 1024 threads and the
   SM supports 2048, you can fit 2 blocks.

The **most restrictive** resource determines occupancy.

### Practical note

Higher occupancy does not always mean better performance! If your kernel is
compute-bound (not waiting for memory), you may achieve peak performance with
50% occupancy. Occupancy is most important for memory-bound kernels.

---

## Warp Divergence

Because all 32 threads in a warp must execute the same instruction, branches
(if/else) cause problems.

### What happens with divergence

```python
# Pseudocode for a CUDA kernel
if threadIdx.x < 16:
    # Path A — first 16 threads take this
    result = expensive_computation_A()
else:
    # Path B — last 16 threads take this
    result = expensive_computation_B()
```

The warp executes BOTH paths, with threads disabled on the path they did not
take:

```
Step 1: Execute Path A
  Thread 0-15:  ACTIVE   [compute A]
  Thread 16-31: DISABLED [waiting]

Step 2: Execute Path B
  Thread 0-15:  DISABLED [waiting]
  Thread 16-31: ACTIVE   [compute B]

Total time = time(A) + time(B)   ← not max(A,B)!
```

### How to minimize divergence

1. **Make branches warp-aligned:** If all threads in a warp take the same path,
   there is no divergence.

```python
# BAD: divergence within a warp
if threadIdx.x % 2 == 0:  # Half of each warp diverges
    ...

# BETTER: divergence between warps (no divergence within)
if threadIdx.x < 128:  # First 4 warps go one way, rest go the other
    ...
```

2. **Use predication for short branches:** Modern GPUs can sometimes use
   predicated execution (both paths computed, result selected) for very short
   branches.

3. **Restructure data** to avoid divergence entirely.

---

## Putting It All Together: Matrix Multiply on GPU

Here is a conceptual tiled matrix multiply kernel, with all the key concepts:

```
Step 1: Partition C into tiles (one block computes one tile of C)

  Grid of blocks, each block computes a TILE_SIZE x TILE_SIZE portion of C.

Step 2: Each block iterates over tiles of A and B

  For each step:
  a. Load TILE_SIZE x TILE_SIZE tile of A into shared memory  (coalesced load)
  b. Load TILE_SIZE x TILE_SIZE tile of B into shared memory  (coalesced load)
  c. __syncthreads()  — wait for all loads to complete
  d. Each thread computes its portion of the tile product using shared memory
     (fast access, no bank conflicts if layout is right)
  e. __syncthreads()  — wait for all computation before loading next tile
  f. Accumulate into thread-local register

Step 3: Write final result from register to global memory (coalesced store)
```

```
Illustrated for 4x4 matrices with TILE_SIZE=2:

     A                    B                    C = A @ B
┌────┬────┐         ┌────┬────┐          ┌────┬────┐
│A00 │A01 │         │B00 │B01 │          │C00 │C01 │
├────┼────┤         ├────┼────┤          ├────┼────┤
│A10 │A11 │         │B10 │B11 │          │C10 │C11 │
└────┴────┘         └────┴────┘          └────┴────┘
  (2x2 tiles)        (2x2 tiles)         (2x2 tiles)

Block (0,0) computes C00:
  Load A00 into shared memory (2x2)
  Load B00 into shared memory (2x2)
  __syncthreads()
  C00_partial = A00 @ B00       ← using shared memory (fast!)
  __syncthreads()

  Load A01 into shared memory (2x2)
  Load B10 into shared memory (2x2)
  __syncthreads()
  C00 += A01 @ B10              ← accumulate in registers
  __syncthreads()

  Store C00 to global memory
```

### Why this is fast

- **Global memory accesses:** Only 2 * N/T tiles loaded per block (instead of
  N random accesses per element).
- **Shared memory reuse:** Each loaded tile is used T times.
- **Coalescing:** Tiles are loaded with adjacent threads reading adjacent
  addresses.
- **Bank conflicts:** Can be avoided with padding (T+1 columns in shared memory).

---

## Modern GPU Specs (A100 / H100)

These numbers come up frequently in system design discussions.

### NVIDIA A100 (2020)

| Spec | Value |
|------|-------|
| SMs | 108 |
| FP32 cores per SM | 64 |
| Total FP32 cores | 6912 |
| Tensor Cores per SM | 4 (3rd gen) |
| Total Tensor Cores | 432 |
| Peak FP32 | 19.5 TFLOPS |
| Peak FP16 Tensor | 312 TFLOPS |
| Peak INT8 Tensor | 624 TOPS |
| HBM2e | 80 GB |
| Memory Bandwidth | 2.0 TB/s |
| L2 Cache | 40 MB |
| Shared Memory per SM | up to 164 KB |
| Max threads per SM | 2048 |
| Max threads per block | 1024 |
| TDP | 400W |
| NVLink Bandwidth | 600 GB/s (bidirectional) |

### NVIDIA H100 (2022)

| Spec | Value |
|------|-------|
| SMs | 132 |
| FP32 cores per SM | 128 |
| Total FP32 cores | 16896 |
| Tensor Cores per SM | 4 (4th gen) |
| Peak FP32 | 67 TFLOPS |
| Peak FP16 Tensor | 990 TFLOPS |
| Peak FP8 Tensor | 1979 TFLOPS |
| HBM3 | 80 GB |
| Memory Bandwidth | 3.35 TB/s |
| L2 Cache | 50 MB |
| NVLink Bandwidth | 900 GB/s |

### Key ratios to know

- **Compute-to-memory ratio (A100):** 312 TFLOPS / 2 TB/s = 156 FLOPs per byte
  loaded. This means you need 156 FLOPs of compute for every byte you load
  from HBM to keep the GPU busy. Matrix multiply has high arithmetic intensity
  and can achieve this; elementwise operations cannot.

- **NVLink vs HBM bandwidth:** NVLink (600 GB/s) is ~30% of HBM bandwidth
  (2 TB/s). Communication between GPUs is expensive relative to local memory.

---

## Key Takeaways for Interviews

1. **GPUs are throughput machines.** They are not faster per thread — they win
   by running thousands of threads and hiding latency through parallelism.

2. **Think in warps (32 threads).** Memory access and branching behavior are
   determined at the warp level. Adjacent threads should access adjacent memory.

3. **Memory is the bottleneck.** Most GPU kernels are memory-bound. Reducing
   memory traffic (via tiling, fusion, quantization) is the primary optimization
   strategy.

4. **Shared memory is programmable L1.** Use it to stage data from global memory
   and reuse it within a block. Watch for bank conflicts.

5. **Occupancy helps hide latency.** More active warps give the scheduler more
   options. But higher occupancy is not always better — sometimes you trade
   occupancy for more registers or shared memory per thread.

6. **Coalescing is critical.** Uncoalesced memory access can waste 90%+ of
   memory bandwidth. Use Structure of Arrays (SoA), not Array of Structures
   (AoS).

7. **Avoid warp divergence.** Keep branches warp-aligned (all threads in a warp
   should take the same path).

8. **Know the numbers.** A100: 80 GB HBM, 2 TB/s bandwidth, 312 TFLOPS FP16.
   H100: 80 GB HBM3, 3.35 TB/s, ~1000 TFLOPS FP16. These let you estimate
   whether a kernel is compute-bound or memory-bound.

9. **Tensor Cores** are specialized hardware for matrix multiply (e.g., 16x16
   matrix tiles in FP16). They provide 10-20x higher throughput than regular
   FP32 cores for matmul. Flash Attention and all modern LLM inference rely
   on Tensor Cores.

10. **The optimization hierarchy:**
    - First: reduce memory traffic (algorithmic changes, fusion, quantization)
    - Second: improve memory access pattern (coalescing, tiling)
    - Third: increase compute utilization (occupancy, Tensor Cores)
    - Last: micro-optimize (instruction scheduling, register pressure)
