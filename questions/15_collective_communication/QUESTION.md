# Question 15: Collective Communication Simulator

## Overview

In distributed GPU training and inference, GPUs must communicate to synchronize gradients, shard activations, and combine results. You are implementing a simulator for collective communication operations across a ring of **N GPUs**.

Understanding these primitives is essential for distributed ML systems: every gradient synchronization in data parallelism, every tensor split in model parallelism, and every pipeline stage handoff relies on these operations.

---

## Background

### Ring Topology

Each GPU can communicate only with its immediate neighbors in a ring:
- GPU `i` can send to GPU `(i + 1) % N` and receive from GPU `(i - 1) % N`.

This is a simplified model of how NCCL (NVIDIA Collective Communications Library) implements its ring algorithm.

### Collective Operations

- **AllReduce**: Every GPU starts with local data. After the operation, every GPU has the element-wise reduction (sum, max, or average) of all GPUs' data.
- **AllGather**: Every GPU starts with a chunk. After the operation, every GPU has the concatenation of all chunks.
- **ReduceScatter**: Every GPU starts with full data. After the operation, GPU `i` has the `i`-th chunk of the element-wise reduction.

### Ring AllReduce Algorithm

The ring allreduce is bandwidth-optimal and consists of two phases:

**Phase 1 -- Reduce-Scatter** (N-1 steps):
- Each GPU's data is logically divided into N equal-sized chunks.
- In each step, every GPU sends one chunk to its right neighbor and receives one chunk from its left neighbor, accumulating (reducing) the received chunk with its own.
- After N-1 steps, GPU `i` holds the fully-reduced version of chunk `i`.

**Phase 2 -- AllGather** (N-1 steps):
- Each GPU sends its fully-reduced chunk to its right neighbor and receives the next GPU's reduced chunk from the left.
- After N-1 steps, every GPU has all N fully-reduced chunks.

Total: 2 * (N-1) steps, each transferring `data_size / N` bytes per GPU.

### Pipeline Parallelism

In pipeline parallelism, the model is split across GPUs by layers (stages). Micro-batching is used to fill the pipeline and reduce idle time ("bubbles").

With `S` stages and `M` micro-batches:
- Total time steps = `S + M - 1` (for the forward pass)
- Pipeline bubble = `S - 1` idle time slots at the start/end
- Bubble ratio = `(S - 1) / (S + M - 1)`

---

## Task

Implement the following functions. Use only the Python standard library.

### 1. Ring AllReduce

```python
def ring_allreduce(
    gpu_data: list[list[float]], op: str = "sum"
) -> tuple[list[list[float]], dict]:
```

- `gpu_data[i]` is the data vector on GPU `i`. All vectors have the same length.
- `op` is one of `"sum"`, `"max"`, or `"avg"`.
- Implement the ring algorithm:
  - Divide each GPU's data into `N` chunks (where `N = len(gpu_data)`).
  - Phase 1 (Reduce-Scatter): N-1 steps of send-right, receive-left, accumulate.
  - Phase 2 (AllGather): N-1 steps of send-right, receive-left, overwrite.
  - For `"avg"`: reduce with sum, then divide by N at the end.
- Return:
  - The final state of all GPUs (all should be identical, containing the reduced data).
  - A stats dict: `{"steps": int, "bytes_transferred_per_gpu": int}` where bytes_transferred_per_gpu counts total floats sent by a single GPU * 4 bytes per float.

### 2. Ring AllGather

```python
def ring_allgather(
    gpu_data: list[list[float]]
) -> tuple[list[list[float]], dict]:
```

- `gpu_data[i]` is a chunk on GPU `i`. Chunks may have different lengths.
- After the operation, every GPU has the concatenation of all chunks in order: `[chunk_0, chunk_1, ..., chunk_{N-1}]`.
- Uses N-1 ring steps: in each step, every GPU sends its most recently received chunk to the right.
- Return the final state and stats.

### 3. Ring Reduce-Scatter

```python
def ring_reduce_scatter(
    gpu_data: list[list[float]], op: str = "sum"
) -> tuple[list[list[float]], dict]:
```

- `gpu_data[i]` is a full vector on GPU `i`. All vectors have the same length.
- After the operation, GPU `i` holds only the `i`-th chunk of the element-wise reduction.
- Uses N-1 ring steps (Phase 1 of allreduce).
- Return the final state (each GPU holds only its chunk) and stats.

### 4. Data Parallel Training Step

```python
def simulate_data_parallel_step(
    gradients_per_gpu: list[list[float]], num_gpus: int
) -> tuple[list[list[float]], dict]:
```

- Simulate one step of data-parallel gradient synchronization:
  - Each GPU has independently computed gradients.
  - Use allreduce to **average** gradients across all GPUs.
  - Return the averaged gradients (identical on all GPUs) and communication stats.

### 5. Pipeline Parallel Simulation

```python
def simulate_pipeline_parallel(
    stages: list[Callable],
    data: list[list[float]],
    num_microbatches: int,
) -> tuple[list[list[float]], dict]:
```

- `stages` is a list of `S` functions, one per pipeline stage/GPU. Each function takes a list of floats and returns a list of floats.
- `data` is a list of input samples to be processed.
- Split `data` into `num_microbatches` equal-sized groups (micro-batches).
- Execute with pipeline scheduling:
  - Time proceeds in discrete steps.
  - At each step, each stage processes the next available micro-batch (if the previous stage has finished it).
  - A stage is idle if it has no micro-batch to process (this is a "bubble").
- Return:
  - The output for each input sample (in original order).
  - Stats dict: `{"total_steps": int, "bubble_ratio": float, "num_stages": int, "num_microbatches": int}`
  - `total_steps` = total discrete time steps for all microbatches to pass through all stages.
  - `bubble_ratio` = fraction of total (stages * total_steps) GPU-time slots that are idle.

---

## Examples

### AllReduce

```python
# 4 GPUs, each with a vector of length 8
gpu_data = [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
]

result, stats = ring_allreduce(gpu_data, op="sum")

# Every GPU now has [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
print(result[0])  # [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
print(result[1])  # [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]  (same)

print(stats["steps"])  # 6 (2 * (4-1) = 6)
```

### AllGather

```python
gpu_data = [
    [1.0, 2.0],   # GPU 0's chunk
    [3.0, 4.0],   # GPU 1's chunk
    [5.0, 6.0],   # GPU 2's chunk
]

result, stats = ring_allgather(gpu_data)

# Every GPU now has [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
print(result[0])  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

### Pipeline Parallelism

```python
# 3 stages, each doubles its input
stages = [
    lambda x: [v * 2 for v in x],
    lambda x: [v + 1 for v in x],
    lambda x: [v * 3 for v in x],
]

data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
result, stats = simulate_pipeline_parallel(stages, data, num_microbatches=3)

# Micro-batch 0: [1.0, 2.0] -> *2 -> +1 -> *3 = [9.0, 15.0]
# total_steps = S + M - 1 = 3 + 3 - 1 = 5
# bubble_ratio = (3 - 1) / (3 * 5) = 2/15 ≈ 0.1333
print(stats["total_steps"])   # 5
print(stats["bubble_ratio"])  # ~0.1333
```

---

## Constraints

- All GPU data vectors for allreduce and reduce-scatter have the same length.
- `op` is one of `"sum"`, `"max"`, or `"avg"`.
- `num_microbatches` evenly divides `len(data)`.
- Stage functions are pure (no side effects) and deterministic.
- Use only the Python standard library.
- Handle edge cases: single GPU (no communication needed), data length not divisible by number of GPUs (pad or handle remainders).
