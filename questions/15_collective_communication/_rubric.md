# Rubric: Collective Communication Simulator

**Total: 100 points**

---

## 1. Correct Ring AllReduce with Proper Chunk Splitting and Accumulation (25 points)

### Full marks (25):
- Correctly divides each GPU's data into N chunks
- Handles data length not evenly divisible by N (some chunks are 1 element larger, or padding is used)
- **Phase 1 (Reduce-Scatter)**: N-1 steps
  - In step `s`, GPU `i` sends chunk `(i - s) % N` to GPU `(i + 1) % N`
  - GPU `i` receives chunk `(i - 1 - s) % N` from GPU `(i - 1) % N`
  - The received chunk is accumulated with the local chunk using the reduction op
  - After N-1 steps, GPU `i` holds the fully-reduced version of chunk `(i + 1) % N`
- **Phase 2 (AllGather)**: N-1 steps
  - GPUs pass their reduced chunks around the ring
  - Each GPU copies the received chunk into the correct position
  - After N-1 steps, all GPUs have all N reduced chunks
- All three ops supported: sum, max, avg
- For avg: reduce with sum during scatter, divide by N after
- Returns correct stats (steps = 2*(N-1))

### Partial credit:
- (20) Correct result but not actually using the ring algorithm (e.g., using naive reduce + broadcast)
- (15) Ring algorithm implemented but has bugs with chunk indexing
- (10) Phase 1 works but Phase 2 is wrong
- (5) Understands the concept but implementation is broken

### Common mistakes:
- Wrong chunk indices in the ring rotation
- Not making deep copies of data being sent (aliasing issues)
- Off-by-one in chunk boundary calculation for non-divisible lengths
- Accumulating in the wrong direction

---

## 2. Correct AllGather and Reduce-Scatter (15 points)

### Full marks (15):
- **AllGather**:
  - N-1 ring steps, each GPU forwards its most recently received chunk
  - Final result is concatenation of all chunks in GPU order
  - Handles unequal chunk sizes
  - Correct stats
- **Reduce-Scatter**:
  - N-1 ring steps (essentially Phase 1 of allreduce)
  - After completion, GPU i holds only chunk i of the reduction
  - Returns each GPU's chunk, not the full vector
  - Correct stats

### Partial credit:
- (10) One of the two is correct
- (7) Both have minor bugs
- (5) Understands concepts but implementations don't work

### Key insight:
- AllReduce = Reduce-Scatter + AllGather (in that order)
- The candidate should recognize this decomposition

---

## 3. Understanding of the 2*(N-1) Step Structure and Bandwidth Optimality (15 points)

*Assessed through implementation, stats computation, and verbal explanation.*

### Full marks (15):
- Correctly computes `steps = 2 * (N - 1)` for allreduce, `N - 1` for each sub-operation
- Correctly computes bytes transferred per GPU:
  - Each step, each GPU sends one chunk of size `data_length / N`
  - Total data sent per GPU = `2 * (N-1) * (data_length / N) * 4` bytes
  - This approaches `2 * data_length * 4` bytes as N grows (bandwidth-optimal!)
- Can explain WHY this is bandwidth-optimal:
  - Each element must be read from N-1 other GPUs -> minimum `(N-1)/N * data_size` bytes sent
  - Each element must be sent to N-1 other GPUs -> minimum `(N-1)/N * data_size` bytes received
  - Ring allreduce achieves exactly these minimums (in each phase)

### Partial credit:
- (10) Correct step count but wrong or missing bytes calculation
- (5) Knows it's 2*(N-1) but cannot explain why it's optimal
- (0) Wrong step count

### Key insight:
- The ring algorithm is bandwidth-optimal: it uses the minimum amount of data transfer to accomplish the allreduce. The latency cost is 2*(N-1) message rounds (not optimal -- tree allreduce has O(log N) latency).

---

## 4. Correct Data Parallel Simulation with Gradient Averaging (10 points)

### Full marks (10):
- Uses allreduce with op="avg" (or op="sum" followed by division by N)
- Returns identical averaged gradients on all GPUs
- Correctly passes through communication stats
- Handles the num_gpus parameter (validates against len(gradients_per_gpu))

### Partial credit:
- (7) Correct but uses sum instead of avg (or vice versa) without adjustment
- (5) Calls allreduce but doesn't properly average
- (3) Doesn't actually use the ring allreduce

---

## 5. Pipeline Parallel Simulation with Correct Bubble Ratio (20 points)

### Full marks (20):
- Correctly splits data into microbatches
- Implements pipeline scheduling:
  - At step `t`, stage `s` processes microbatch `t - s` (if `0 <= t - s < M`)
  - A stage is idle (bubble) if `t - s < 0` or `t - s >= M`
- Processes each microbatch through all stages in order
- Returns results in the correct order (matching original data order)
- Correct total_steps = S + M - 1
- Correct bubble_ratio computation:
  - Total slots = S * (S + M - 1)
  - Active slots = S * M
  - Bubble slots = S * (S + M - 1) - S * M = S * (S - 1)
  - Bubble ratio = (S - 1) / (S + M - 1) -- equivalently, S*(S-1) / (S*(S+M-1))
- Handles edge cases: single stage (0 bubbles), single microbatch

### Partial credit:
- (15) Correct output but wrong bubble ratio calculation
- (12) Pipeline scheduling works but results are in wrong order
- (8) Basic idea is right but doesn't actually simulate the pipeline timing
- (5) Just applies stages sequentially without pipelining

### Key insight:
- The candidate should understand that increasing microbatches reduces bubble ratio:
  - `bubble_ratio = (S-1) / (S + M - 1)` -> approaches 0 as M -> infinity
  - But more microbatches = smaller microbatch size = potentially lower hardware utilization per microbatch
  - This is the fundamental tradeoff in pipeline parallelism

---

## 6. Edge Cases (10 points)

### Full marks (10):
- **Single GPU**: allreduce returns data unchanged, 0 steps, 0 bytes transferred
- **Non-divisible chunk sizes**: data length not a multiple of N GPUs is handled (some chunks get +1 elements)
- **Different reduction ops**: sum, max, avg all work correctly
- **Empty or length-1 data**: handles gracefully
- **Pipeline with 1 stage**: total_steps = M, bubble_ratio = 0
- **Pipeline with 1 microbatch**: total_steps = S, bubble_ratio = (S-1)/S

### Partial credit:
- (7) Most edge cases handled
- (4) Some edge cases cause crashes or wrong results
- (2) Only happy path works

---

## 7. Code Clarity (5 points)

### Full marks (5):
- Clean, readable implementation with good variable names
- Helper functions are well-used (e.g., elementwise_op)
- Deep copies are used where needed (no aliasing bugs)
- Comments explain the ring algorithm steps
- Consistent style

### Partial credit:
- (3) Functional but messy
- (1) Works but hard to follow

---

## Bonus Observations (not scored, but positive signals):

- Mentions latency vs bandwidth trade-off: ring is bandwidth-optimal but latency-suboptimal (2*(N-1) rounds vs O(log N) for tree)
- Discusses how NCCL uses different algorithms for different message sizes (ring for large, tree for small)
- Notes that in practice, allreduce overlaps with backward computation
- Mentions gradient compression or quantized gradients for communication reduction
- Discusses the relationship between pipeline parallelism and gradient accumulation
- Notes that real pipeline parallelism also needs a backward pass (1F1B schedule)
- Mentions interleaved pipeline scheduling (Megatron-LM)
- Discusses how NVLink and NVSwitch enable non-ring topologies

---

## Red Flags:

- Cannot explain what allreduce does conceptually
- Does not understand the ring topology constraint (uses all-to-all communication)
- Cannot explain why ring allreduce takes 2*(N-1) steps
- Pipeline implementation doesn't actually pipeline (processes microbatches fully sequentially)
- Bubble ratio calculation is fundamentally wrong
- Confused about the difference between allreduce, allgather, and reduce-scatter
- Aliasing bugs (mutating data that's being "sent" to another GPU in the same step)
- Does not make deep copies of gpu_data at the start (modifies input)
