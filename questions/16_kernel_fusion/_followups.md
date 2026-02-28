# Follow-Up Questions: Kernel Fusion Simulator

## 1. How does torch.compile / Triton handle kernel fusion automatically?

**What to look for:**
- `torch.compile` traces the computation graph via TorchDynamo (Python bytecode analysis) and hands it to a backend (TorchInductor by default).
- TorchInductor performs fusion analysis similar to what we implemented: it groups elementwise ops, fuses matmul epilogues, etc.
- It then generates Triton kernels for the fused groups. Triton is a DSL that compiles to PTX (NVIDIA GPU assembly).
- Key insight: the fusion decisions are made at the graph level, not the kernel level. The graph is partitioned into "fusion groups" and each group becomes one Triton kernel.
- Strong candidates mention that `torch.compile` can also do "horizontal fusion" (fusing independent ops that can share a grid launch) and "vertical fusion" (what we implemented).

## 2. What is Flash Attention and why is it essentially a hand-fused kernel for attention?

**What to look for:**
- Flash Attention (Dao et al.) fuses the entire Q*K^T -> softmax -> score*V computation into one kernel.
- The key insight is **tiling**: instead of materializing the full (seq_len x seq_len) attention matrix in HBM, it processes tiles in SRAM.
- It uses the **online softmax** trick to compute softmax without materializing the full row.
- Memory complexity goes from O(N^2) to O(N) in sequence length.
- This is "hand-fused" because automatic compilers generally cannot discover the online softmax trick -- it requires algorithmic restructuring, not just operator merging.
- Strong candidates note that Flash Attention 2 and 3 further optimize for GPU occupancy and use asynchronous memory copies.

## 3. What are the limits of fusion? When does fusion hurt performance?

**What to look for:**
- **Register pressure**: fusing too many ops can require too many registers per thread, reducing occupancy (fewer warps can run concurrently).
- **Shared memory limits**: fused kernels may need more shared memory than available, forcing spills to global memory.
- **Divergent memory access patterns**: fusing an elementwise op with a reduction changes the parallelism strategy. The elementwise op wants one thread per element; the reduction needs cooperative threads.
- **Kernel launch overhead**: for very large ops, launch overhead is negligible, so fusion provides diminishing returns.
- **Compilation time**: more complex fused kernels take longer to compile (JIT compilation in Triton/CUDA).
- **Debugging difficulty**: fused kernels are harder to profile and debug.

## 4. How do memory-bound vs. compute-bound kernels affect fusion decisions?

**What to look for:**
- **Memory-bound kernels** (elementwise ops, reductions): bottlenecked by memory bandwidth. Fusion helps the most here because it eliminates memory round-trips.
- **Compute-bound kernels** (matmul): bottlenecked by arithmetic throughput. Fusion of the matmul itself doesn't help, but fusing the epilogue (bias + activation) is beneficial because the epilogue is essentially "free" -- the data is already in registers after the matmul.
- The **arithmetic intensity** (FLOPs per byte transferred) determines which category a kernel falls into. Low AI = memory-bound, high AI = compute-bound.
- Fusion increases arithmetic intensity by reducing bytes transferred while keeping FLOPs the same.
- Strong candidates can compute arithmetic intensity for specific ops (e.g., matmul of M x K x N has 2*M*K*N FLOPs and 2*(M*K + K*N + M*N)*bytes_per_element memory traffic).

## 5. What is the "roofline model" and how do you use it to guide fusion?

**What to look for:**
- The roofline model plots achievable performance (FLOP/s) as a function of arithmetic intensity (FLOP/byte).
- There's a "roof" at the peak compute throughput and a sloped "ramp" at peak memory bandwidth.
- A kernel's position on this plot tells you whether it's memory-bound (on the ramp) or compute-bound (on the roof).
- **Fusion moves kernels right on the roofline plot** (higher arithmetic intensity) by reducing memory traffic.
- The goal is to get all kernels to the "ridge point" where the ramp meets the roof.
- For example, on an A100: peak compute ~312 TFLOP/s (FP16 Tensor Core), peak bandwidth ~2 TB/s. Ridge point = 312/2 = 156 FLOP/byte.
- Strong candidates can sketch a roofline and place specific operations on it.

## 6. How does mixed-precision (FP16 compute, FP32 accumulation) interact with fusion?

**What to look for:**
- In mixed precision, matmul is done in FP16 (or BF16) with FP32 accumulators. The accumulator must be cast back to FP16 for storage.
- If we fuse the matmul epilogue, the FP32->FP16 cast can happen AFTER the epilogue ops (bias add, activation), keeping intermediate values in FP32 registers. This improves numerical accuracy.
- Without fusion, the matmul writes FP16 to memory, losing precision before the epilogue.
- Loss scaling (for FP16 training) can also be fused into the kernel.
- Some ops (like layer norm) require FP32 computation for numerical stability even if inputs/outputs are FP16. A fused kernel can handle the upcast/downcast internally.

## 7. What is operator tiling and how does it relate to fusion in practice?

**What to look for:**
- **Tiling** partitions a large operation into smaller tiles that fit in SRAM/registers.
- A matmul of (M, K) x (K, N) is tiled into blocks of size (BM, BK) x (BK, BN) that are computed by individual thread blocks.
- Fusion interacts with tiling because the fused epilogue must operate on the SAME tile of data that just computed. If the tiling doesn't match (e.g., the elementwise op needs a different data layout), fusion becomes impossible or requires data reshuffling.
- In Triton, the programmer specifies tile sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) and the fused epilogue operates on the same tile.
- Auto-tuning tile sizes is important: too small = poor compute utilization, too large = register pressure.

## 8. How would you profile real kernel fusion opportunities using Nsight Compute?

**What to look for:**
- **Nsight Compute** (ncu) profiles individual CUDA kernels and reports metrics like memory throughput, compute throughput, occupancy, and achieved bandwidth.
- To find fusion opportunities: look for sequences of memory-bound kernels with short durations -- these are "skinny" kernels where launch overhead and memory round-trips dominate.
- Key metrics: `dram_read_throughput`, `dram_write_throughput`, `sm_efficiency`, `achieved_occupancy`.
- If two consecutive kernels both show low arithmetic intensity and high memory traffic, they are candidates for fusion.
- The **Nsight Systems** (nsys) timeline view shows kernel launches over time and can reveal "kernel launch gaps" where the GPU is idle between kernels.
- Strong candidates mention using `torch.profiler` with `with_stack=True` to correlate Python-level ops with CUDA kernels, then drilling into Nsight Compute for specific kernels.
- Mention of the `ncu --set full` command for comprehensive kernel analysis.
