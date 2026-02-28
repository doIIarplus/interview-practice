# Follow-Up Questions: Collective Communication Simulator

---

## 1. Why is ring allreduce bandwidth-optimal? What is its latency vs bandwidth cost?

**Expected answer:**

**Bandwidth optimality:**

Consider allreduce on N GPUs, each with D bytes. Every element must incorporate contributions from all N GPUs. The minimum data that must leave any single GPU is:
- Phase 1 (Reduce-Scatter): Each GPU must send (N-1)/N * D bytes (all data except its own final chunk)
- Phase 2 (AllGather): Each GPU must send (N-1)/N * D bytes (the reduced chunks to everyone else)
- Total: 2 * (N-1)/N * D bytes per GPU

The ring algorithm achieves exactly this: in each of the 2*(N-1) steps, each GPU sends D/N bytes. Total = 2*(N-1) * D/N = 2*(N-1)/N * D. This matches the lower bound, so it is **bandwidth-optimal**.

**Cost breakdown:**
- **Bandwidth cost**: 2 * (N-1)/N * D -- approaches 2D as N grows, independent of N. This means adding more GPUs does not increase per-GPU bandwidth requirement.
- **Latency cost**: 2 * (N-1) * alpha, where alpha is the per-message latency. This grows linearly with N, which is **not optimal**. A tree allreduce has O(log N) latency.

**When to use which:**
- **Ring**: Best for large messages (bandwidth-dominated). This is the common case for gradient allreduce in training.
- **Tree**: Best for small messages (latency-dominated). Better for short control messages or small tensors.
- **NCCL in practice**: Uses a hybrid -- ring for large messages, tree for small, and can use multi-ring across NVLink domains.

---

## 2. How does NCCL implement allreduce differently from a pure ring? (tree, hierarchical)

**Expected answer:**

NCCL (NVIDIA Collective Communications Library) uses several strategies beyond a single ring:

**Multi-ring:**
- Uses multiple independent rings simultaneously across different NVLink/PCIe paths to utilize all available bandwidth.
- E.g., on a DGX A100 with NVSwitch, data is split across multiple rings that use different NVLink connections.

**Tree allreduce:**
- For small messages, uses a binary tree reduction + broadcast.
- Latency: O(log N) rounds instead of O(N).
- Bandwidth: 2 * D bytes total (not per-GPU optimal), but for small D the latency savings dominate.

**Hierarchical / multi-level:**
- **Intra-node**: Use NVLink/NVSwitch (fast, ~600 GB/s per GPU on DGX H100).
- **Inter-node**: Use InfiniBand/RoCE (slower, ~400 Gb/s per NIC).
- First reduce within a node (fast NVLink ring or all-to-all via NVSwitch), then reduce across nodes (slower network), then broadcast back.
- This minimizes expensive inter-node communication.

**Double binary tree:**
- NCCL uses two binary trees (a "tree" and its complement) to balance load.
- Each tree reduces half the data. Together they achieve bandwidth optimality.

**Direct send/receive (NCCL 2.12+):**
- For NVSwitch-connected GPUs, can use all-to-all communication pattern.
- Every GPU sends 1/N of its data directly to every other GPU.
- Single step, but requires full bisection bandwidth (NVSwitch provides this).

---

## 3. What is NVLink and NVSwitch? How do they change the communication topology?

**Expected answer:**

**NVLink:**
- High-bandwidth GPU-to-GPU interconnect (proprietary to NVIDIA).
- NVLink 4.0 (Hopper): 900 GB/s bidirectional per GPU (18 links x 25 GB/s each).
- Much faster than PCIe (PCIe 5.0 x16 ≈ 64 GB/s).
- Each GPU has a limited number of NVLink connections (12-18 links), so not every GPU can be directly connected to every other GPU in large systems.

**NVSwitch:**
- A crossbar switch that connects all GPUs with NVLink.
- DGX H100 has NVSwitch connecting 8 GPUs, providing full bisection bandwidth.
- Every GPU can send to every other GPU simultaneously at full NVLink speed.
- NVLink Network (NVLink Switch systems): extends NVSwitch across nodes, potentially connecting 256+ GPUs in a single NVLink domain.

**Impact on topology:**
- **Without NVSwitch**: GPUs form a limited graph (e.g., 4-way ring, mesh). Communication algorithms must respect the topology. Ring allreduce is the natural fit.
- **With NVSwitch (intra-node)**: Full all-to-all connectivity. Can use direct all-to-all patterns instead of ring. Allreduce can be done in a single step (reduce-scatter via direct sends).
- **Multi-node**: Hierarchical topology -- NVSwitch within node, InfiniBand/Ethernet between nodes. Algorithms must be topology-aware: fast intra-node, minimize inter-node traffic.

**NCCL topology detection:**
- NCCL auto-detects the topology (NVLink connections, PCIe switches, network NICs) and selects the optimal algorithm and channel configuration.

---

## 4. Explain tensor parallelism vs pipeline parallelism vs data parallelism. When would you use each?

**Expected answer:**

**Data Parallelism:**
- Each GPU has a complete copy of the model.
- Training data is split across GPUs.
- Forward/backward pass on local data, then allreduce to average gradients.
- Scales well for small-to-medium models. Bottleneck: gradient communication.
- Use when: model fits on one GPU, want to increase throughput.

**Tensor Parallelism (Intra-layer):**
- Individual layers are split across GPUs. E.g., a large linear layer W is split column-wise across 4 GPUs, each computing part of the output.
- Requires all-reduce or all-gather at specific points within each layer (e.g., after the column-parallel linear, before the row-parallel linear).
- Communication is frequent (every layer) but small (activation-sized).
- Use when: individual layers are too large for one GPU. Requires fast interconnect (NVLink). Typically used for up to 8 GPUs within a node.

**Pipeline Parallelism (Inter-layer):**
- Model layers are partitioned into stages, each on a different GPU.
- Micro-batching fills the pipeline to reduce bubbles.
- Communication is infrequent (only at stage boundaries) and involves only activations.
- Use when: model is very large, you want to spread across multiple nodes where inter-node bandwidth is limited. The activation-only communication minimizes network traffic.

**Typical large-scale training (e.g., Megatron-LM):**
- **3D parallelism**: Combine all three.
  - Tensor parallel: 8 GPUs within a node (NVLink)
  - Pipeline parallel: across nodes in a pipeline (InfiniBand)
  - Data parallel: replicate pipeline groups for throughput (allreduce over InfiniBand)
- Example: 512 GPUs = 8 TP x 8 PP x 8 DP

---

## 5. What is ZeRO (Zero Redundancy Optimizer) and how does it use communication differently?

**Expected answer:**

ZeRO (by DeepSpeed/Microsoft) is a memory optimization that eliminates redundancy in data parallelism. Standard data parallelism replicates the entire model state (parameters, gradients, optimizer states) on every GPU -- a huge memory waste.

**ZeRO stages:**

- **ZeRO Stage 1 (Optimizer State Partitioning):**
  - Optimizer states (e.g., Adam's momentum and variance, 2x the parameter size) are partitioned across GPUs.
  - Each GPU only updates 1/N of the parameters.
  - Communication: allgather of updated parameters after optimizer step.
  - Memory savings: ~4x for Adam (optimizer states dominate).

- **ZeRO Stage 2 (+ Gradient Partitioning):**
  - Gradients are also partitioned. Each GPU only keeps gradients for its 1/N of parameters.
  - Communication: reduce-scatter during backward (instead of allreduce). Each GPU gets only the gradient chunk it needs.
  - Memory savings: eliminates gradient redundancy.

- **ZeRO Stage 3 (+ Parameter Partitioning):**
  - Parameters themselves are partitioned! Each GPU stores only 1/N of the parameters.
  - Communication: allgather of parameters before each forward/backward layer (on-demand).
  - Memory savings: model parameters, gradients, and optimizer states are all partitioned -> almost N-fold memory reduction.
  - Cost: more communication (allgather per layer in forward AND backward).

**Key communication differences from standard DP:**
- Standard DP: allreduce of all gradients once per step.
- ZeRO-1/2: reduce-scatter instead of allreduce (half the communication for gradients).
- ZeRO-3: reduce-scatter for gradients + allgather for parameters (more total communication but spread over the forward/backward pass, enabling overlap with compute).

**ZeRO-Offload / ZeRO-Infinity:**
- Offload partitioned optimizer states and parameters to CPU memory or NVMe, further expanding the effective memory.

---

## 6. How does sequence parallelism work for long-context inference?

**Expected answer:**

Sequence parallelism distributes the sequence dimension across GPUs, which is critical for long-context scenarios where the KV-cache and attention computation grow quadratically with sequence length.

**Ring Attention (Liu et al., 2023):**
- The KV-cache is split across GPUs along the sequence dimension.
- Each GPU holds a contiguous chunk of the sequence.
- Attention is computed by passing K/V blocks around a ring:
  - Each GPU computes partial attention with its local Q against the local K/V chunk.
  - K/V chunks are rotated around the ring, and each GPU accumulates attention contributions from all K/V chunks.
  - Uses online softmax (similar to Flash Attention) to accumulate without materializing the full attention matrix.
- Communication: K/V blocks are passed ring-style, overlapping with attention computation.

**Ulysses/DeepSpeed Sequence Parallelism:**
- Splits the sequence across GPUs after the attention QKV projection.
- Uses all-to-all communication to redistribute data from sequence-parallel to head-parallel layout.
- Each GPU computes attention for a subset of heads on the full sequence.
- Another all-to-all redistributes back to sequence-parallel layout.

**Megatron Sequence Parallelism:**
- In regions of the transformer that are embarrassingly parallel along the sequence dimension (LayerNorm, Dropout, activation functions), splits the sequence.
- At attention and feedforward layers, switches to tensor parallelism.
- Reduces activation memory by 1/TP_degree for sequence-parallel regions.

**Key benefit:** enables context lengths of 100K+ tokens that would not fit on a single GPU's memory.

---

## 7. What is the difference between synchronous and asynchronous allreduce? When would you overlap communication with computation?

**Expected answer:**

**Synchronous allreduce:**
- All GPUs participate in the allreduce simultaneously.
- The allreduce blocks until complete -- no computation during communication.
- Simplest to reason about, guarantees all GPUs have consistent gradients before the optimizer step.

**Asynchronous allreduce:**
- The allreduce is initiated and computation continues while communication happens in the background.
- The result is retrieved later when needed.

**Communication-computation overlap in practice:**
- **During backward pass**: Gradients for earlier layers are computed while later layers are still in backward. As soon as a layer's gradient is computed, its allreduce can start while the backward pass continues.
- **Bucket-based**: PyTorch DDP groups parameters into buckets (e.g., 25 MB). When a bucket of gradients is ready, allreduce starts for that bucket while other buckets are still computing.
- **Scheduling**: The computation graph is analyzed to maximize overlap. NCCL streams run concurrently with CUDA compute streams.

**When to overlap:**
- Almost always in training -- it is standard practice.
- The backward pass is naturally suited because layer gradients are produced sequentially.
- For very small models, the overlap overhead (kernel launch, stream synchronization) may not be worth it.
- For inference (pipeline parallelism), overlap activation transfer with compute on other stages.

**Gradient staleness concern:**
- Fully asynchronous SGD (using stale gradients) can harm convergence.
- The standard practice is to overlap within a single step (not use stale gradients). Each step still completes its allreduce before the optimizer step.

---

## 8. How would you handle a straggler GPU in distributed training?

**Expected answer:**

A straggler is a GPU that runs slower than others, causing all other GPUs to wait (since collective operations are synchronous).

**Detection:**
- Monitor per-GPU iteration times.
- Track time spent in collective operations (NCCL provides timing).
- Detect GPUs that consistently take longer (e.g., due to thermal throttling, ECC errors, or bad memory).

**Mitigation strategies:**

1. **Redundant computation:**
   - Run training on N+K GPUs, where K are backups.
   - If a GPU is slow, use the backup's gradient instead.
   - Wastes compute but maintains throughput.

2. **Dynamic load balancing:**
   - Adjust batch sizes per GPU -- give the straggler a smaller micro-batch.
   - Requires knowing which GPU is slow ahead of time.

3. **Asynchronous/bounded-stale SGD:**
   - Allow fast GPUs to proceed without waiting for the straggler.
   - Use gradients that may be 1-2 steps stale.
   - Requires careful convergence analysis.

4. **Timeout and skip:**
   - Set a timeout on collective operations.
   - If a GPU doesn't respond, exclude it from the reduction (reduce with N-1 GPUs).
   - Requires fault-tolerant allreduce implementations.

5. **Checkpointing and restart:**
   - If a GPU is consistently slow (hardware issue), checkpoint and restart with N-1 GPUs or on a healthy node.
   - Elastic training frameworks (PyTorch Elastic, Horovod Elastic) support dynamic scaling.

6. **Gradient compression:**
   - Compress gradients (top-k sparsification, quantization) to reduce communication time.
   - Particularly helps when the straggler's bottleneck is network bandwidth.

**Systemic prevention:**
- GPU health monitoring (nvidia-smi, DCGM)
- Pre-training hardware validation
- Homogeneous hardware clusters
- Network topology-aware job placement (avoid stragglers due to network congestion)
