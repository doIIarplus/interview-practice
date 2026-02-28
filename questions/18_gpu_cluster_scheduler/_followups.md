# Follow-Up Questions: GPU Cluster Scheduler

## 1. How does Megatron-LM handle 3D parallelism (TP + PP + DP) placement?

**What to look for:**
- Megatron-LM uses a **hierarchical placement** strategy identical to what we implemented: TP innermost (within node), PP in the middle, DP outermost.
- The key insight is **ordering by communication intensity**: TP has the highest communication volume (allreduce after every layer), so it gets the fastest interconnect (NVLink). DP has the lowest per-step volume (gradients once per step), so it can use the slower inter-node links.
- Megatron-LM assigns "ranks" to GPUs in a specific order: GPUs within a TP group have consecutive ranks, then PP stages, then DP replicas. This mapping is called the "process group" structure.
- Strong candidates mention that Megatron-LM also supports **interleaved pipeline scheduling** (1F1B with interleaving) which reduces pipeline bubble by assigning non-contiguous layers to each stage.
- The virtual pipeline parallelism in Megatron-LM further subdivides each PP stage into multiple "virtual stages" to reduce bubble time.

## 2. What is expert parallelism in Mixture-of-Experts models? How does it change placement?

**What to look for:**
- In MoE models (e.g., Mixtral, Switch Transformer), different tokens are routed to different "expert" FFN layers. Expert Parallelism (EP) distributes experts across GPUs.
- EP adds a **fourth dimension** to the parallelism: TP x PP x DP x EP.
- EP requires an **all-to-all** communication pattern (not allreduce): each GPU sends tokens to the GPU hosting the relevant expert and receives results back.
- All-to-all is harder to overlap with computation and is more sensitive to network topology than allreduce.
- Placement strategy: EP should typically be within a node (high bandwidth for all-to-all), but this competes with TP for intra-node slots. Trade-offs depend on the model architecture.
- Strong candidates mention that MoE models have unique load-balancing challenges: if all tokens route to the same expert, that GPU becomes a bottleneck.

## 3. How would you handle elastic training where the number of GPUs can change?

**What to look for:**
- **Elastic training** allows the training job to scale up (add GPUs) or scale down (lose GPUs) without restarting from scratch.
- Scaling the DP dimension is easiest: add/remove DP replicas. Each replica is independent except for gradient allreduce.
- Scaling TP or PP is much harder: it requires resharding model weights and repartitioning layers.
- Key challenges: (a) maintaining training correctness (learning rate schedules, batch size adjustments), (b) efficient checkpoint/restore when the parallelism configuration changes, (c) handling stragglers.
- PyTorch Elastic (torchelastic / torchrun) provides primitives for fault tolerance and elasticity.
- Strong candidates mention that elastic training is critical for **spot instances** (preemptible GPUs) which are much cheaper but can be reclaimed at any time.

## 4. What is the difference between NVLink, NVSwitch, and NVLink Switch systems (DGX, HGX)?

**What to look for:**
- **NVLink**: a high-bandwidth point-to-point interconnect between GPUs. Each link provides ~25 GB/s (NVLink 3.0) or ~50 GB/s (NVLink 4.0). GPUs have multiple NVLink connections.
- **NVSwitch**: a switch chip that connects all GPUs within a node with full bisection bandwidth. With NVSwitch, any GPU can communicate with any other GPU at full NVLink bandwidth simultaneously.
- Without NVSwitch: GPUs are connected in a mesh/ring topology. Communication between non-adjacent GPUs requires multiple hops, reducing effective bandwidth.
- **DGX A100**: 8 A100 GPUs + 6 NVSwitch chips. Full bisection bandwidth of 600 GB/s per GPU.
- **DGX H100**: 8 H100 GPUs + 4 NVSwitch chips. NVLink 4.0, 900 GB/s per GPU.
- **NVLink Switch** (GB200 NVL72): extends NVLink across nodes. 72 GPUs connected via NVLink at 1.8 TB/s, eliminating the NVLink-to-InfiniBand bandwidth cliff.
- Strong candidates note that our scheduler assumes uniform intra-node bandwidth, which is only true with NVSwitch. Without NVSwitch, the topology within a node also matters.

## 5. How does InfiniBand RDMA work and why is it important for distributed training?

**What to look for:**
- **RDMA (Remote Direct Memory Access)** allows one machine to read/write memory on another machine without involving the CPU or OS kernel on either side.
- **InfiniBand** is the network technology that provides RDMA. It uses a **lossless** network fabric with hardware-level flow control.
- Why RDMA matters: traditional TCP/IP requires data to be copied to kernel buffers, processed by the CPU, and sent through the network stack. RDMA bypasses all of this -- data goes directly from GPU memory to the network adapter to the remote GPU memory.
- **GPUDirect RDMA**: extends RDMA to GPU memory. Data can be transferred between GPUs on different nodes without staging through CPU memory at all.
- This reduces latency from milliseconds (TCP) to microseconds (RDMA) and increases throughput.
- NCCL (NVIDIA Collective Communication Library) uses RDMA under the hood for inter-node communication.
- Strong candidates mention that InfiniBand uses credit-based flow control to prevent packet drops, which is crucial for collective operations that require all participants to complete.

## 6. What checkpoint strategies exist for large model training? (Full, sharded, async)

**What to look for:**
- **Full checkpoint**: every GPU saves the complete model state. Simple but wastes storage (N copies of the same model).
- **Sharded checkpoint**: each GPU saves only its shard (TP shard, PP stage, DP replica). Total storage = 1x model size. Loading requires the same parallelism configuration. Used by FSDP, DeepSpeed ZeRO.
- **Consolidated checkpoint**: sharded during save, consolidated into a single file after. Allows loading with different parallelism configurations.
- **Asynchronous checkpoint**: overlap checkpoint writing with training. While the GPU trains on the next step, the previous step's state is being written to storage in the background. Requires pinned CPU memory for staging.
- **Incremental checkpoint**: only save the diff from the previous checkpoint. Reduces I/O but complicates recovery.
- Strong candidates mention that checkpoint I/O can take minutes for large models (hundreds of GB) and discuss strategies like NVMe SSDs, parallel file systems (Lustre, GPFS), or cloud object storage.
- The optimal checkpoint frequency is a trade-off: too often wastes time on I/O, too rare risks losing hours of training on failure.

## 7. How does Kubernetes/Slurm schedule GPU jobs in practice? What are the limitations?

**What to look for:**
- **Slurm**: the dominant scheduler for HPC/GPU clusters. Jobs request resources (GPUs, memory, nodes) and are queued. Slurm allocates nodes but doesn't understand GPU topology within nodes.
- **Kubernetes**: container orchestration. The `nvidia-device-plugin` exposes GPUs as resources. Pods request GPUs, and the scheduler assigns them to nodes.
- **Limitations**: neither scheduler understands the 3D parallelism constraints natively. They treat GPUs as fungible resources, not as having a topology.
- **Topology awareness**: Slurm has `--gres=gpu:4` (request 4 GPUs) but doesn't guarantee they're NVLink-connected. Kubernetes has topology-aware scheduling plugins but they're not standard.
- In practice: training frameworks (Megatron, DeepSpeed) handle the parallelism mapping AFTER the scheduler assigns nodes. The scheduler just provides nodes; the framework decides which GPU gets which rank.
- Strong candidates mention that GPU scheduling can lead to **fragmentation**: if a node has 3/8 GPUs free, it can't serve a TP=4 job, even though the cluster has enough total GPUs.
- Gang scheduling: all GPUs for a job must be available simultaneously. Slurm supports this natively; Kubernetes requires the `volcano` or `kueue` scheduler.

## 8. How would you implement preemption for high-priority jobs?

**What to look for:**
- **Preemption**: a high-priority job evicts a lower-priority job to claim its GPUs.
- Steps: (1) identify the lowest-priority job(s) using the needed GPUs, (2) checkpoint their state, (3) kill them and free their GPUs, (4) assign GPUs to the high-priority job, (5) queue the preempted jobs for resumption.
- **Graceful preemption**: give the preempted job time to checkpoint (e.g., finish the current training step and save state). This requires coordination between the scheduler and the training framework.
- **Priority inversion**: if a high-priority job is waiting for resources held by a low-priority job, which is waiting for resources held by a medium-priority job. Solution: priority inheritance.
- Strong candidates discuss the cost of preemption: checkpoint time + restart time + wasted work since last checkpoint. For large models, this can be 5-10 minutes.
- In cloud settings, preemption is related to **spot instances**: the cloud provider can reclaim instances at any time, effectively preempting the job.
- The scheduler should track checkpoints and restart preempted jobs automatically when resources become available.
