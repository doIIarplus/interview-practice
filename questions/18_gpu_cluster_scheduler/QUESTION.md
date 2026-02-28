# Question 18: GPU Cluster Scheduler

## Background

Training and serving large language models requires distributing work across many GPUs. A modern GPU cluster has a **hierarchical topology** with vastly different bandwidths at each level:

```
Cluster
├── Node 0: 8 GPUs connected via NVLink (600 GB/s between any two GPUs)
│   └── GPUs 0-7
├── Node 1: 8 GPUs connected via NVLink
│   └── GPUs 8-15
├── ...
└── Node N-1: 8 GPUs connected via NVLink
    └── GPUs (N-1)*8 to N*8-1

Inter-node communication: InfiniBand (400 Gb/s = 50 GB/s)
```

The key constraint: **NVLink bandwidth (600 GB/s) is 12x faster than InfiniBand (50 GB/s)**. Communication-heavy operations must be placed on GPUs within the same node.

Model parallelism has three dimensions, each with different communication requirements:

| Parallelism | Communication Pattern | Volume | Placement Preference |
|---|---|---|---|
| **Tensor Parallel (TP)** | Allreduce after every layer | Very high | MUST be within same node (NVLink) |
| **Pipeline Parallel (PP)** | Send activations between stages | Moderate | Prefer same node, can span nodes |
| **Data Parallel (DP)** | Allreduce gradients once per step | Lower | Can be anywhere |

You are designing a scheduler that places model-parallel workloads onto the cluster while respecting these topology constraints.

---

## Part 1: Cluster Topology

```python
@dataclass
class GPU:
    gpu_id: int
    node_id: int
    memory_gb: int = 80        # e.g., A100 80GB
    memory_used_gb: float = 0.0
    is_healthy: bool = True

@dataclass
class ClusterTopology:
    num_nodes: int
    gpus_per_node: int = 8
    intra_node_bw_gbps: float = 600.0   # NVLink bandwidth (GB/s)
    inter_node_bw_gbps: float = 50.0    # InfiniBand bandwidth (GB/s)
```

### 1. `__init__(self, topology: ClusterTopology)`

Initialize the scheduler with the cluster topology. Create all GPU objects.

### 2. `bandwidth_between(self, gpu_a: int, gpu_b: int) -> float`

Return the bandwidth in GB/s between two GPUs:
- Same GPU: infinity (or a very large number).
- Same node: `intra_node_bw_gbps` (NVLink).
- Different nodes: `inter_node_bw_gbps` (InfiniBand).

---

## Part 2: Model Placement

A model-parallel job specifies its parallelism requirements:

```python
@dataclass
class Job:
    job_id: str
    tensor_parallel_size: int    # GPUs within a TP group (high communication)
    pipeline_parallel_size: int  # number of pipeline stages (moderate communication)
    data_parallel_size: int      # independent replicas (low communication)
    memory_per_gpu_gb: float     # memory required per GPU
    # Communication parameters (for cost estimation)
    hidden_dim: int = 4096
    seq_len: int = 2048
    micro_batch_size: int = 1
    num_params_billions: float = 70.0
    dtype_bytes: int = 2         # FP16
```

**Total GPUs needed** = `tensor_parallel_size * pipeline_parallel_size * data_parallel_size`.

### 3. `place_job(self, job: Job) -> dict[str, list[list[int]]] | None`

Assign GPU IDs to the job while respecting these constraints:

1. **Tensor parallel groups MUST be within the same node.** TP requires allreduce after every layer -- this is the highest-bandwidth communication. If `tensor_parallel_size > gpus_per_node`, placement is impossible.

2. **Pipeline parallel stages SHOULD prefer the same node, but can span nodes.** PP only sends activations between adjacent stages.

3. **Data parallel replicas can be anywhere.** DP only communicates gradients once per training step.

Return a placement dictionary:
```python
{
    "tp_groups": [[gpu_ids], ...],      # each inner list is one TP group
    "pp_stages": [[gpu_ids], ...],      # each inner list is one PP stage
    "dp_replicas": [[gpu_ids], ...],    # each inner list is one DP replica
}
```

Or return `None` if placement is impossible (not enough GPUs, not enough memory, TP size exceeds node size).

**Hierarchical placement**: the standard approach is:
- TP within a node (innermost).
- PP across GPUs or nodes (middle).
- DP across nodes (outermost).

### 4. `estimate_communication_cost(self, job: Job, placement: dict) -> dict`

Estimate communication time for one training step using a simplified model (`time = data_size / bandwidth`):

- **TP allreduce**: each TP group does an allreduce of activation tensors after each layer. Data size per allreduce = `seq_len * hidden_dim * dtype_bytes` (simplified). Use the bandwidth between GPUs in the TP group. Assume the ring allreduce algorithm: time = `2 * (tp_size - 1) / tp_size * data_size / bandwidth`.

- **PP send/recv**: between adjacent pipeline stages. Data size = `micro_batch_size * seq_len * hidden_dim * dtype_bytes`. Use the bandwidth between the GPUs of adjacent stages.

- **DP allreduce**: across all DP replicas. Data size = `num_params_billions * 1e9 * dtype_bytes` (gradient size). Use the **minimum** bandwidth among any pair of DP replicas (bottleneck link). Ring allreduce time = `2 * (dp_size - 1) / dp_size * data_size / bandwidth`.

Return:
```python
{
    "tp_time_ms": float,
    "pp_time_ms": float,
    "dp_time_ms": float,
    "total_ms": float,         # sum of all three
    "bottleneck": str,         # "tp", "pp", or "dp" — whichever is largest
}
```

---

## Part 3: Fault Tolerance

### 5. `handle_gpu_failure(self, failed_gpu_id: int) -> list[dict]`

When a GPU fails:
1. Mark it as unhealthy.
2. Identify all jobs that use the failed GPU.
3. For each affected job, attempt to reschedule onto healthy GPUs (maintaining all topology constraints).
4. If rescheduling is impossible, the job is killed.

Return a list of actions:
```python
[
    {
        "job_id": str,
        "action": "rescheduled" | "killed",
        "old_placement": dict,         # previous GPU assignment
        "new_placement": dict | None,  # new assignment, or None if killed
    }
]
```

### 6. `find_checkpointable_boundary(self, job: Job) -> int`

For a pipeline-parallel job, determine the optimal pipeline stage boundary for checkpointing:
- The checkpoint should be placed at the stage boundary that **minimizes the activation tensor size** being checkpointed.
- For a uniform model, this is the boundary between any two stages (all activations are the same size). But for models with varying hidden dimensions or stages with different numbers of layers, it matters.
- For simplicity, return the stage index (0-indexed) that is the **midpoint** of the pipeline (minimizes recomputation on recovery).

---

## Example

```python
topo = ClusterTopology(num_nodes=4)  # 32 GPUs total
scheduler = GPUClusterScheduler(topo)

# A 70B model with 3D parallelism
job = Job(
    job_id="llama-70b",
    tensor_parallel_size=4,     # 4-way TP within a node
    pipeline_parallel_size=2,   # 2 pipeline stages
    data_parallel_size=2,       # 2 DP replicas
    memory_per_gpu_gb=70.0,
)
# Total GPUs = 4 * 2 * 2 = 16

placement = scheduler.place_job(job)
assert placement is not None

# Verify TP groups are within same node
for tp_group in placement["tp_groups"]:
    nodes = {gpu_id // 8 for gpu_id in tp_group}
    assert len(nodes) == 1, "TP group must be within one node"

# Estimate communication cost
cost = scheduler.estimate_communication_cost(job, placement)
print(f"TP: {cost['tp_time_ms']:.1f} ms")
print(f"PP: {cost['pp_time_ms']:.1f} ms")
print(f"DP: {cost['dp_time_ms']:.1f} ms")
print(f"Bottleneck: {cost['bottleneck']}")

# Second job: smaller model
job2 = Job(
    job_id="llama-7b",
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
    data_parallel_size=4,
    memory_per_gpu_gb=30.0,
    num_params_billions=7.0,
)
# Total GPUs = 2 * 1 * 4 = 8
placement2 = scheduler.place_job(job2)
assert placement2 is not None

# Simulate GPU failure
actions = scheduler.handle_gpu_failure(failed_gpu_id=0)
for a in actions:
    print(f"Job {a['job_id']}: {a['action']}")
```

---

## Constraints

- GPU IDs are globally unique integers: node `n` has GPUs `n * gpus_per_node` to `(n+1) * gpus_per_node - 1`.
- Memory checking: a GPU cannot be assigned to a job if `memory_used_gb + job.memory_per_gpu_gb > memory_gb`.
- A GPU can only be assigned to one job at a time.
- For communication cost estimation, use the simplified model described (not a detailed network simulator).
- Focus on placement correctness and topology awareness, not on scheduling fairness or queueing.
