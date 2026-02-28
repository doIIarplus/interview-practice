# Rubric: GPU Cluster Scheduler

**Total: 100 points**

---

## 1. Correct Topology Bandwidth Model (10 points)

| Points | Criteria |
|--------|----------|
| 10 | `bandwidth_between` correctly returns: `inf` (same GPU), `intra_node_bw` (same node), `inter_node_bw` (different nodes). Node membership is derived from GPU ID: `node_id = gpu_id // gpus_per_node`. GPU objects are created correctly with proper node assignments. |
| 7 | Bandwidth logic is correct but node membership derivation has an off-by-one or hardcodes `gpus_per_node=8`. |
| 3 | Returns some bandwidth values but the node derivation is wrong. |
| 0 | Not implemented. |

### Key check
```python
scheduler.bandwidth_between(7, 8)  # different nodes: 50 GB/s
scheduler.bandwidth_between(7, 0)  # same node: 600 GB/s
scheduler.bandwidth_between(5, 5)  # same GPU: inf
```

---

## 2. TP Groups Placed Within Same Node (20 points)

| Points | Criteria |
|--------|----------|
| 20 | TP groups are ALWAYS within the same node. The scheduler validates `tp_size <= gpus_per_node` and returns `None` if violated. Multiple TP groups across different DP replicas / PP stages are each correctly constrained to single nodes. |
| 15 | TP constraint is usually satisfied but can break in edge cases (e.g., when nodes are partially full from other jobs). |
| 10 | TP constraint works for the basic case but not when combined with PP and DP. |
| 5 | Attempts to enforce TP locality but the node-selection logic is broken. |
| 0 | Not implemented or TP groups span nodes. |

### What to verify
- `tp_size=4, pp=2, dp=2` on 4 nodes: each of the 4 TP groups (4 GPUs each) must be within one node.
- `tp_size=8, pp=1, dp=1` on 1 node: uses the entire node.
- `tp_size=8, pp=1, dp=2`: needs 2 full nodes.
- `tp_size=16`: returns `None` (impossible on 8-GPU nodes).

---

## 3. Correct Total GPU Calculation and Capacity Checking (10 points)

| Points | Criteria |
|--------|----------|
| 10 | `total_gpus = tp * pp * dp` is used correctly. Placement fails if total_gpus > available healthy GPUs. Memory constraint is checked per GPU: `memory_used + job.memory_per_gpu > gpu.memory` -> skip that GPU. Duplicate job IDs are rejected. |
| 7 | GPU count and memory checks work but edge cases (all GPUs busy, fractional memory) are not handled. |
| 4 | Total GPU calculation correct but memory isn't checked. |
| 0 | Not implemented or fundamentally wrong. |

### Important details
- A GPU assigned to job A cannot be assigned to job B.
- Memory is tracked: if a GPU has 80GB and job A uses 40GB, job B can potentially share it only if the implementation supports multi-tenancy. The problem says "one job at a time" per GPU, so this shouldn't be allowed.
- After a job is freed (from failure handling), its GPUs become available again.

---

## 4. Communication Cost Estimation with Correct Bandwidth Lookups (15 points)

| Points | Criteria |
|--------|----------|
| 15 | All three communication costs computed correctly using the right bandwidth. TP uses intra-node bandwidth. PP uses bandwidth between the specific GPU pair (may be intra or inter-node). DP uses the minimum bandwidth among replica pairs. Ring allreduce formula applied correctly. Results in milliseconds. |
| 12 | Two of three costs correct, or bandwidth lookups are correct but the allreduce formula is wrong. |
| 8 | Computes some cost but uses wrong bandwidth or wrong data sizes. |
| 4 | Returns hardcoded or incorrect values. |
| 0 | Not implemented. |

### Expected formulas
```
TP allreduce:
    data = seq_len * hidden_dim * dtype_bytes
    time_s = 2 * (tp_size - 1) / tp_size * data / bw_intra_node
    tp_time_ms = time_s * 1000

PP send/recv:
    data = micro_batch_size * seq_len * hidden_dim * dtype_bytes
    time_s = data / bw_between_stages
    pp_time_ms = time_s * 1000

DP allreduce:
    data = num_params * 1e9 * dtype_bytes
    time_s = 2 * (dp_size - 1) / dp_size * data / min_bw_across_replicas
    dp_time_ms = time_s * 1000
```

### Sanity check
For `tp=4, hidden=4096, seq=2048, fp16` on NVLink:
- TP data = 2048 * 4096 * 2 = 16 MB
- TP time = 2 * 3/4 * 16M / 600G = 0.04 ms (very fast due to NVLink)

For DP with 70B params, fp16 across InfiniBand:
- DP data = 70e9 * 2 = 140 GB
- DP time = 2 * 1/2 * 140G / 50G = 2800 ms (very slow -- this is the bottleneck)

---

## 5. Fault Tolerance: Identify Affected Jobs, Attempt Reschedule (20 points)

| Points | Criteria |
|--------|----------|
| 20 | GPU is marked unhealthy. All jobs using that GPU are found. For each job: (a) old placement is recorded, (b) GPU resources are freed, (c) `place_job` is re-called to find a new placement on healthy GPUs, (d) if re-placement succeeds, new placement is returned; otherwise "killed". Failed GPU is excluded from future placements. |
| 15 | Failure handling works for the single-job case but breaks when multiple jobs are affected or when the freed GPUs from the failed job are needed for rescheduling. |
| 10 | Identifies affected jobs but rescheduling doesn't work correctly. |
| 5 | Marks GPU as failed but doesn't handle affected jobs. |
| 0 | Not implemented. |

### Subtle point: cascading failure
When a GPU fails:
1. Free ALL GPUs of the affected job (not just the failed one).
2. Try to re-place the job. The freed healthy GPUs are now available for the new placement.
3. If re-placement fails (not enough healthy GPUs/nodes), the job is killed.

Strong candidates will handle this correctly without double-freeing or leaving orphaned state.

---

## 6. Handling Fragmentation: Multiple Jobs Competing for Resources (10 points)

| Points | Criteria |
|--------|----------|
| 10 | Placement algorithm handles partial node occupancy: if node 0 has 4 GPUs used and 4 free, a `tp=4` group can still be placed there. Multiple jobs can coexist on different portions of the cluster. First-fit or best-fit strategy is used intelligently. |
| 7 | Works when jobs cleanly fill nodes but breaks when nodes are partially occupied. |
| 4 | Can place one job but second job fails even though resources are available. |
| 0 | No multi-job support. |

### Test scenario
```
4 nodes, 32 GPUs
Job 1: tp=4, pp=2, dp=2 -> 16 GPUs (uses 2 nodes fully, or 4 nodes partially)
Job 2: tp=2, pp=1, dp=4 -> 8 GPUs (should fit in remaining GPUs)
```

---

## 7. Edge Cases (10 points)

| Points | Criteria |
|--------|----------|
| 10 | Handles: (a) `tp_size > gpus_per_node` -> None, (b) `total_gpus > cluster_size` -> None, (c) `memory_per_gpu > gpu.memory_gb` -> None, (d) placing a job that exactly fills the cluster, (e) failure of a GPU not used by any job (no-op), (f) `pp=1` (no pipeline, single stage), (g) `dp=1` (no data parallelism). |
| 7 | Most edge cases work but one or two crash. |
| 4 | Basic cases work but edge cases are unhandled. |
| 0 | No edge case handling. |

---

## 8. Code Clarity (5 points)

| Points | Criteria |
|--------|----------|
| 5 | Clean, well-structured code. Placement algorithm is readable and commented. Helper methods are used appropriately. Good variable names (not `n`, `x`, `tmp`). |
| 3 | Readable but could be better organized. Placement logic is one long function. |
| 1 | Works but hard to follow. |
| 0 | Incomprehensible. |

---

## Grading Thresholds

| Grade | Points | Notes |
|-------|--------|-------|
| Strong Hire | 85-100 | All parts work. TP constraint always enforced. Communication costs are realistic. Fault tolerance handles rescheduling correctly. |
| Hire | 65-84 | Placement and TP constraint work well. Communication cost has minor issues. Fault tolerance mostly works. |
| Lean Hire | 45-64 | Basic placement works but multi-job or fault tolerance is broken. Understanding of parallelism concepts is solid. |
| No Hire | < 45 | Cannot place a job correctly or doesn't understand the TP locality constraint. |

---

## Red Flags
- Placing TP groups across nodes (fundamental misunderstanding of the bandwidth hierarchy).
- Not tracking GPU occupancy (allowing double-allocation).
- Ignoring memory constraints.
- Confusing the three parallelism dimensions (e.g., thinking DP needs high bandwidth).
- Treating all GPU pairs as having the same bandwidth.

## Green Flags
- Explains WHY TP must be within a node (allreduce after every layer, O(layers) communications per step).
- Discusses the trade-off: more TP = more intra-node communication but less memory per GPU.
- Mentions that in practice, the placement is hierarchical: TP innermost, DP outermost.
- Notes that PP has "bubble" overhead (idle time) proportional to `(pp_stages - 1) / micro_batches`.
- Discusses how NVSwitch vs. pairwise NVLink affects the intra-node topology (NVSwitch gives full bisection bandwidth).
- Considers affinity: PP stages in the same DP replica should be nearby.
- Mentions NCCL (NVIDIA Collective Communication Library) and how it selects algorithms based on topology.
- Discusses gradient compression or gradient accumulation as alternatives to frequent DP allreduce.
