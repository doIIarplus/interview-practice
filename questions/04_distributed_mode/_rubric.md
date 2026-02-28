# Rubric: Distributed Mode Finding

**Total: 100 points**

---

## 1. Correct Mode Computation (25 points)

| Points | Criteria |
|--------|----------|
| 25 | Returns correct mode for all test cases including tie-breaking (smallest value) |
| 15 | Returns correct mode for basic cases but fails edge cases (ties, single node) |
| 5  | Attempts to compute mode but has logical errors |
| 0  | Does not return correct mode |

**Key checks:**
- Correct result on the default 10-node, 10,000-value cluster
- Correct tie-breaking: returns smallest value when multiple values share the max frequency
- Correct on single-node cluster (no network needed)
- Correct when all values are the same

---

## 2. Efficient Local Computation vs Network Communication (30 points)

| Points | Criteria |
|--------|----------|
| 30 | Each node computes local frequency counts; only counts are sent over network |
| 20 | Computes locally but sends more data than necessary (e.g., sorted data, partial raw) |
| 10 | Sends raw data but recognizes it's inefficient |
| 0  | Sends all raw data to one node naively |

**Ideal approach:**
1. Each node reads its local data via `read_local_data(node_id)`
2. Each node computes a local frequency dictionary: `{value: count}`
3. Non-coordinator nodes serialize their frequency dict and send to coordinator (node 0)
4. Coordinator merges all frequency dicts and finds the global mode

**Why this is optimal:**
- Local reads: ~1,000 ints * 4 bytes / 10 bytes/sec = ~400s per node (but parallelizable)
- Sending counts: at most 1,000 entries per node (value-count pairs)
- Sending raw data: ~1,000 ints * 4 bytes per node = 4,000 bytes per node at 1 byte/sec

---

## 3. Minimizing Total Data Sent Over Network (20 points)

| Points | Criteria |
|--------|----------|
| 20 | Network bytes sent is proportional to unique values (not raw data size); uses compact serialization |
| 15 | Sends frequency counts but uses verbose serialization (e.g., uncompressed JSON) |
| 10 | Sends partial aggregations but still sends more than necessary |
| 5  | Sends raw data but attempts some optimization |
| 0  | No attempt to minimize network traffic |

**Benchmarks (approximate for 10 nodes, 10,000 values in [0,1000)):**
- Naive (raw data): ~40,000 bytes sent -> ~40,000 simulated seconds
- Counts via JSON: ~5,000-15,000 bytes sent -> significant savings
- Counts via compact binary: ~2,000-8,000 bytes sent -> best
- Only the coordinator's own data is "free" (no send needed for node 0)

**Bonus considerations:**
- Candidate realizes node 0 doesn't need to send to itself
- Candidate uses `json.dumps` or `struct.pack` for compact serialization
- Candidate considers parallel aggregation (tree reduction)

---

## 4. Code Clarity and Correctness of Distributed Logic (15 points)

| Points | Criteria |
|--------|----------|
| 15 | Clean separation of local computation and communication phases; proper use of barrier; well-structured code |
| 10 | Correct logic but somewhat disorganized or missing barrier usage |
| 5  | Logic works but is convoluted or hard to follow |
| 0  | Incorrect distributed logic (e.g., race conditions, missing synchronization) |

**Look for:**
- Clear phase separation: (1) local read, (2) local computation, (3) send, (4) barrier, (5) aggregate
- Proper use of `barrier()` between send and receive phases
- Good variable names and comments explaining the distributed protocol
- Helper functions for serialization/deserialization

---

## 5. Handling Edge Cases (10 points)

| Points | Criteria |
|--------|----------|
| 10 | Handles all edge cases correctly |
| 7  | Handles most edge cases |
| 3  | Handles basic case only |
| 0  | Fails on edge cases |

**Edge cases to check:**
- **Tie-breaking**: multiple values with same max frequency -> return smallest
- **Single node**: no network communication needed
- **All same value**: every element is identical
- **Single element per node**: minimal data
- **Empty consideration**: what if a node has no data? (not tested but good to discuss)

---

## Ideal Solution Outline

```python
def find_mode(cluster: Cluster) -> int:
    from collections import Counter
    import json

    COORDINATOR = 0

    # Phase 1: Each node reads local data and computes local frequency counts
    local_counts = {}
    for node_id in range(cluster.num_nodes):
        data = cluster.read_local_data(node_id)
        local_counts[node_id] = Counter(data)

    # Phase 2: Non-coordinator nodes send their counts to the coordinator
    for node_id in range(1, cluster.num_nodes):
        payload = json.dumps(dict(local_counts[node_id])).encode()
        cluster.send(node_id, COORDINATOR, payload)

    cluster.barrier()

    # Phase 3: Coordinator aggregates all counts
    global_counts = Counter(local_counts[COORDINATOR])
    for message in cluster.get_received(COORDINATOR):
        remote_counts = json.loads(message.decode())
        for val_str, count in remote_counts.items():
            global_counts[int(val_str)] += count

    # Phase 4: Find mode with tie-breaking
    max_count = max(global_counts.values())
    mode = min(val for val, count in global_counts.items() if count == max_count)
    return mode
```

**Note on the simulation model:** The simulation is sequential (not truly parallel),
so "each node reads local data" translates to a loop. In a real distributed system,
these reads would happen in parallel. The key insight being tested is minimizing
network communication, not parallelism of local reads.

---

## Red Flags

- Sends raw data over the network without computing local counts
- Doesn't use `barrier()` at all
- Forgets to handle tie-breaking (returns arbitrary tied value)
- Over-engineers with actual threading/multiprocessing for a simulation
- Doesn't read local data at all (tries to access `cluster._node_data` directly)

## Green Flags

- Immediately recognizes the local-aggregation-then-merge pattern (MapReduce)
- Considers the serialization format and its impact on network bytes
- Mentions that in a real system, local reads would be parallel
- Discusses tree-structured aggregation as an optimization
- Considers the bounded range [0, 1000) as a way to use arrays instead of dicts
