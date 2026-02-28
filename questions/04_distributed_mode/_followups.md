# Follow-up Questions: Distributed Mode Finding

---

## 1. What if the range of values was unknown (not bounded to [0, 1000))?

**Expected discussion:**
- The local frequency count approach still works -- just use a dictionary/Counter
- The difference is you can no longer use a fixed-size array (1000 entries)
- Network payload size now depends on the number of **unique** values per node, not the range
- If the number of unique values is very large, you may need to consider:
  - Approximate algorithms (Count-Min Sketch, Heavy Hitters)
  - Multi-round communication (identify top candidates, then verify)
  - Sampling to estimate the mode

**Strong answer includes:** Recognition that the approach generalizes naturally but may
need approximation if the number of unique values is very large.

---

## 2. How would you find the median instead of the mode? Why is this significantly harder in a distributed setting?

**Expected discussion:**
- Median requires knowing the **global ordering** of all values, not just frequencies
- You cannot compute local medians and combine them (local medians don't compose)
- Approaches:
  - **Selection algorithm**: Binary search on the value range; ask each node "how many values are <= X?" and aggregate. O(log(range) * communication_rounds)
  - **Sampling**: Take a random sample from each node, compute approximate median, then refine
  - **Full sort**: Send all data to one node (expensive but correct)
- Key insight: mode is an **embarrassingly parallel** statistic (decomposable), while median is **order-dependent** and requires global coordination

**Strong answer includes:** Clear explanation of why median doesn't decompose like mode,
and at least one concrete distributed median algorithm.

---

## 3. What if nodes can fail during computation? How would you add fault tolerance?

**Expected discussion:**
- **Replication**: Store data on multiple nodes so if one fails, others have copies
- **Checkpointing**: Save intermediate results (local counts) so failed nodes can recover
- **Timeout and retry**: If a node doesn't respond to barrier, retry or skip it
- **Heartbeat/health checks**: Detect failed nodes before relying on their data
- **Partial results**: If a node fails, can we still compute an approximate mode from the remaining data?
- **Coordinator failure**: What if node 0 fails? Need coordinator election (Raft, Paxos)

**Strong answer includes:** Distinguishes between data loss (need replication) and
compute failure (need retry/reassignment). Mentions coordinator as a single point of failure.

---

## 4. How would you parallelize the aggregation instead of sending everything to one coordinator?

**Expected discussion:**
- **Tree reduction**: Organize nodes in a binary tree
  - Round 1: nodes 1->0, 3->2, 5->4, 7->6, 9->8 (5 pairs)
  - Round 2: nodes 2->0, 6->4, 8->? (merge pairs)
  - Round 3: nodes 4->0 (final merge)
  - Total rounds: O(log N) instead of all-to-one
- **All-reduce**: Every node ends up with the global counts (useful if all need the answer)
- **Ring-based aggregation**: Pass partial aggregates around a ring of nodes
- Benefits: reduces bottleneck at coordinator, distributes network load, reduces latency
  in a real system where sends are truly parallel

**Strong answer includes:** Tree reduction with concrete analysis of rounds and bandwidth.
Recognizes that in the simulation (sequential sends), this doesn't help wall-clock time,
but in a real system with parallel communication it reduces latency from O(N) to O(log N).

---

## 5. What if the data doesn't fit in memory on any single node?

**Expected discussion:**
- The frequency-count approach already solves this for mode! Counts are much smaller than raw data
- Even if counts don't fit: use streaming/external algorithms
  - Process data in chunks, maintain running frequency counts
  - For mode with bounded range: fixed-size count array always fits
  - For unbounded range: use approximate counting (Count-Min Sketch)
- For the aggregation step: if merged counts don't fit on coordinator:
  - Use tree reduction so no single node holds all counts
  - Use external storage (disk-backed dictionary)
  - Use approximate data structures with bounded memory

**Strong answer includes:** Recognizes that the count-based approach naturally handles
this for bounded ranges. Discusses approximate algorithms for the unbounded case.
