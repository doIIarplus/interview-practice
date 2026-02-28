# Rubric: Kernel Fusion Simulator

**Total: 100 points**

---

## 1. Correct Graph Construction (10 points)

| Points | Criteria |
|--------|----------|
| 10 | `build_graph` correctly builds the DAG: `tensor_producer` maps each output tensor to its producing Op; `tensor_consumers` maps each tensor to all Ops that consume it; `predecessors` and `successors` are derived correctly from tensor-level edges. |
| 7 | Graph is mostly correct but missing one of the four fields, or has an off-by-one in consumer tracking. |
| 4 | Builds some form of adjacency structure but it is incomplete or uses a different schema without adapting downstream code. |
| 0 | Not implemented or fundamentally broken. |

### Key things to look for
- Does `tensor_consumers` correctly list ALL consumers of a tensor (important for diamond-dependency detection)?
- Are external inputs (tensors not produced by any op, e.g., weights) handled gracefully (not crashing)?
- Is the graph usable by downstream functions (`find_fusion_groups`, `estimate_memory_traffic`)?

---

## 2. Correct Fusion Group Identification (25 points)

| Points | Criteria |
|--------|----------|
| 25 | All five rules are correctly implemented: (1) consecutive elementwise fuse, (2) matmul + elementwise epilogue fuse, (3) reduction + preceding elementwise fuse, (4) matmul-matmul boundary, (5) multi-consumer boundary. Every op appears in exactly one group. Groups are in topological order. |
| 20 | Four of five rules correct, or minor issue with group ordering. |
| 15 | Core cases work (matmul + epilogue, elementwise chain) but multi-consumer or reduction boundary is wrong. |
| 10 | Only the simplest case (all elementwise) works. |
| 5 | Attempts fusion logic but produces incorrect groups. |
| 0 | Not implemented. |

### Key rules to verify

1. **Elementwise chain**: `[ew1, ew2, ew3]` -> one group.
2. **MatMul epilogue**: `[matmul, bias_add, relu]` -> one group. The matmul "anchors" the group.
3. **Reduction boundary**: `[ew, reduction]` fuse; next op starts a new group.
4. **MatMul-MatMul**: two consecutive matmuls are separate groups.
5. **Multi-consumer**: if `op_A.output` is consumed by both `op_B` and `op_C`, then `op_A` cannot be fused with either (its output must be materialized). `op_B` and `op_C` start their own groups.

### Acceptable variations
- Some candidates may allow reshape ops to be fused into any adjacent group (since reshapes are "free"). This is acceptable as long as they don't violate other rules.
- The exact algorithm (greedy forward scan vs. backward scan vs. graph coloring) doesn't matter as long as the output is correct.

---

## 3. Memory Traffic Estimation -- Unfused (15 points)

| Points | Criteria |
|--------|----------|
| 15 | Correctly computes total bytes read and written when every op is its own kernel. Each op reads all its input tensors and writes its output tensor. External inputs (weights, activations not produced by any op) are counted as reads. Tensor size = `prod(shape) * bytes_per_element`. |
| 10 | Mostly correct but miscounts shared inputs (e.g., a weight tensor read by multiple ops counted only once instead of once per consumer, or vice versa). |
| 5 | Computes some traffic but with significant errors. |
| 0 | Not implemented. |

### Important detail
- For unfused execution, if tensor `y1` is produced by `op_A` and consumed by `op_B`, it is written once (by `op_A`) and read once (by `op_B`). If `y1` is consumed by both `op_B` and `op_C`, it is written once and read twice.
- External inputs (not produced by any op in the graph) are read but not written.

---

## 4. Memory Traffic Estimation -- Fused (20 points)

| Points | Criteria |
|--------|----------|
| 20 | Correctly identifies intermediate tensors within fusion groups and eliminates their read/write from the traffic count. Only external inputs to the group are read; only the final output(s) of the group are written. The `saved_by_fusion` field is the difference between unfused and fused totals. |
| 15 | Intermediate elimination is correct for simple groups but breaks for groups with multiple inputs or outputs. |
| 10 | Attempts fusion-aware counting but the intermediate identification is wrong (e.g., eliminates tensors that cross group boundaries). |
| 5 | Returns some reduced number but logic is not defensible. |
| 0 | Not implemented or same as unfused. |

### What makes this tricky
- A tensor is "internal" to a group only if its producer AND all of its consumers are in the same group.
- If a tensor is consumed by an op outside the group, it must still be written to global memory.
- Weight tensors are always external reads.

---

## 5. Transformer Block Analysis (15 points)

| Points | Criteria |
|--------|----------|
| 15 | Correctly models all 14 operations with realistic shapes. Fusion groups are plausible (e.g., `out_proj + residual_add1`, `ffn_up + gelu`, etc.). Traffic numbers are in the right ballpark. Returns all required fields. |
| 10 | Operations are modeled but some shapes are wrong, or fusion groups are suboptimal but not incorrect. |
| 5 | Partial implementation; some operations missing or analysis incomplete. |
| 0 | Not implemented. |

### Expected fusion groups (approximate)
A strong candidate will identify groups like:
- `[qkv_proj, split_heads]` (matmul + reshape epilogue)
- `[attn_scores]` (standalone matmul; or fused with softmax depending on interpretation)
- `[softmax]` or `[attn_scores, softmax]`
- `[attn_output, concat_heads]` (matmul + reshape epilogue)
- `[out_proj, residual_add1]` (matmul + elementwise epilogue)
- `[layer_norm1]` (reduction)
- `[ffn_up, gelu]` (matmul + elementwise epilogue)
- `[ffn_down, residual_add2]` (matmul + elementwise epilogue)
- `[layer_norm2]` (reduction)

Note: the exact groups depend on whether `input` is consumed by `residual_add1` (multi-consumer handling). Strong candidates will notice this.

---

## 6. Edge Cases (10 points)

| Points | Criteria |
|--------|----------|
| 10 | Handles: (a) diamond dependencies correctly, (b) single-op graph, (c) all-elementwise chain, (d) graph with no fusable ops, (e) ops with different shapes/dtypes in the same chain. |
| 7 | Handles most edge cases but one or two break. |
| 4 | Basic cases work but edge cases crash or produce wrong results. |
| 0 | No edge case handling. |

---

## 7. Code Clarity (5 points)

| Points | Criteria |
|--------|----------|
| 5 | Clean, well-structured code. Functions are modular. Good variable names. Comments explain non-obvious decisions (e.g., why a particular fusion rule applies). |
| 3 | Readable but could be better organized. |
| 1 | Works but hard to follow. |
| 0 | Incomprehensible. |

---

## Grading Thresholds

| Grade | Points | Notes |
|-------|--------|-------|
| Strong Hire | 85-100 | All parts work correctly. Transformer analysis is realistic. Edge cases handled. |
| Hire | 65-84 | Core fusion logic and memory estimation work. Minor issues with edge cases or transformer analysis. |
| Lean Hire | 45-64 | Basic fusion works but memory estimation or transformer block has significant errors. |
| No Hire | < 45 | Fundamental misunderstanding of fusion rules or memory traffic accounting. |

---

## Red Flags
- Confusing global memory traffic with compute FLOPs.
- Not understanding that fusion eliminates memory reads AND writes of intermediates.
- Treating all ops as independently fusable without respecting dependency order.
- Ignoring the multi-consumer constraint (this is the subtlest rule and trips up many candidates).

## Green Flags
- Mentions that fusion analysis is essentially operator scheduling / graph partitioning.
- Discusses real-world fusion in Triton, XLA, or TVM.
- Notes that reshape ops are "free" (no data movement) and handles them specially.
- Considers that softmax is really a reduction + elementwise (exp, sum, divide) and could be partially fused.
- Mentions that Flash Attention is essentially a hand-written fused kernel for the entire attention block.
