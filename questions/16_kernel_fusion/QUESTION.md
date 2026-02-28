# Question 16: Kernel Fusion Simulator

## Background

In GPU computing, each kernel launch reads its inputs from **global memory** (HBM) and writes its outputs back. When you have a sequence of operations -- for example, `matmul -> bias add -> ReLU -> dropout` -- launching them as separate kernels means each intermediate result is written to and then read from global memory. This is expensive: on an A100, global memory bandwidth is ~2 TB/s, but a single matmul can produce intermediate data far faster than memory can absorb it.

**Kernel fusion** combines multiple operations into a single kernel, keeping intermediate results in registers or shared memory (which are orders of magnitude faster than global memory). This is one of the most impactful optimizations in GPU performance engineering.

You are building a **kernel fusion optimizer** that analyzes a computation graph and determines which operations can be fused together.

---

## Part 1: Operation Graph

Model GPU operations using the following types:

| Op Type       | Description                                           | Fusion Rules                                                             |
|---------------|-------------------------------------------------------|--------------------------------------------------------------------------|
| `elementwise` | Operates element-by-element (ReLU, GELU, add, mul, dropout) | Can always be fused with adjacent operations.                            |
| `reduction`   | Reduces along an axis (sum, mean, softmax normalizer) | Can be fused with preceding elementwise ops, but creates a **fusion boundary after** it. |
| `matmul`      | Matrix multiplication (heavy compute)                 | Creates a fusion boundary. Elementwise ops immediately after CAN be fused as the **matmul epilogue**. |
| `reshape`     | Changes tensor shape without data copy                | Free if fused, but changes indexing patterns.                            |

Use this data model:

```python
@dataclass
class Op:
    name: str
    op_type: str           # "elementwise", "reduction", "matmul", "reshape"
    inputs: list[str]      # tensor names consumed
    output: str            # tensor name produced
    shape: tuple[int, ...] # output shape
    bytes_per_element: int  # e.g., 4 for float32, 2 for float16
```

---

## Part 2: Fusion Analysis

Implement the following functions:

### 1. `build_graph(ops: list[Op]) -> dict`

Build a DAG (directed acyclic graph) from the operations. The graph should capture:
- Which op produces each tensor.
- Which ops consume each tensor.
- Adjacency information (predecessors and successors of each op).

### 2. `find_fusion_groups(ops: list[Op]) -> list[list[Op]]`

Identify groups of operations that can be fused into single GPU kernels. Follow these rules:

- **Consecutive elementwise ops** -> fuse into one kernel.
- **MatMul + elementwise epilogue** -> fuse (e.g., matmul followed by bias add and activation).
- **Reduction + preceding elementwise ops** -> fuse.
- **MatMul cannot be fused with another MatMul.**
- **An op whose output is consumed by multiple downstream ops** cannot have its output eliminated by fusion (the intermediate tensor must be materialized in global memory because multiple consumers need it).

Each op must appear in exactly one fusion group.

### 3. `estimate_memory_traffic(ops: list[Op], fused: bool = False) -> dict`

Estimate total bytes read from and written to global memory.

- **Unfused**: every op reads all its inputs from global memory and writes its output to global memory. Each tensor's size = `product(shape) * bytes_per_element`.
- **Fused**: intermediate tensors within a fusion group are NOT materialized in global memory. Only the group's external inputs are read and external outputs are written.

Return:
```python
{
    "bytes_read": int,       # total bytes read from global memory
    "bytes_written": int,    # total bytes written to global memory
    "total": int,            # bytes_read + bytes_written
    "saved_by_fusion": int   # difference vs. unfused total (0 if fused=False)
}
```

---

## Part 3: Transformer Block Analysis

### 4. `analyze_transformer_block(seq_len: int, hidden_dim: int, num_heads: int, dtype_bytes: int = 2) -> dict`

Model a standard transformer self-attention + FFN block with these operations (in order):

1. **QKV projection** (matmul): input `(seq_len, hidden_dim)` x weight `(hidden_dim, 3*hidden_dim)` -> output `(seq_len, 3*hidden_dim)`
2. **Split heads** (reshape): -> `(num_heads, seq_len, head_dim)` for Q, K, V each
3. **Attention scores** (matmul): Q x K^T -> `(num_heads, seq_len, seq_len)`
4. **Softmax** (reduction): -> `(num_heads, seq_len, seq_len)`
5. **Attention output** (matmul): scores x V -> `(num_heads, seq_len, head_dim)`
6. **Concat heads** (reshape): -> `(seq_len, hidden_dim)`
7. **Output projection** (matmul): -> `(seq_len, hidden_dim)`
8. **Residual add** (elementwise): -> `(seq_len, hidden_dim)`
9. **Layer norm** (reduction + elementwise): -> `(seq_len, hidden_dim)`
10. **FFN up projection** (matmul): -> `(seq_len, 4*hidden_dim)` (standard 4x expansion)
11. **GELU** (elementwise): -> `(seq_len, 4*hidden_dim)`
12. **FFN down projection** (matmul): -> `(seq_len, hidden_dim)`
13. **Residual add** (elementwise): -> `(seq_len, hidden_dim)`
14. **Layer norm** (reduction + elementwise): -> `(seq_len, hidden_dim)`

Compute memory traffic for unfused vs. optimally fused execution.

Return:
```python
{
    "ops": list[Op],                    # the operations modeled
    "fusion_groups": list[list[str]],   # groups as lists of op names
    "unfused_traffic": dict,            # from estimate_memory_traffic
    "fused_traffic": dict,              # from estimate_memory_traffic
    "savings_pct": float                # percentage reduction in total traffic
}
```

---

## Example

```python
ops = [
    Op("matmul1", "matmul", ["x", "w1"], "y1", (1024, 4096), 2),
    Op("bias1", "elementwise", ["y1", "b1"], "y2", (1024, 4096), 2),
    Op("relu", "elementwise", ["y2"], "y3", (1024, 4096), 2),
    Op("matmul2", "matmul", ["y3", "w2"], "y4", (1024, 1024), 2),
]

groups = find_fusion_groups(ops)
# Expected: [[matmul1, bias1, relu], [matmul2]]
# matmul1's epilogue fuses bias_add and relu; matmul2 starts a new group

traffic_unfused = estimate_memory_traffic(ops, fused=False)
# Each op reads/writes everything to global memory

traffic_fused = estimate_memory_traffic(ops, fused=True)
# y1 and y2 are NOT written/read from global memory (internal to group 1)
# Savings come from eliminating those intermediate tensors
```

---

## Constraints

- All tensor sizes are given as shapes + bytes_per_element. Compute size as `product(shape) * bytes_per_element`.
- Assume all reads and writes are to global memory (no cache modeling).
- Weight tensors (those not produced by any op in the graph) are always read from global memory.
- Focus on correctness of the fusion rules and memory accounting, not on actual kernel code generation.
