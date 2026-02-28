"""
Question 16: Kernel Fusion Simulator

In GPU computing, kernel fusion combines multiple operations into a single kernel
to reduce global memory traffic. This module analyzes computation graphs to find
fusion opportunities and estimate memory savings.

See QUESTION.md for full problem description.
"""

from dataclasses import dataclass, field
from math import prod
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Op:
    """Represents a single GPU operation in a computation graph."""
    name: str
    op_type: str           # "elementwise", "reduction", "matmul", "reshape"
    inputs: list[str]      # tensor names consumed
    output: str            # tensor name produced
    shape: tuple[int, ...]  # output shape
    bytes_per_element: int  # e.g., 4 for float32, 2 for float16

    def output_bytes(self) -> int:
        """Total bytes of this op's output tensor."""
        return prod(self.shape) * self.bytes_per_element

    def __repr__(self) -> str:
        return f"Op({self.name!r}, {self.op_type!r})"


# ---------------------------------------------------------------------------
# Part 2: Fusion Analysis
# ---------------------------------------------------------------------------

def build_graph(ops: list[Op]) -> dict:
    """
    Build a DAG from the list of operations.

    Returns a dict with:
        - "ops": list[Op]                      — the operations in order
        - "tensor_producer": dict[str, Op]     — maps tensor name -> the Op that produces it
        - "tensor_consumers": dict[str, list[Op]] — maps tensor name -> list of Ops that consume it
        - "predecessors": dict[str, list[Op]]  — maps op name -> list of Ops that produce this op's inputs
        - "successors": dict[str, list[Op]]    — maps op name -> list of Ops that consume this op's output
    """
    # TODO: Implement this
    pass


def find_fusion_groups(ops: list[Op]) -> list[list[Op]]:
    """
    Identify groups of operations that can be fused into single GPU kernels.

    Fusion rules:
        - Consecutive elementwise ops -> fuse.
        - MatMul + following elementwise ops -> fuse (matmul epilogue).
        - Preceding elementwise ops + Reduction -> fuse.
        - MatMul cannot fuse with another MatMul.
        - Ops whose output has multiple consumers cannot be fused (output must
          be materialized so all consumers can read it).

    Every op appears in exactly one group. Returns groups in topological order.
    """
    # TODO: Implement this
    pass


def estimate_memory_traffic(ops: list[Op], fused: bool = False) -> dict:
    """
    Estimate total bytes read/written to global memory.

    Args:
        ops:   list of operations in the computation graph.
        fused: if True, use fusion groups to eliminate intermediate traffic.

    Returns:
        {
            "bytes_read": int,
            "bytes_written": int,
            "total": int,
            "saved_by_fusion": int,   # 0 when fused=False
        }
    """
    # TODO: Implement this
    pass


# ---------------------------------------------------------------------------
# Part 3: Transformer Block Analysis
# ---------------------------------------------------------------------------

def build_transformer_block_ops(
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    dtype_bytes: int = 2,
) -> list[Op]:
    """
    Helper: construct the list of Ops for one transformer block.

    Operations (in order):
        1.  qkv_proj      (matmul)      — (seq_len, hidden_dim) -> (seq_len, 3*hidden_dim)
        2.  split_heads    (reshape)     — -> (num_heads, seq_len, head_dim) conceptually x3
        3.  attn_scores    (matmul)      — Q @ K^T -> (num_heads, seq_len, seq_len)
        4.  softmax        (reduction)   — -> (num_heads, seq_len, seq_len)
        5.  attn_output    (matmul)      — scores @ V -> (num_heads, seq_len, head_dim)
        6.  concat_heads   (reshape)     — -> (seq_len, hidden_dim)
        7.  out_proj       (matmul)      — -> (seq_len, hidden_dim)
        8.  residual_add1  (elementwise) — -> (seq_len, hidden_dim)
        9.  layer_norm1    (reduction)   — -> (seq_len, hidden_dim)
        10. ffn_up         (matmul)      — -> (seq_len, 4*hidden_dim)
        11. gelu           (elementwise) — -> (seq_len, 4*hidden_dim)
        12. ffn_down       (matmul)      — -> (seq_len, hidden_dim)
        13. residual_add2  (elementwise) — -> (seq_len, hidden_dim)
        14. layer_norm2    (reduction)   — -> (seq_len, hidden_dim)
    """
    head_dim = hidden_dim // num_heads
    d = dtype_bytes

    ops = [
        Op("qkv_proj",     "matmul",      ["input", "w_qkv"],           "qkv",          (seq_len, 3 * hidden_dim), d),
        Op("split_heads",  "reshape",     ["qkv"],                      "qkv_split",    (num_heads, seq_len, head_dim * 3), d),
        Op("attn_scores",  "matmul",      ["qkv_split", "qkv_split"],   "scores_raw",   (num_heads, seq_len, seq_len), d),
        Op("softmax",      "reduction",   ["scores_raw"],               "scores",       (num_heads, seq_len, seq_len), d),
        Op("attn_output",  "matmul",      ["scores", "qkv_split"],      "attn_out",     (num_heads, seq_len, head_dim), d),
        Op("concat_heads", "reshape",     ["attn_out"],                 "attn_concat",  (seq_len, hidden_dim), d),
        Op("out_proj",     "matmul",      ["attn_concat", "w_out"],     "proj_out",     (seq_len, hidden_dim), d),
        Op("residual_add1","elementwise",  ["proj_out", "input"],       "resid1",       (seq_len, hidden_dim), d),
        Op("layer_norm1",  "reduction",   ["resid1"],                   "ln1",          (seq_len, hidden_dim), d),
        Op("ffn_up",       "matmul",      ["ln1", "w_ffn_up"],         "ffn_up_out",   (seq_len, 4 * hidden_dim), d),
        Op("gelu",         "elementwise", ["ffn_up_out"],               "gelu_out",     (seq_len, 4 * hidden_dim), d),
        Op("ffn_down",     "matmul",      ["gelu_out", "w_ffn_down"],  "ffn_down_out", (seq_len, hidden_dim), d),
        Op("residual_add2","elementwise",  ["ffn_down_out", "ln1"],     "resid2",       (seq_len, hidden_dim), d),
        Op("layer_norm2",  "reduction",   ["resid2"],                   "ln2",          (seq_len, hidden_dim), d),
    ]
    return ops


def analyze_transformer_block(
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    dtype_bytes: int = 2,
) -> dict:
    """
    Analyze memory traffic of one transformer block, unfused vs. fused.

    Returns:
        {
            "ops": list[Op],
            "fusion_groups": list[list[str]],    # groups as lists of op names
            "unfused_traffic": dict,
            "fused_traffic": dict,
            "savings_pct": float,                # percentage reduction
        }
    """
    # TODO: Implement this
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_fusion():
    """Test: matmul + elementwise epilogue fuses; second matmul is separate."""
    ops = [
        Op("matmul1", "matmul",      ["x", "w1"], "y1", (1024, 4096), 2),
        Op("bias1",   "elementwise", ["y1", "b1"], "y2", (1024, 4096), 2),
        Op("relu",    "elementwise", ["y2"],        "y3", (1024, 4096), 2),
        Op("matmul2", "matmul",      ["y3", "w2"], "y4", (1024, 1024), 2),
    ]

    # -- graph --
    graph = build_graph(ops)
    assert graph is not None, "build_graph returned None"
    assert graph["tensor_producer"]["y1"].name == "matmul1"
    assert len(graph["tensor_consumers"]["y1"]) == 1

    # -- fusion groups --
    groups = find_fusion_groups(ops)
    assert len(groups) == 2, f"Expected 2 fusion groups, got {len(groups)}"
    assert [op.name for op in groups[0]] == ["matmul1", "bias1", "relu"]
    assert [op.name for op in groups[1]] == ["matmul2"]

    # -- memory traffic --
    unfused = estimate_memory_traffic(ops, fused=False)
    fused = estimate_memory_traffic(ops, fused=True)
    assert unfused["total"] > fused["total"], "Fusion should reduce total traffic"
    assert fused["saved_by_fusion"] > 0
    assert unfused["saved_by_fusion"] == 0

    print("[PASS] test_basic_fusion")


def test_all_elementwise():
    """All elementwise ops should fuse into one group."""
    ops = [
        Op("add",  "elementwise", ["a", "b"], "c", (1024,), 4),
        Op("relu", "elementwise", ["c"],      "d", (1024,), 4),
        Op("mul",  "elementwise", ["d", "e"], "f", (1024,), 4),
    ]
    groups = find_fusion_groups(ops)
    assert len(groups) == 1, f"Expected 1 fusion group, got {len(groups)}"
    assert len(groups[0]) == 3
    print("[PASS] test_all_elementwise")


def test_diamond_dependency():
    """
    Diamond: one op's output feeds two consumers -> cannot eliminate intermediate.

        matmul1 -> y1
                    ├── relu  -> y2
                    └── gelu  -> y3
    """
    ops = [
        Op("matmul1", "matmul",      ["x", "w"], "y1", (1024, 1024), 2),
        Op("relu",    "elementwise", ["y1"],      "y2", (1024, 1024), 2),
        Op("gelu",    "elementwise", ["y1"],      "y3", (1024, 1024), 2),
    ]
    groups = find_fusion_groups(ops)
    # y1 has two consumers, so it must be materialized.
    # matmul1 cannot fuse its epilogue with relu because gelu also needs y1.
    # Each should be its own group (or matmul1 alone, relu alone, gelu alone).
    for g in groups:
        names = [op.name for op in g]
        assert not ("relu" in names and "gelu" in names), \
            "relu and gelu should not be in the same fusion group"
    print("[PASS] test_diamond_dependency")


def test_reduction_boundary():
    """Reduction fuses with preceding elementwise but creates boundary after."""
    ops = [
        Op("add",    "elementwise", ["a", "b"], "c", (1024,), 4),
        Op("sum",    "reduction",   ["c"],      "d", (1,),    4),
        Op("scale",  "elementwise", ["d", "e"], "f", (1024,), 4),
    ]
    groups = find_fusion_groups(ops)
    # add + sum should fuse; scale starts a new group (boundary after reduction).
    assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"
    assert "add" in [op.name for op in groups[0]]
    assert "sum" in [op.name for op in groups[0]]
    assert "scale" in [op.name for op in groups[1]]
    print("[PASS] test_reduction_boundary")


def test_single_op():
    """A single op is its own fusion group."""
    ops = [Op("matmul", "matmul", ["x", "w"], "y", (512, 512), 2)]
    groups = find_fusion_groups(ops)
    assert len(groups) == 1
    assert len(groups[0]) == 1
    print("[PASS] test_single_op")


def test_memory_traffic_values():
    """Verify exact memory traffic numbers for a simple case."""
    ops = [
        Op("matmul1", "matmul",      ["x", "w1"], "y1", (1024, 4096), 2),
        Op("relu",    "elementwise", ["y1"],        "y2", (1024, 4096), 2),
    ]
    unfused = estimate_memory_traffic(ops, fused=False)

    # Unfused matmul1: reads x (unknown shape — use output shape heuristic or
    # track input shapes). For simplicity, we count only tensors whose sizes
    # we know from op outputs.  External inputs are read; every output written.
    # y1 size = 1024*4096*2 = 8,388,608
    # y2 size = 1024*4096*2 = 8,388,608
    # unfused writes = y1 + y2 = 16,777,216

    fused = estimate_memory_traffic(ops, fused=True)
    # Fused: y1 is internal (matmul1 -> relu in same group), so not written/read.
    # fused writes = y2 only = 8,388,608
    # Saved = 2 * y1 (one write from matmul1 + one read by relu) = 16,777,216

    assert fused["bytes_written"] < unfused["bytes_written"]
    print("[PASS] test_memory_traffic_values")


def test_transformer_block():
    """Smoke test: transformer block analysis runs and produces savings."""
    result = analyze_transformer_block(seq_len=2048, hidden_dim=4096, num_heads=32)
    assert result is not None
    assert result["savings_pct"] > 0, "Fusion should yield some savings"
    assert len(result["fusion_groups"]) > 0
    assert len(result["ops"]) == 14

    print(f"[PASS] test_transformer_block")
    print(f"       Unfused traffic: {result['unfused_traffic']['total'] / 1e9:.2f} GB")
    print(f"       Fused traffic:   {result['fused_traffic']['total'] / 1e9:.2f} GB")
    print(f"       Savings:         {result['savings_pct']:.1f}%")
    print(f"       Fusion groups:   {len(result['fusion_groups'])}")
    for i, g in enumerate(result['fusion_groups']):
        print(f"         Group {i}: {g}")


if __name__ == "__main__":
    test_basic_fusion()
    test_all_elementwise()
    test_diamond_dependency()
    test_reduction_boundary()
    test_single_op()
    test_memory_traffic_values()
    test_transformer_block()
    print("\nAll tests passed!")
