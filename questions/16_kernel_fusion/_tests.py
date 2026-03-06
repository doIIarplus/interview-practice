"""Hidden tests for Question 16: Kernel Fusion Simulator
Run: python questions/16_kernel_fusion/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import Op, build_graph, find_fusion_groups, estimate_memory_traffic, analyze_transformer_block


def test_basic_fusion():
    """Test: matmul + elementwise epilogue fuses; second matmul is separate."""
    ops = [
        Op("matmul1", "matmul",      ["x", "w1"], "y1", (1024, 4096), 2),
        Op("bias1",   "elementwise", ["y1", "b1"], "y2", (1024, 4096), 2),
        Op("relu",    "elementwise", ["y2"],        "y3", (1024, 4096), 2),
        Op("matmul2", "matmul",      ["y3", "w2"], "y4", (1024, 1024), 2),
    ]
    graph = build_graph(ops)
    assert graph is not None, "build_graph returned None"
    assert graph["tensor_producer"]["y1"].name == "matmul1"
    assert len(graph["tensor_consumers"]["y1"]) == 1

    groups = find_fusion_groups(ops)
    assert len(groups) == 2, f"Expected 2 fusion groups, got {len(groups)}"
    assert [op.name for op in groups[0]] == ["matmul1", "bias1", "relu"]
    assert [op.name for op in groups[1]] == ["matmul2"]

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
    """Diamond: one op's output feeds two consumers -> cannot eliminate intermediate."""
    ops = [
        Op("matmul1", "matmul",      ["x", "w"], "y1", (1024, 1024), 2),
        Op("relu",    "elementwise", ["y1"],      "y2", (1024, 1024), 2),
        Op("gelu",    "elementwise", ["y1"],      "y3", (1024, 1024), 2),
    ]
    groups = find_fusion_groups(ops)
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
    """Verify memory traffic numbers for a simple case."""
    ops = [
        Op("matmul1", "matmul",      ["x", "w1"], "y1", (1024, 4096), 2),
        Op("relu",    "elementwise", ["y1"],        "y2", (1024, 4096), 2),
    ]
    unfused = estimate_memory_traffic(ops, fused=False)
    fused = estimate_memory_traffic(ops, fused=True)
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


def run_tests():
    print("Running Kernel Fusion Simulator tests...\n")
    test_basic_fusion()
    test_all_elementwise()
    test_diamond_dependency()
    test_reduction_boundary()
    test_single_op()
    test_memory_traffic_values()
    test_transformer_block()
    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
