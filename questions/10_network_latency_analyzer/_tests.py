"""Hidden tests for Question 10: Network Latency Analyzer
Run: python questions/10_network_latency_analyzer/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import (
    Span, SpanNode, generate_sample_traces,
    build_trace_tree, find_critical_path, find_bottlenecks, detect_anomalies,
)


def test_build_trace_tree():
    """Test building a trace tree from flat spans."""
    spans = [
        Span("s0", None, "api-gateway", "route", 0.0, 55.0),
        Span("s1", "s0", "auth", "validate_token", 5.0, 15.0),
        Span("s2", "s0", "model-service", "inference", 15.0, 40.0),
        Span("s3", "s2", "gpu-worker", "forward_pass", 17.0, 37.0),
    ]

    root = build_trace_tree(spans)
    assert root.span.service == "api-gateway"
    assert len(root.children) == 2
    print("[PASS] test_build_trace_tree")


def test_find_critical_path():
    """Test finding the critical path in a trace tree."""
    spans = [
        Span("s0", None, "api-gateway", "route", 0.0, 55.0),
        Span("s1", "s0", "auth", "validate_token", 5.0, 15.0),
        Span("s2", "s0", "model-service", "inference", 15.0, 40.0),
        Span("s3", "s2", "gpu-worker", "forward_pass", 17.0, 37.0),
    ]

    root = build_trace_tree(spans)
    path = find_critical_path(root)
    services_on_path = [s.service for s in path]
    assert services_on_path[0] == "api-gateway"
    assert "model-service" in services_on_path
    assert "gpu-worker" in services_on_path
    print("[PASS] test_find_critical_path")


def test_find_bottlenecks():
    """Test bottleneck detection across traces."""
    traces = generate_sample_traces(200)
    bottlenecks = find_bottlenecks(traces, threshold_pct=0.1)
    assert len(bottlenecks) > 0
    assert all(b["avg_pct_of_total"] > 0.1 for b in bottlenecks)
    print("[PASS] test_find_bottlenecks")


def test_detect_anomalies():
    """Test anomaly detection in traces."""
    traces = generate_sample_traces(200)
    anomalies = detect_anomalies(traces, std_dev_threshold=2.0)
    if anomalies:
        assert all(a["z_score"] > 2.0 for a in anomalies)
        print(f"[PASS] test_detect_anomalies — found {len(anomalies)} anomalies")
    else:
        print("[PASS] test_detect_anomalies — no anomalies (possible with random data)")


def run_tests():
    print("=" * 60)
    print("Network Latency Analyzer — Hidden Tests")
    print("=" * 60 + "\n")

    test_build_trace_tree()
    test_find_critical_path()
    test_find_bottlenecks()
    test_detect_anomalies()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
