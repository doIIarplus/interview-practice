"""
Question 10: Network Latency Analyzer
======================================

Analyze distributed system trace data to find critical paths, bottlenecks,
and anomalies. This simulates the kind of analysis performed by distributed
tracing systems like Jaeger, Zipkin, and OpenTelemetry.

Implement the four analysis functions below.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Span:
    """A single span in a distributed trace.

    Represents one hop or operation in a request's path through the system.
    """

    span_id: str
    parent_span_id: str | None
    service: str
    operation: str
    start_time_ms: float
    end_time_ms: float

    @property
    def duration_ms(self) -> float:
        """Duration of this span in milliseconds."""
        return self.end_time_ms - self.start_time_ms


@dataclass
class SpanNode:
    """A node in the trace tree, wrapping a Span with child references."""

    span: Span
    children: list[SpanNode] = field(default_factory=list)


def generate_sample_traces(n_traces: int = 100) -> list[list[Span]]:
    """Generate sample trace data for testing.

    Produces traces that simulate a typical ML inference pipeline:
        api-gateway/route
            -> auth/validate_token
            -> model-service/inference
                -> gpu-worker/forward_pass

    Approximately 5% of traces will contain an anomalously slow gpu-worker span.

    Args:
        n_traces: Number of traces to generate.

    Returns:
        A list of traces, where each trace is a list of Span objects.
    """
    import random

    traces = []
    for t in range(n_traces):
        spans = []
        root_start = 0.0

        # API Gateway — total request duration
        gateway_end = random.gauss(5, 1)
        spans.append(
            Span(
                f"t{t}_s0",
                None,
                "api-gateway",
                "route",
                root_start,
                root_start + gateway_end + random.gauss(50, 10),
            )
        )

        # Auth service — child of gateway
        auth_start = root_start + gateway_end
        auth_dur = random.gauss(10, 2)
        spans.append(
            Span(
                f"t{t}_s1",
                f"t{t}_s0",
                "auth",
                "validate_token",
                auth_start,
                auth_start + auth_dur,
            )
        )

        # Model service — child of gateway, starts after auth
        model_start = auth_start + auth_dur
        model_dur = random.gauss(25, 5)
        spans.append(
            Span(
                f"t{t}_s2",
                f"t{t}_s0",
                "model-service",
                "inference",
                model_start,
                model_start + model_dur,
            )
        )

        # GPU worker — child of model service
        gpu_start = model_start + random.gauss(2, 0.5)
        gpu_dur = random.gauss(20, 4)
        spans.append(
            Span(
                f"t{t}_s3",
                f"t{t}_s2",
                "gpu-worker",
                "forward_pass",
                gpu_start,
                gpu_start + gpu_dur,
            )
        )

        # Add occasional anomaly (5% of traces)
        if random.random() < 0.05:
            anomalous = spans[-1]
            spans[-1] = Span(
                anomalous.span_id,
                anomalous.parent_span_id,
                anomalous.service,
                anomalous.operation,
                gpu_start,
                gpu_start + gpu_dur * 5,
            )

        traces.append(spans)

    return traces


# ---------------------------------------------------------------------------
# Implement the following functions
# ---------------------------------------------------------------------------


def build_trace_tree(spans: list[Span]) -> SpanNode:
    """Build a tree from a flat list of spans.

    Each SpanNode wraps a Span and contains a list of its child SpanNodes.
    The root span is the one whose parent_span_id is None.

    Args:
        spans: A flat list of Span objects belonging to a single trace.

    Returns:
        The root SpanNode of the trace tree.

    Raises:
        ValueError: If no root span is found.
    """
    pass


def find_critical_path(root: SpanNode) -> list[Span]:
    """Find the critical path in the trace tree.

    The critical path is the chain of spans from the root to the leaf
    whose end_time_ms is the latest. This is the path that determines
    the total request latency.

    Note: This is the longest path by TIME, not by number of spans.

    Args:
        root: The root SpanNode of a trace tree.

    Returns:
        A list of Span objects representing the critical path,
        ordered from root to leaf.
    """
    pass


def find_bottlenecks(
    traces: list[list[Span]], threshold_pct: float = 0.5
) -> list[dict]:
    """Find operations that are bottlenecks across multiple traces.

    A bottleneck is an operation whose self-time (duration minus time
    spent in children) accounts for more than threshold_pct of the
    total trace duration, on average.

    Self-time must handle overlapping child spans correctly — do not
    double-count overlapping time regions.

    Args:
        traces: A list of traces (each trace is a list of Spans).
        threshold_pct: Minimum fraction of total trace time for an
                       operation to be considered a bottleneck.

    Returns:
        A list of dicts with keys:
            - "service": str
            - "operation": str
            - "avg_self_time_ms": float
            - "avg_pct_of_total": float
        Sorted by avg_pct_of_total descending.
    """
    pass


def detect_anomalies(
    traces: list[list[Span]], std_dev_threshold: float = 2.0
) -> list[dict]:
    """Detect anomalous spans whose duration is unusually high.

    For each (service, operation) pair, compute the mean and standard
    deviation of durations across all traces. Flag spans whose duration
    is more than std_dev_threshold standard deviations above the mean.

    Args:
        traces: A list of traces (each trace is a list of Spans).
        std_dev_threshold: Number of standard deviations above the mean
                           for a span to be considered anomalous.

    Returns:
        A list of dicts with keys:
            - "span_id": str
            - "service": str
            - "operation": str
            - "duration_ms": float
            - "mean_ms": float
            - "std_dev_ms": float
            - "z_score": float
        Sorted by z_score descending.
    """
    pass


# ---------------------------------------------------------------------------
# Quick smoke tests
# ---------------------------------------------------------------------------


def _smoke_test():
    """Run basic smoke tests to verify implementations."""
    # Test build_trace_tree
    spans = [
        Span("s0", None, "api-gateway", "route", 0.0, 55.0),
        Span("s1", "s0", "auth", "validate_token", 5.0, 15.0),
        Span("s2", "s0", "model-service", "inference", 15.0, 40.0),
        Span("s3", "s2", "gpu-worker", "forward_pass", 17.0, 37.0),
    ]

    root = build_trace_tree(spans)
    assert root.span.service == "api-gateway"
    assert len(root.children) == 2
    print("[PASS] build_trace_tree")

    # Test find_critical_path
    path = find_critical_path(root)
    services_on_path = [s.service for s in path]
    assert services_on_path[0] == "api-gateway"
    assert "model-service" in services_on_path
    assert "gpu-worker" in services_on_path
    print("[PASS] find_critical_path")

    # Test find_bottlenecks
    traces = generate_sample_traces(200)
    bottlenecks = find_bottlenecks(traces, threshold_pct=0.1)
    assert len(bottlenecks) > 0
    assert all(b["avg_pct_of_total"] > 0.1 for b in bottlenecks)
    print("[PASS] find_bottlenecks")

    # Test detect_anomalies
    anomalies = detect_anomalies(traces, std_dev_threshold=2.0)
    if anomalies:
        assert all(a["z_score"] > 2.0 for a in anomalies)
        print(f"[PASS] detect_anomalies — found {len(anomalies)} anomalies")
    else:
        print("[PASS] detect_anomalies — no anomalies (possible with random data)")

    print("\nAll smoke tests passed!")


if __name__ == "__main__":
    _smoke_test()
