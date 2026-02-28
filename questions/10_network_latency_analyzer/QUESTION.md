# Question 10: Network Latency Analyzer

## Difficulty: Hard
## Time: 60 minutes
## Category: Performance Debugging / Distributed Systems

---

## Background

You are debugging a distributed system where latency has spiked. You have collected
**trace data** from the system: each trace represents an end-to-end request and contains
a list of **spans**. Each span represents one hop or operation in the request path.

This is the same conceptual model used by distributed tracing systems like Jaeger,
Zipkin, and OpenTelemetry. Understanding how to programmatically analyze trace data
is a core performance engineering skill.

### Span Structure

A span has the following fields:

| Field             | Type            | Description                          |
|-------------------|-----------------|--------------------------------------|
| `span_id`         | `str`           | Unique identifier for this span      |
| `parent_span_id`  | `str \| None`   | Parent span ID (`None` for the root) |
| `service`         | `str`           | Which service handled this span      |
| `operation`       | `str`           | What operation was performed         |
| `start_time_ms`   | `float`         | Start time in milliseconds           |
| `end_time_ms`     | `float`         | End time in milliseconds             |

### Example Trace

```
api-gateway/route          [0ms ==============================================> 55ms]
  auth/validate_token        [5ms =======> 15ms]
  model-service/inference      [15ms =========================> 40ms]
    gpu-worker/forward_pass       [17ms ======================> 37ms]
```

---

## Tasks

Implement the following four analysis functions.

### 1. `build_trace_tree(spans: list[Span]) -> SpanNode`

Build a tree from a flat list of spans. Each `SpanNode` wraps a `Span` and has a list
of children.

**Example:**
```python
spans = [
    Span("s0", None,  "api-gateway",    "route",          0.0, 55.0),
    Span("s1", "s0",  "auth",           "validate_token", 5.0, 15.0),
    Span("s2", "s0",  "model-service",  "inference",     15.0, 40.0),
    Span("s3", "s2",  "gpu-worker",     "forward_pass",  17.0, 37.0),
]

root = build_trace_tree(spans)
# root.span.service == "api-gateway"
# len(root.children) == 2  (auth, model-service)
# root.children[1].children[0].span.service == "gpu-worker"
```

### 2. `find_critical_path(root: SpanNode) -> list[Span]`

Find the **critical path** -- the chain of spans from the root to the leaf whose
`end_time_ms` is the latest. This is the path that determines total request latency.

The critical path is NOT simply the longest chain by span count; it is the path
that ends at the latest absolute time.

**Example:**
```python
root = build_trace_tree(spans)  # Using spans from above
path = find_critical_path(root)
# path == [
#     Span("s0", ..., "api-gateway",   "route",         0.0, 55.0),
#     Span("s2", ..., "model-service", "inference",     15.0, 40.0),
#     Span("s3", ..., "gpu-worker",    "forward_pass",  17.0, 37.0),
# ]
# The critical path goes through model-service, not auth,
# because the model-service subtree ends later.
```

### 3. `find_bottlenecks(traces, threshold_pct=0.5) -> list[dict]`

Analyze multiple traces. For each `(service, operation)` pair, compute its **self-time**:
the span's total duration minus the time covered by its child spans.

Return operations where the average self-time accounts for more than `threshold_pct`
(as a fraction, e.g., 0.5 = 50%) of the total trace duration, on average across traces.

**Example:**
```python
traces = generate_sample_traces(100)
bottlenecks = find_bottlenecks(traces, threshold_pct=0.3)
# Returns something like:
# [
#     {
#         "service": "gpu-worker",
#         "operation": "forward_pass",
#         "avg_self_time_ms": 20.1,
#         "avg_pct_of_total": 0.37
#     },
#     ...
# ]
```

**Self-time computation:** If a parent span runs from 0-100ms and has two children
spanning 10-30ms and 50-70ms, the parent's self-time is 100 - 20 - 20 = 60ms.
Be careful with overlapping children -- do not double-count overlapping regions.

### 4. `detect_anomalies(traces, std_dev_threshold=2.0) -> list[dict]`

For each `(service, operation)` pair, compute the mean and standard deviation of
span durations across all traces. Flag individual spans whose duration is more than
`std_dev_threshold` standard deviations above the mean.

**Example:**
```python
traces = generate_sample_traces(100)
anomalies = detect_anomalies(traces, std_dev_threshold=2.0)
# Returns something like:
# [
#     {
#         "span_id": "t42_s3",
#         "service": "gpu-worker",
#         "operation": "forward_pass",
#         "duration_ms": 102.5,
#         "mean_ms": 20.3,
#         "std_dev_ms": 4.1,
#         "z_score": 20.05
#     },
#     ...
# ]
```

---

## Starter Code

See `starter.py` for dataclass definitions, sample data generator, and function
signatures.

---

## Evaluation Criteria

- Correctness of tree construction from flat span list
- Critical path identification based on time, not span count
- Accurate self-time computation, handling overlapping child spans
- Correct statistical computation for anomaly detection
- Edge case handling: single-span traces, missing parents, overlapping spans
- Code clarity and efficiency
