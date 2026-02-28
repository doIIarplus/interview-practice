# Rubric: Network Latency Analyzer

**Total: 100 points**

---

## 1. Correct Tree Building from Flat Span List (15 points)

### Full marks (15):
- Correctly identifies the root span (parent_span_id is None)
- Builds parent-child relationships using a dictionary lookup (O(n) construction)
- Handles spans provided in any order (not just parent-before-child)
- Raises an appropriate error if no root span exists

### Partial credit (8-12):
- Works but assumes spans are in parent-before-child order
- Works but uses O(n^2) nested loops to find parents

### Minimal credit (1-7):
- Only works for specific orderings
- Does not handle orphan spans gracefully

### Key implementation detail:
The optimal approach builds a dict of span_id -> SpanNode first, then wires up
parent-child relationships in a second pass:
```python
nodes = {s.span_id: SpanNode(s) for s in spans}
root = None
for s in spans:
    if s.parent_span_id is None:
        root = nodes[s.span_id]
    else:
        nodes[s.parent_span_id].children.append(nodes[s.span_id])
return root
```

---

## 2. Critical Path Identification (20 points)

### Full marks (20):
- Correctly finds the path from root to the leaf with the latest end_time_ms
- Returns spans ordered from root to leaf
- Uses recursive DFS or BFS traversal
- Handles single-span traces (root with no children)

### Partial credit (10-15):
- Finds the longest path by span count instead of by time
- Finds the correct leaf but returns spans in wrong order

### Minimal credit (1-9):
- Returns all leaves instead of the critical one
- Does not trace back to root

### Key distinction:
The critical path is determined by **latest end time**, not longest duration or
deepest nesting. The candidate must traverse to find the leaf with max end_time_ms,
then return the path from root to that leaf.

### Good approach:
```python
def find_critical_path(root):
    if not root.children:
        return [root.span]
    # Recurse into each child, find the one with latest end time
    best_path = None
    for child in root.children:
        child_path = find_critical_path(child)
        if best_path is None or child_path[-1].end_time_ms > best_path[-1].end_time_ms:
            best_path = child_path
    return [root.span] + best_path
```

---

## 3. Self-Time Computation and Bottleneck Detection (20 points)

### Full marks (20):
- Correctly computes self-time = span duration - union of child span durations
- Handles overlapping child spans (merges intervals before subtracting)
- Aggregates across multiple traces correctly (averages)
- Returns results sorted by avg_pct_of_total descending

### Partial credit (10-15):
- Computes self-time but doesn't handle overlapping children
- Correct logic but doesn't normalize by total trace duration properly

### Minimal credit (1-9):
- Simply uses span duration without subtracting children
- Off-by-one or incorrect interval merging

### Critical detail — overlapping child intervals:
If a parent has children spanning [10, 30] and [25, 50], the total child time
is 40ms (not 45ms). The candidate MUST merge overlapping intervals:
```python
def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
```

### Self-time for the total trace:
Total trace time = root span's (end_time - start_time). Each span's self-time
fraction = self_time / total_trace_time. This is aggregated per (service, operation)
across traces.

---

## 4. Anomaly Detection with Statistical Computation (15 points)

### Full marks (15):
- Groups spans by (service, operation)
- Correctly computes mean and standard deviation of durations
- Identifies spans where (duration - mean) / std_dev > threshold
- Returns z-scores and relevant metadata
- Handles edge case where std_dev is 0 (all same duration)

### Partial credit (8-12):
- Correct grouping and z-score computation
- Doesn't handle std_dev = 0

### Minimal credit (1-7):
- Incorrect mean/std_dev computation
- Uses population vs sample std dev inconsistently (either is acceptable but should be intentional)

### Key formula:
```python
z_score = (duration - mean) / std_dev
anomalous if z_score > std_dev_threshold
```

---

## 5. Edge Case Handling (15 points)

### Full marks (15):
- Single-span traces (root with no children): critical path = [root], self-time = duration
- Missing parent spans: handles gracefully (orphan spans logged or attached to a synthetic root)
- Overlapping children: interval merging in self-time calculation
- Empty trace list: returns empty results
- Zero-duration spans: handled without division errors
- Negative durations: handled or flagged

### Partial credit (8-12):
- Handles most cases but crashes on edge cases
- Handles single-span but not missing parents

### Minimal credit (1-7):
- Only works on the happy path

---

## 6. Code Clarity and Efficiency (15 points)

### Full marks (15):
- Clear variable names and function structure
- O(n) tree building (dictionary-based)
- O(n) critical path (single DFS pass)
- O(n * m) bottleneck detection (n traces, m spans per trace)
- Appropriate use of collections (defaultdict, etc.)
- Good separation of concerns

### Partial credit (8-12):
- Readable but some inefficiencies (e.g., O(n^2) tree building)
- Could be better factored

### Minimal credit (1-7):
- Hard to follow, deeply nested, or overly complex
- Clearly inefficient algorithms

---

## Red Flags (Automatic Deductions)

- **-10 points**: Critical path found by span count instead of time
- **-10 points**: Self-time doesn't subtract child time at all
- **-5 points**: No handling of overlapping child intervals
- **-5 points**: Division by zero possible with std_dev = 0
- **-5 points**: Modifying input data structures

---

## Exceptional Answers (Bonus Discussion Points)

- Mentions that clock skew between machines can make start/end times unreliable
- Discusses how to handle partial/incomplete traces
- Mentions that real systems use sampling to handle trace volume
- Suggests using percentile-based anomaly detection instead of z-score
  (z-score assumes normal distribution, latency is often log-normal)
- Proposes streaming/online algorithms for computing statistics
