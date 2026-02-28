# Rubric: Profiler Trace Conversion

> **This file is hidden from the candidate.**

## Scoring Breakdown

### Correct Basic Conversion (25%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Start events for first sample | 8 | All functions in the first sample generate start events at index 0 |
| End events for last sample | 8 | All functions in the last sample generate end events at `len(samples)` |
| Transition detection | 9 | Correctly detects when functions appear/disappear between consecutive samples |

**Green flags:**
- Compares samples position-by-position
- Uses a clear loop structure: iterate through pairs of consecutive samples
- Handles the "first sample" and "last sample" as special cases or elegantly as transitions from/to an empty list

**Red flags:**
- Only compares by function name presence (ignoring position) -- breaks for recursion
- Off-by-one errors on sample indices for end events

### Proper Event Ordering (20%)

| Criteria | Points | Description |
|----------|--------|-------------|
| End events: inner before outer | 8 | Deeper stack positions end first (reverse order) |
| Start events: outer before inner | 8 | Shallower stack positions start first (natural order) |
| Ends before starts at same index | 4 | All end events at a sample index precede all start events |

**Green flags:**
- Collects end events in reverse order (from deepest to shallowest)
- Collects start events in natural order (from shallowest to deepest)
- Appends ends then starts to the result list

**Red flags:**
- Events are in arbitrary order
- Doesn't consider that end and start events can occur at the same sample index

### Handling Identical Consecutive Samples (15%)

| Criteria | Points | Description |
|----------|--------|-------------|
| No spurious events | 10 | Identical samples produce no events between them |
| Correct behavior after identical run | 5 | Events resume correctly when samples change after a run of identical ones |

**Green flags:**
- Position-by-position comparison naturally handles this (if positions match, no events)
- Explicitly short-circuits with an equality check (optimization, not required)

### Handling Recursive Calls (20%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Position-based tracking | 10 | Tracks functions by stack position, not just name |
| Multiple same-name functions | 5 | Correctly handles `["main", "f", "f", "f"]` |
| Partial recursion unwind | 5 | Correctly handles going from depth 3 to depth 1 of same function |

**Green flags:**
- Uses `zip` or index-based iteration to compare position-by-position
- Handles differing stack lengths by treating missing positions as absent
- Each position is independent -- a function at position 2 is separate from the same function at position 3

**Red flags:**
- Uses sets of function names to track what changed (loses position information)
- Counts occurrences of each function name (fragile, doesn't handle all cases)

### Code Clarity and Edge Cases (20%)

| Criteria | Points | Description |
|----------|--------|-------------|
| Empty input | 4 | Returns `[]` for empty samples list |
| Single sample | 4 | Correctly produces start and end events |
| Empty stacks in samples | 4 | Handles `[[]]` or `[[], ["main"]]` |
| Code readability | 4 | Clear variable names, comments where needed |
| Efficiency | 4 | O(n * m) where n is number of samples and m is max stack depth |

---

## Algorithm Sketch

The key insight is to compare consecutive samples position-by-position:

```python
def samples_to_trace(samples):
    if not samples:
        return []

    events = []
    prev = []

    for i, curr in enumerate(samples):
        ends = []
        starts = []

        # Find the first position where samples differ
        # (or where one is shorter than the other)
        min_len = min(len(prev), len(curr))
        diverge = 0
        while diverge < min_len and prev[diverge] == curr[diverge]:
            diverge += 1

        # Everything in prev from diverge onward has ended
        # (reverse order: inner ends before outer)
        for j in range(len(prev) - 1, diverge - 1, -1):
            ends.append(("end", prev[j], i))

        # Everything in curr from diverge onward has started
        # (natural order: outer starts before inner)
        for j in range(diverge, len(curr)):
            starts.append(("start", curr[j], i))

        events.extend(ends)
        events.extend(starts)
        prev = curr

    # End events for the last sample
    for j in range(len(prev) - 1, -1, -1):
        events.append(("end", prev[j], len(samples)))

    return events
```

**Key insight:** Once two stacks diverge at position `k`, *everything* from position `k` onward must be ended (in the old stack) and started (in the new stack), even if some function names happen to match. This is because a divergence at position `k` means the call was different, so deeper calls are necessarily different calls even if they share names.

---

## Overall Assessment

| Rating | Description |
|--------|-------------|
| **Strong hire** | Clean position-based comparison, correct ordering, handles recursion and all edge cases |
| **Hire** | Working solution for basic cases, minor issues with ordering or recursion |
| **Borderline** | Gets basic cases right but fails on recursion or event ordering |
| **No hire** | Cannot produce correct events for the basic example |

## Time Expectations

- Understanding the problem: 5 minutes
- Basic implementation: 15-20 minutes
- Edge cases and recursion: 5-10 minutes
- Total: 25-35 minutes
