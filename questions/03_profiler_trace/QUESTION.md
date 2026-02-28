# Question 03: Profiler Trace Conversion

## Overview

A sampling profiler periodically captures the current call stack of a running program. Your task is to convert a sequence of stack samples into a list of trace events that describe when functions started and stopped executing.

---

## Background

A **sampling profiler** works by periodically interrupting a running program and recording the current call stack. Each sample is a snapshot of which functions are currently on the stack, from the bottom (e.g., `main`) to the top (the currently executing function).

For example, if `main` calls `foo`, which calls `bar`, a sample might look like:

```
["main", "foo", "bar"]
```

By comparing consecutive samples, we can infer when functions started and stopped executing.

---

## Task

Implement the following function:

```python
def samples_to_trace(samples: list[list[str]]) -> list[tuple[str, str, int]]:
    """Convert stack samples into trace events.

    Args:
        samples: A list of stack samples. Each sample is a list of function
                 names from bottom of the stack (e.g., main) to top
                 (currently executing function).

    Returns:
        A list of trace events. Each event is a tuple of:
        (event_type, function_name, sample_index)
        where event_type is "start" or "end".
    """
```

### Trace Event Rules

1. **Start event:** If a function appears at position `i` in sample `n` but was not at position `i` in sample `n-1` (or sample `n` is the first sample), emit `("start", function_name, n)`.

2. **End event:** If a function appears at position `i` in sample `n` but is not at position `i` in sample `n+1` (or sample `n` is the last sample), emit `("end", function_name, n+1)`. (The end happens at the *next* sample index.)

3. **Event ordering within a sample index:**
   - **End events** for deeper (inner) functions come **before** end events for shallower (outer) functions. (Inner functions end first -- stack unwinding order.)
   - **Start events** for shallower (outer) functions come **before** start events for deeper (inner) functions. (Outer functions start first -- stack building order.)
   - All **end events** at a given sample index come **before** all **start events** at that sample index.

4. **Identical consecutive samples:** If sample `n` and sample `n+1` are identical, no events are generated between them.

5. **First sample:** All functions in the first sample generate start events at index 0.

6. **Last sample:** All functions in the last sample generate end events at index `len(samples)`.

7. **Recursive calls:** The same function name can appear multiple times in a single stack. Each occurrence is tracked by its **position** in the stack, not just by name.

---

## Examples

### Example 1: Basic transitions

```python
samples = [
    ["main", "foo", "bar"],
    ["main", "foo"],
    ["main", "baz"],
]

result = samples_to_trace(samples)
```

Expected output:
```python
[
    ("start", "main", 0),
    ("start", "foo", 0),
    ("start", "bar", 0),
    ("end", "bar", 1),       # bar stopped at sample 1
    ("end", "foo", 1),       # foo stopped at sample 1 (but wait, foo is still in sample 1)
    # Actually, let's reconsider...
]
```

Let me trace through carefully:

- **Sample 0 vs nothing (first sample):** `main`, `foo`, `bar` all start.
  - Events: `("start", "main", 0)`, `("start", "foo", 0)`, `("start", "bar", 0)`

- **Sample 0 -> Sample 1:** `["main", "foo", "bar"]` -> `["main", "foo"]`
  - `bar` was at position 2 in sample 0, not present at position 2 in sample 1. End event.
  - `main` and `foo` remain at the same positions. No events.
  - Events: `("end", "bar", 1)`

- **Sample 1 -> Sample 2:** `["main", "foo"]` -> `["main", "baz"]`
  - Position 1: was `foo`, now `baz`. `foo` ends, `baz` starts.
  - Events: `("end", "foo", 2)`, `("start", "baz", 2)`

- **Sample 2 is last:** `main` and `baz` end.
  - Events: `("end", "baz", 3)`, `("end", "main", 3)` (inner ends before outer)

Full result:
```python
[
    ("start", "main", 0),
    ("start", "foo", 0),
    ("start", "bar", 0),
    ("end", "bar", 1),
    ("end", "foo", 2),
    ("start", "baz", 2),
    ("end", "baz", 3),
    ("end", "main", 3),
]
```

### Example 2: Identical consecutive samples

```python
samples = [
    ["main", "foo"],
    ["main", "foo"],
    ["main", "bar"],
]

result = samples_to_trace(samples)
```

Expected output:
```python
[
    ("start", "main", 0),
    ("start", "foo", 0),
    # No events between sample 0 and 1 (identical)
    ("end", "foo", 2),
    ("start", "bar", 2),
    ("end", "bar", 3),
    ("end", "main", 3),
]
```

### Example 3: Recursion

```python
samples = [
    ["main", "factorial", "factorial", "factorial"],
    ["main", "factorial", "factorial"],
    ["main", "factorial"],
    ["main"],
]

result = samples_to_trace(samples)
```

Expected output:
```python
[
    ("start", "main", 0),
    ("start", "factorial", 0),       # position 1
    ("start", "factorial", 0),       # position 2
    ("start", "factorial", 0),       # position 3
    ("end", "factorial", 1),         # position 3 ends
    ("end", "factorial", 2),         # position 2 ends
    ("end", "factorial", 3),         # position 1 ends
    ("end", "main", 4),
]
```

### Example 4: Empty and single samples

```python
# Empty input
samples_to_trace([])
# => []

# Single sample
samples_to_trace([["main"]])
# => [("start", "main", 0), ("end", "main", 1)]

# Sample with empty stack
samples_to_trace([[]])
# => []
```

---

## Constraints

- `0 <= len(samples) <= 10000`
- Each sample is a list of 0 or more function name strings.
- Function names are non-empty strings.
- The same function name can appear multiple times in a single sample (recursion).
