"""
Profiler Trace Conversion

Convert a sequence of stack samples from a sampling profiler into
a list of trace events (start/end) for each function.
See QUESTION.md for full problem description.
"""


def samples_to_trace(
    samples: list[list[str]],
) -> list[tuple[str, str, int]]:
    """Convert stack samples into trace events.

    A sampling profiler periodically captures the current call stack. Given
    a list of these stack samples, produce a list of trace events that
    describe when each function started and stopped executing.

    Each sample is a list of function names from the bottom of the stack
    (e.g., "main") to the top (the currently executing function).

    Each trace event is a tuple of (event_type, function_name, sample_index):
      - ("start", function_name, index): function began executing at this sample
      - ("end", function_name, index): function stopped executing at this sample

    Rules:
      - Compare consecutive samples position-by-position to detect changes.
      - End events for inner (deeper) functions come before outer functions.
      - Start events for outer (shallower) functions come before inner functions.
      - All end events at a sample index come before all start events.
      - Handle recursive calls by tracking position, not just function name.

    Args:
        samples: A list of stack samples. Each sample is a list of function
                 names from bottom of stack to top.

    Returns:
        A list of trace events as (event_type, function_name, sample_index) tuples.
    """
    pass


# =============================================================================
# Quick Smoke Tests
# =============================================================================
if __name__ == "__main__":
    # Example 1: Basic transitions
    print("--- Example 1: Basic Transitions ---")
    samples1 = [
        ["main", "foo", "bar"],
        ["main", "foo"],
        ["main", "baz"],
    ]
    result1 = samples_to_trace(samples1)
    expected1 = [
        ("start", "main", 0),
        ("start", "foo", 0),
        ("start", "bar", 0),
        ("end", "bar", 1),
        ("end", "foo", 2),
        ("start", "baz", 2),
        ("end", "baz", 3),
        ("end", "main", 3),
    ]
    for event in result1 or []:
        print(event)
    print(f"Match: {result1 == expected1}\n")

    # Example 2: Identical consecutive samples
    print("--- Example 2: Identical Consecutive ---")
    samples2 = [
        ["main", "foo"],
        ["main", "foo"],
        ["main", "bar"],
    ]
    result2 = samples_to_trace(samples2)
    expected2 = [
        ("start", "main", 0),
        ("start", "foo", 0),
        ("end", "foo", 2),
        ("start", "bar", 2),
        ("end", "bar", 3),
        ("end", "main", 3),
    ]
    for event in result2 or []:
        print(event)
    print(f"Match: {result2 == expected2}\n")

    # Example 3: Recursion
    print("--- Example 3: Recursion ---")
    samples3 = [
        ["main", "factorial", "factorial", "factorial"],
        ["main", "factorial", "factorial"],
        ["main", "factorial"],
        ["main"],
    ]
    result3 = samples_to_trace(samples3)
    expected3 = [
        ("start", "main", 0),
        ("start", "factorial", 0),
        ("start", "factorial", 0),
        ("start", "factorial", 0),
        ("end", "factorial", 1),
        ("end", "factorial", 2),
        ("end", "factorial", 3),
        ("end", "main", 4),
    ]
    for event in result3 or []:
        print(event)
    print(f"Match: {result3 == expected3}\n")

    # Example 4: Edge cases
    print("--- Example 4: Edge Cases ---")
    print(f"Empty:  {samples_to_trace([])}")                     # => []
    print(f"Single: {samples_to_trace([['main']])}")             # => [("start","main",0), ("end","main",1)]
    print(f"EmptyStack: {samples_to_trace([[]])}")               # => []
