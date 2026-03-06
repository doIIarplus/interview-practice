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
# Usage Example
# =============================================================================
if __name__ == "__main__":
    samples = [
        ["main", "foo", "bar"],
        ["main", "foo"],
        ["main", "baz"],
    ]
    result = samples_to_trace(samples)
    for event in result or []:
        print(event)
