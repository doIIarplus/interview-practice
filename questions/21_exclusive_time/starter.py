"""
Question 21: Exclusive Time of Functions (LeetCode 636)

Given n functions and a list of call logs, compute the exclusive time of
each function. Exclusive time is the time a function spent executing,
NOT including time spent in functions it called.

Key detail: "end" timestamps are inclusive. A function starting at 0 and
ending at 0 has an exclusive time of 1 unit.
"""


def exclusive_time(n: int, logs: list[str]) -> list[int]:
    """
    Compute the exclusive time of each function.

    Args:
        n: Number of functions (IDs from 0 to n-1).
        logs: List of log strings in format "{func_id}:{start|end}:{timestamp}".
              Logs are in chronological order.

    Returns:
        List of length n where result[i] is the exclusive time of function i.

    Approach hint:
        Use a stack to track the currently executing function.
        - On "start": credit the current top-of-stack function for elapsed time,
          then push the new function.
        - On "end": credit the ending function for elapsed time (+1 because
          end is inclusive), pop it, and set prev_timestamp = timestamp + 1.
    """
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_nesting():
    """Example 1: Function 0 calls function 1."""
    print("TEST: Basic nesting (function 0 calls function 1)")

    n = 2
    logs = ["0:start:0", "1:start:2", "1:end:5", "0:end:6"]
    expected = [3, 4]

    result = exclusive_time(n, logs)
    if result is None:
        print("  exclusive_time() returned None -- not yet implemented\n")
        return

    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def test_recursive():
    """Example 2: Function 0 calls itself (recursion)."""
    print("TEST: Recursive call (function 0 calls function 0)")

    n = 1
    logs = ["0:start:0", "0:start:2", "0:end:5", "0:end:6"]
    expected = [7]

    result = exclusive_time(n, logs)
    if result is None:
        print("  exclusive_time() returned None -- not yet implemented\n")
        return

    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def test_sequential():
    """Example 3: Sequential calls, no nesting."""
    print("TEST: Sequential calls (no nesting)")

    n = 2
    logs = ["0:start:0", "0:end:0", "1:start:1", "1:end:1", "0:start:2", "0:end:2"]
    expected = [2, 1]

    result = exclusive_time(n, logs)
    if result is None:
        print("  exclusive_time() returned None -- not yet implemented\n")
        return

    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def test_deeply_nested():
    """Example 4: Three levels of nesting."""
    print("TEST: Deeply nested (3 levels)")

    n = 3
    logs = ["0:start:0", "1:start:1", "2:start:2", "2:end:3", "1:end:4", "0:end:5"]
    expected = [2, 2, 2]

    result = exclusive_time(n, logs)
    if result is None:
        print("  exclusive_time() returned None -- not yet implemented\n")
        return

    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def test_single_function():
    """Edge case: Only one function, starts and ends immediately."""
    print("TEST: Single function (starts and ends at same timestamp)")

    n = 1
    logs = ["0:start:0", "0:end:0"]
    expected = [1]

    result = exclusive_time(n, logs)
    if result is None:
        print("  exclusive_time() returned None -- not yet implemented\n")
        return

    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def test_single_function_longer():
    """Single function that runs for multiple time units."""
    print("TEST: Single function (runs for 10 time units)")

    n = 1
    logs = ["0:start:0", "0:end:9"]
    expected = [10]

    result = exclusive_time(n, logs)
    if result is None:
        print("  exclusive_time() returned None -- not yet implemented\n")
        return

    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def test_rapid_start_end():
    """Multiple functions starting and ending rapidly."""
    print("TEST: Rapid start-end sequences")

    n = 3
    logs = [
        "0:start:0", "0:end:0",
        "1:start:1", "1:end:1",
        "2:start:2", "2:end:2",
    ]
    expected = [1, 1, 1]

    result = exclusive_time(n, logs)
    if result is None:
        print("  exclusive_time() returned None -- not yet implemented\n")
        return

    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def test_nested_with_immediate_return():
    """Nested call where inner function returns immediately."""
    print("TEST: Nested with immediate return")

    n = 2
    logs = ["0:start:0", "1:start:1", "1:end:1", "0:end:3"]
    expected = [3, 1]

    result = exclusive_time(n, logs)
    if result is None:
        print("  exclusive_time() returned None -- not yet implemented\n")
        return

    # Function 0: [0,0] + [2,3] = 1 + 2 = 3
    # Function 1: [1,1] = 1
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def test_multiple_children():
    """One parent function calls multiple children sequentially."""
    print("TEST: Multiple children (parent calls 3 different functions)")

    n = 4
    logs = [
        "0:start:0",
        "1:start:1", "1:end:2",
        "2:start:3", "2:end:4",
        "3:start:5", "3:end:6",
        "0:end:7",
    ]
    expected = [2, 2, 2, 2]

    result = exclusive_time(n, logs)
    if result is None:
        print("  exclusive_time() returned None -- not yet implemented\n")
        return

    # Function 0: [0,0] + [7,7] = 1 + 1 = 2
    # Function 1: [1,2] = 2
    # Function 2: [3,4] = 2
    # Function 3: [5,6] = 2
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def test_deep_recursion():
    """Deep recursion of the same function."""
    print("TEST: Deep recursion (same function 4 levels deep)")

    n = 1
    logs = [
        "0:start:0",
        "0:start:1",
        "0:start:2",
        "0:start:3",
        "0:end:4",
        "0:end:5",
        "0:end:6",
        "0:end:7",
    ]
    expected = [8]

    result = exclusive_time(n, logs)
    if result is None:
        print("  exclusive_time() returned None -- not yet implemented\n")
        return

    # Level 0 (outermost): [0,0] + [7,7] = 2
    # Level 1: [1,1] + [6,6] = 2
    # Level 2: [2,2] + [5,5] = 2
    # Level 3 (innermost): [3,4] = 2
    # Total: 2+2+2+2 = 8
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def test_large_timestamps():
    """Timestamps don't start at 0 and can be large."""
    print("TEST: Large timestamps")

    n = 2
    logs = ["0:start:100", "1:start:200", "1:end:300", "0:end:400"]
    expected = [200, 101]

    result = exclusive_time(n, logs)
    if result is None:
        print("  exclusive_time() returned None -- not yet implemented\n")
        return

    # Trace:
    # "0:start:100" -> push 0, prev=100
    # "1:start:200" -> result[0] += 200-100 = 100, push 1, prev=200
    # "1:end:300"   -> result[1] += 300-200+1 = 101, pop 1, prev=301
    # "0:end:400"   -> result[0] += 400-301+1 = 100, pop 0, prev=401
    # result = [200, 101]
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


if __name__ == "__main__":
    test_basic_nesting()
    test_recursive()
    test_sequential()
    test_deeply_nested()
    test_single_function()
    test_single_function_longer()
    test_rapid_start_end()
    test_nested_with_immediate_return()
    test_multiple_children()
    test_deep_recursion()
    test_large_timestamps()

    print("=" * 60)
    print("All tests complete!")
    print("=" * 60)
