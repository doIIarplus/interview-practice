"""Hidden tests for Question 21: Exclusive Time of Functions
Run: python questions/21_exclusive_time/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import exclusive_time


def test_basic_nesting():
    """Function 0 calls function 1."""
    print("TEST: Basic nesting")
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
    """Function 0 calls itself (recursion)."""
    print("TEST: Recursive call")
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
    """Sequential calls, no nesting."""
    print("TEST: Sequential calls")
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
    """Three levels of nesting."""
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
    """Single function, starts and ends immediately."""
    print("TEST: Single function")
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
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def test_multiple_children():
    """One parent function calls multiple children sequentially."""
    print("TEST: Multiple children")
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
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  PASS: {result}\n")


def run_tests():
    print("=" * 60)
    print("Exclusive Time of Functions — Hidden Tests")
    print("=" * 60 + "\n")

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


if __name__ == "__main__":
    run_tests()
