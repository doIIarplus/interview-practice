"""Hidden tests for Question 03: Profiler Trace
Run: python questions/03_profiler_trace/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import samples_to_trace


def test_basic_transitions():
    """Test basic function transitions."""
    samples = [
        ["main", "foo", "bar"],
        ["main", "foo"],
        ["main", "baz"],
    ]
    result = samples_to_trace(samples)
    expected = [
        ("start", "main", 0),
        ("start", "foo", 0),
        ("start", "bar", 0),
        ("end", "bar", 1),
        ("end", "foo", 2),
        ("start", "baz", 2),
        ("end", "baz", 3),
        ("end", "main", 3),
    ]
    assert result == expected, f"Expected {expected}, got {result}"
    print("[PASS] test_basic_transitions")


def test_identical_consecutive():
    """Test identical consecutive samples."""
    samples = [
        ["main", "foo"],
        ["main", "foo"],
        ["main", "bar"],
    ]
    result = samples_to_trace(samples)
    expected = [
        ("start", "main", 0),
        ("start", "foo", 0),
        ("end", "foo", 2),
        ("start", "bar", 2),
        ("end", "bar", 3),
        ("end", "main", 3),
    ]
    assert result == expected, f"Expected {expected}, got {result}"
    print("[PASS] test_identical_consecutive")


def test_recursion():
    """Test recursive calls."""
    samples = [
        ["main", "factorial", "factorial", "factorial"],
        ["main", "factorial", "factorial"],
        ["main", "factorial"],
        ["main"],
    ]
    result = samples_to_trace(samples)
    expected = [
        ("start", "main", 0),
        ("start", "factorial", 0),
        ("start", "factorial", 0),
        ("start", "factorial", 0),
        ("end", "factorial", 1),
        ("end", "factorial", 2),
        ("end", "factorial", 3),
        ("end", "main", 4),
    ]
    assert result == expected, f"Expected {expected}, got {result}"
    print("[PASS] test_recursion")


def test_edge_cases():
    """Test edge cases."""
    assert samples_to_trace([]) == [], "Empty input should return []"
    assert samples_to_trace([["main"]]) == [
        ("start", "main", 0), ("end", "main", 1)
    ], "Single sample should produce start and end"
    assert samples_to_trace([[]]) == [], "Empty stack should return []"
    print("[PASS] test_edge_cases")


def run_tests():
    print("=" * 60)
    print("Profiler Trace — Hidden Tests")
    print("=" * 60 + "\n")

    test_basic_transitions()
    test_identical_consecutive()
    test_recursion()
    test_edge_cases()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
