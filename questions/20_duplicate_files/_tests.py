"""Hidden tests for Question 20: Find Duplicate Files
Run: python questions/20_duplicate_files/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import find_duplicates, FileSystem, find_duplicates_optimized


def test_part1_basic():
    """Basic example from the problem description."""
    print("TEST Part 1: Basic example")
    paths = [
        "root/a 1.txt(abcd) 2.txt(efgh)",
        "root/c 3.txt(abcd)",
        "root/c/d 4.txt(efgh)",
        "root 4.txt(efgh)",
    ]
    result = find_duplicates(paths)
    if result is None:
        print("  find_duplicates() returned None -- not yet implemented\n")
        return
    result_sets = {frozenset(group) for group in result}
    expected_sets = {
        frozenset(["root/a/1.txt", "root/c/3.txt"]),
        frozenset(["root/a/2.txt", "root/c/d/4.txt", "root/4.txt"]),
    }
    assert result_sets == expected_sets, f"Expected {expected_sets}, got {result_sets}"
    print("  PASS\n")


def test_part1_no_duplicates():
    """No duplicate files."""
    print("TEST Part 1: No duplicates")
    paths = ["root/a 1.txt(abc) 2.txt(def) 3.txt(ghi)"]
    result = find_duplicates(paths)
    if result is None:
        print("  find_duplicates() returned None -- not yet implemented\n")
        return
    assert result == [], f"Expected [], got {result}"
    print("  PASS\n")


def test_part1_all_same():
    """All files have the same content."""
    print("TEST Part 1: All same content")
    paths = [
        "root/a 1.txt(same)",
        "root/b 2.txt(same)",
        "root/c 3.txt(same)",
    ]
    result = find_duplicates(paths)
    if result is None:
        print("  find_duplicates() returned None -- not yet implemented\n")
        return
    assert len(result) == 1, f"Expected 1 group, got {len(result)}"
    assert set(result[0]) == {"root/a/1.txt", "root/b/2.txt", "root/c/3.txt"}
    print("  PASS\n")


def test_part1_empty_content():
    """Files with empty content are duplicates of each other."""
    print("TEST Part 1: Empty content")
    paths = ["root/a 1.txt() 2.txt()"]
    result = find_duplicates(paths)
    if result is None:
        print("  find_duplicates() returned None -- not yet implemented\n")
        return
    assert len(result) == 1, f"Expected 1 group, got {len(result)}"
    assert set(result[0]) == {"root/a/1.txt", "root/a/2.txt"}
    print("  PASS\n")


def test_part1_multiple_groups():
    """Multiple groups of duplicates."""
    print("TEST Part 1: Multiple duplicate groups")
    paths = [
        "dir1 a.txt(x) b.txt(y) c.txt(z)",
        "dir2 d.txt(x) e.txt(y)",
        "dir3 f.txt(z) g.txt(w)",
    ]
    result = find_duplicates(paths)
    if result is None:
        print("  find_duplicates() returned None -- not yet implemented\n")
        return
    result_sets = {frozenset(group) for group in result}
    expected_sets = {
        frozenset(["dir1/a.txt", "dir2/d.txt"]),
        frozenset(["dir1/b.txt", "dir2/e.txt"]),
        frozenset(["dir1/c.txt", "dir3/f.txt"]),
    }
    assert result_sets == expected_sets, f"Expected {expected_sets}, got {result_sets}"
    print("  PASS\n")


def test_part2_basic():
    """Basic optimized solution test."""
    print("TEST Part 2: Basic optimized example")
    files = {
        "root/a/1.txt": b"hello world",
        "root/a/2.txt": b"different content",
        "root/b/3.txt": b"hello world",
        "root/b/4.txt": b"another thing",
        "root/c/5.txt": b"hello world",
        "root/c/6.txt": b"different content",
    }
    fs = FileSystem(files)
    result = find_duplicates_optimized(fs, "root")
    if result is None:
        print("  find_duplicates_optimized() returned None -- not yet implemented\n")
        return
    result_sets = {frozenset(group) for group in result}
    expected_sets = {
        frozenset(["root/a/1.txt", "root/b/3.txt", "root/c/5.txt"]),
        frozenset(["root/a/2.txt", "root/c/6.txt"]),
    }
    assert result_sets == expected_sets, f"Expected {expected_sets}, got {result_sets}"
    print("  PASS\n")


def test_part2_size_filtering():
    """Files with different sizes should be filtered out early."""
    print("TEST Part 2: Size filtering")
    files = {
        "root/short.txt": b"abc",
        "root/long.txt": b"abcdef",
        "root/also_short.txt": b"xyz",
        "root/also_long.txt": b"abcdef",
    }
    fs = FileSystem(files)
    result = find_duplicates_optimized(fs, "root")
    if result is None:
        print("  find_duplicates_optimized() returned None -- not yet implemented\n")
        return
    result_sets = {frozenset(group) for group in result}
    expected_sets = {frozenset(["root/long.txt", "root/also_long.txt"])}
    assert result_sets == expected_sets, f"Expected {expected_sets}, got {result_sets}"
    print("  PASS\n")


def test_part2_no_duplicates():
    """No duplicates in the file system."""
    print("TEST Part 2: No duplicates")
    files = {
        "root/a.txt": b"unique1",
        "root/b.txt": b"unique2",
        "root/c.txt": b"unique3",
    }
    fs = FileSystem(files)
    result = find_duplicates_optimized(fs, "root")
    if result is None:
        print("  find_duplicates_optimized() returned None -- not yet implemented\n")
        return
    assert result == [], f"Expected [], got {result}"
    print("  PASS\n")


def test_part2_same_size_different_content():
    """Files with same size but different content should not be grouped."""
    print("TEST Part 2: Same size, different content")
    files = {
        "root/a.txt": b"aaaa",
        "root/b.txt": b"bbbb",
        "root/c.txt": b"cccc",
        "root/d.txt": b"aaaa",
    }
    fs = FileSystem(files)
    result = find_duplicates_optimized(fs, "root")
    if result is None:
        print("  find_duplicates_optimized() returned None -- not yet implemented\n")
        return
    result_sets = {frozenset(group) for group in result}
    expected_sets = {frozenset(["root/a.txt", "root/d.txt"])}
    assert result_sets == expected_sets, f"Expected {expected_sets}, got {result_sets}"
    print("  PASS\n")


def test_part2_large_files_chunk_comparison():
    """Test with larger content to exercise chunk comparison."""
    print("TEST Part 2: Larger files (chunk comparison)")
    base_content = b"A" * 10000
    files = {
        "root/big1.txt": base_content + b"ending1",
        "root/big2.txt": base_content + b"ending2",
        "root/big3.txt": base_content + b"ending1",
        "root/small.txt": b"tiny",
    }
    fs = FileSystem(files)
    result = find_duplicates_optimized(fs, "root")
    if result is None:
        print("  find_duplicates_optimized() returned None -- not yet implemented\n")
        return
    result_sets = {frozenset(group) for group in result}
    expected_sets = {frozenset(["root/big1.txt", "root/big3.txt"])}
    assert result_sets == expected_sets, f"Expected {expected_sets}, got {result_sets}"
    print("  PASS\n")


def run_tests():
    print("=" * 60)
    print("Part 1: String Parsing Solution")
    print("=" * 60 + "\n")

    test_part1_basic()
    test_part1_no_duplicates()
    test_part1_all_same()
    test_part1_empty_content()
    test_part1_multiple_groups()

    print("=" * 60)
    print("Part 2: Optimized Real-World Solution")
    print("=" * 60 + "\n")

    test_part2_basic()
    test_part2_size_filtering()
    test_part2_no_duplicates()
    test_part2_same_size_different_content()
    test_part2_large_files_chunk_comparison()

    print("=" * 60)
    print("All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
