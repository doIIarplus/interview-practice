"""
Question 20: Find Duplicate File in System (LeetCode 609)

Part 1: Parse directory info strings and find duplicate files by content.
Part 2: Optimize for real-world file systems using size -> hash -> chunk pipeline.
"""

from collections import defaultdict
import hashlib


# ===========================================================================
# Part 1: String Parsing Solution
# ===========================================================================

def find_duplicates(paths: list[str]) -> list[list[str]]:
    """
    Find all groups of duplicate files based on their content.

    Each string in paths has the format:
        "directory_path file1.txt(content1) file2.txt(content2) ..."

    Return a list of groups where each group contains file paths with
    identical content. Only include groups with 2+ files.

    Args:
        paths: List of directory info strings.

    Returns:
        List of groups of duplicate file paths.
    """
    pass


# ===========================================================================
# Part 2: Optimized Real-World Solution
# ===========================================================================

class FileSystem:
    """
    Simulated file system interface for testing the optimized solution.

    In a real implementation, these would be actual system calls.
    """

    def __init__(self, files: dict[str, bytes]):
        """
        Args:
            files: Mapping from file path to file content (bytes).
        """
        self._files = files

    def list_files(self, directory: str) -> list[str]:
        """List all file paths under a directory (recursive).

        Returns all files whose path starts with the given directory prefix.
        """
        prefix = directory.rstrip("/") + "/"
        return [
            path for path in self._files
            if path.startswith(prefix) or path == directory
        ]

    def get_size(self, path: str) -> int:
        """Get file size in bytes. Very cheap O(1) call."""
        if path not in self._files:
            raise FileNotFoundError(f"File not found: {path}")
        return len(self._files[path])

    def get_hash(self, path: str) -> str:
        """Get SHA-256 hash of file contents. Reads entire file."""
        if path not in self._files:
            raise FileNotFoundError(f"File not found: {path}")
        return hashlib.sha256(self._files[path]).hexdigest()

    def read_chunk(self, path: str, offset: int, size: int) -> bytes:
        """Read a chunk of bytes from a file starting at offset.

        Returns up to `size` bytes starting at `offset`.
        Returns empty bytes if offset is past end of file.
        """
        if path not in self._files:
            raise FileNotFoundError(f"File not found: {path}")
        content = self._files[path]
        return content[offset:offset + size]


def find_duplicates_optimized(fs: FileSystem, root: str) -> list[list[str]]:
    """
    Find all groups of duplicate files using an optimized pipeline.

    Optimization strategy (implement in this order):
    1. Group files by size -- different sizes cannot be duplicates (cheap).
    2. Within same-size groups, group by hash (more expensive, reads whole file).
    3. (Optional) For very large files, compare chunk-by-chunk to bail out
       early without reading entire files.

    Only return groups with 2+ files.

    Args:
        fs: A FileSystem instance providing file operations.
        root: The root directory to search for duplicates.

    Returns:
        List of groups of duplicate file paths.
    """
    pass


# ===========================================================================
# Tests
# ===========================================================================

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

    # Convert to sets of frozensets for comparison (order doesn't matter)
    result_sets = {frozenset(group) for group in result}
    expected_sets = {
        frozenset(["root/a/1.txt", "root/c/3.txt"]),
        frozenset(["root/a/2.txt", "root/c/d/4.txt", "root/4.txt"]),
    }

    assert result_sets == expected_sets, (
        f"Expected {expected_sets}, got {result_sets}"
    )
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
    assert set(result[0]) == {"root/a/1.txt", "root/b/2.txt", "root/c/3.txt"}, (
        f"Wrong group: {result[0]}"
    )
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
    assert set(result[0]) == {"root/a/1.txt", "root/a/2.txt"}, (
        f"Wrong group: {result[0]}"
    )
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
        frozenset(["dir1/a.txt", "dir2/d.txt"]),       # content "x"
        frozenset(["dir1/b.txt", "dir2/e.txt"]),       # content "y"
        frozenset(["dir1/c.txt", "dir3/f.txt"]),       # content "z"
    }
    assert result_sets == expected_sets, (
        f"Expected {expected_sets}, got {result_sets}"
    )
    print("  PASS\n")


def test_part2_basic():
    """Basic optimized solution test."""
    print("TEST Part 2: Basic optimized example")

    files = {
        "root/a/1.txt": b"hello world",
        "root/a/2.txt": b"different content",
        "root/b/3.txt": b"hello world",         # duplicate of 1.txt
        "root/b/4.txt": b"another thing",
        "root/c/5.txt": b"hello world",         # duplicate of 1.txt
        "root/c/6.txt": b"different content",   # duplicate of 2.txt
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
    assert result_sets == expected_sets, (
        f"Expected {expected_sets}, got {result_sets}"
    )
    print("  PASS\n")


def test_part2_size_filtering():
    """Files with different sizes should be filtered out early."""
    print("TEST Part 2: Size filtering")

    # These files have the same hash prefix but different sizes
    files = {
        "root/short.txt": b"abc",
        "root/long.txt": b"abcdef",
        "root/also_short.txt": b"xyz",
        "root/also_long.txt": b"abcdef",  # duplicate of long.txt
    }
    fs = FileSystem(files)

    result = find_duplicates_optimized(fs, "root")

    if result is None:
        print("  find_duplicates_optimized() returned None -- not yet implemented\n")
        return

    result_sets = {frozenset(group) for group in result}
    expected_sets = {
        frozenset(["root/long.txt", "root/also_long.txt"]),
    }
    assert result_sets == expected_sets, (
        f"Expected {expected_sets}, got {result_sets}"
    )
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
        "root/d.txt": b"aaaa",  # duplicate of a.txt
    }
    fs = FileSystem(files)

    result = find_duplicates_optimized(fs, "root")

    if result is None:
        print("  find_duplicates_optimized() returned None -- not yet implemented\n")
        return

    result_sets = {frozenset(group) for group in result}
    expected_sets = {
        frozenset(["root/a.txt", "root/d.txt"]),
    }
    assert result_sets == expected_sets, (
        f"Expected {expected_sets}, got {result_sets}"
    )
    print("  PASS\n")


def test_part2_large_files_chunk_comparison():
    """Test with larger content to exercise chunk comparison."""
    print("TEST Part 2: Larger files (chunk comparison)")

    # Create files that are identical for the first part but differ later
    base_content = b"A" * 10000
    files = {
        "root/big1.txt": base_content + b"ending1",
        "root/big2.txt": base_content + b"ending2",  # same prefix, different ending
        "root/big3.txt": base_content + b"ending1",  # duplicate of big1
        "root/small.txt": b"tiny",
    }
    fs = FileSystem(files)

    result = find_duplicates_optimized(fs, "root")

    if result is None:
        print("  find_duplicates_optimized() returned None -- not yet implemented\n")
        return

    result_sets = {frozenset(group) for group in result}
    expected_sets = {
        frozenset(["root/big1.txt", "root/big3.txt"]),
    }
    assert result_sets == expected_sets, (
        f"Expected {expected_sets}, got {result_sets}"
    )
    print("  PASS\n")


if __name__ == "__main__":
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
