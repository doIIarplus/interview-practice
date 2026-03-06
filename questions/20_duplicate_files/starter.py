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


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    paths = ["root/a 1.txt(hello) 2.txt(world)", "root/b 3.txt(hello)"]
    result = find_duplicates(paths)
    print(f"Duplicates: {result}")
