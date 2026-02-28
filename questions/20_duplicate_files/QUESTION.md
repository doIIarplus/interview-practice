# Question 20: Find Duplicate File in System

*(Based on LeetCode 609)*

## Problem

Given a list of directory info strings, find all groups of duplicate files in the file system. A group of duplicate files consists of at least 2 files that have the same content.

Each string in the input has the format:

```
"directory_path file1.txt(content1) file2.txt(content2) ... fn.txt(contentn)"
```

- The first token is the directory path.
- Each subsequent token is a filename with its content in parentheses.

Return a list of groups of duplicate file paths. A file path is constructed as `directory_path/filename`.

---

## Part 1: String Parsing Solution

Implement:

```python
def find_duplicates(paths: list[str]) -> list[list[str]]:
```

### Example

```python
paths = [
    "root/a 1.txt(abcd) 2.txt(efgh)",
    "root/c 3.txt(abcd)",
    "root/c/d 4.txt(efgh)",
    "root 4.txt(efgh)"
]

# Output (order of groups and within groups doesn't matter):
[
    ["root/a/1.txt", "root/c/3.txt"],              # content "abcd"
    ["root/a/2.txt", "root/c/d/4.txt", "root/4.txt"]  # content "efgh"
]
```

### More Examples

```python
# Single directory, no duplicates
paths = ["root/a 1.txt(abc) 2.txt(def) 3.txt(ghi)"]
# Output: []  (no duplicates)

# Multiple directories, all same content
paths = [
    "root/a 1.txt(same)",
    "root/b 2.txt(same)",
    "root/c 3.txt(same)"
]
# Output: [["root/a/1.txt", "root/b/2.txt", "root/c/3.txt"]]

# Empty content
paths = ["root/a 1.txt() 2.txt()"]
# Output: [["root/a/1.txt", "root/a/2.txt"]]
```

---

## Part 2: Optimized Real-World Solution (Follow-Up)

In the real world, file contents are not given as strings. Files live on disk, can be very large (e.g., 10GB), and you can only interact with them through system calls.

Given a `FileSystem` interface:

```python
class FileSystem:
    def list_files(self, directory: str) -> list[str]:
        """List all file paths under a directory (recursive)."""

    def get_size(self, path: str) -> int:
        """Get file size in bytes. Very cheap O(1) call."""

    def get_hash(self, path: str) -> str:
        """Get SHA-256 hash of file contents. Reads entire file -- expensive for large files."""

    def read_chunk(self, path: str, offset: int, size: int) -> bytes:
        """Read a chunk of bytes from a file starting at offset."""
```

Implement:

```python
def find_duplicates_optimized(fs: FileSystem, root: str) -> list[list[str]]:
```

### Optimization Pipeline

Think about how to minimize I/O by filtering candidates cheaply before doing expensive comparisons:

1. **Group by size** -- Files with different sizes cannot be duplicates. `get_size()` is O(1).
2. **Group by hash** -- Among same-size files, compute hashes. `get_hash()` reads the entire file.
3. **Byte-by-byte comparison** -- For very large files, you might want to compare chunks instead of hashing the entire file. `read_chunk()` lets you bail out early on the first difference.

Consider: When is hashing better than byte-by-byte comparison? When is it worse?

---

## Constraints

- `1 <= paths.length <= 2 * 10^4`
- `1 <= paths[i].length <= 3000`
- File contents consist of lowercase English letters and digits
- No two files in the same directory have the same name
- All directory paths and file names are valid

---

## Starter Code

See `starter.py` for function stubs, the `FileSystem` interface, and test cases for both parts.
