# Rubric: Find Duplicate File in System (Question 20)

## Grading Criteria

### 1. Correct String Parsing (20%)

- Correctly splits each path string into directory and file entries
- Correctly extracts filename and content from `filename(content)` format
- Constructs full file paths as `directory/filename`

**Full marks:** Handles all edge cases: empty content `()`, directory with single file, multiple files per directory.
**Partial:** Works for basic cases but fails on edge cases (e.g., empty content).
**Minimal:** Incorrect parsing, wrong path construction.

Correct parsing approach:
```python
parts = path_str.split()
directory = parts[0]
for file_entry in parts[1:]:
    paren_idx = file_entry.index("(")
    filename = file_entry[:paren_idx]
    content = file_entry[paren_idx + 1:-1]  # strip ( and )
    full_path = directory + "/" + filename
```

---

### 2. Correct Grouping by Content Using HashMap (20%)

- Uses a dictionary/hashmap to group files by content
- Only returns groups with 2+ files

**Full marks:** Clean use of `defaultdict(list)` or similar. Correctly filters singleton groups.
**Partial:** Correct grouping but includes singleton groups (files with unique content).
**Minimal:** No grouping, or grouping by wrong key.

```python
content_map = defaultdict(list)
# ... populate ...
return [group for group in content_map.values() if len(group) >= 2]
```

---

### 3. Clean Code, Correct Path Construction (15%)

- Code is well-organized and readable
- Path construction handles edge cases (no double slashes, etc.)
- Good variable naming

**Full marks:** Clean, Pythonic code. Proper use of standard library.
**Partial:** Works but messy or hard to follow.
**Minimal:** Very hard to read, poor structure.

---

### 4. Part 2: Size-First Filtering Optimization (15%)

- Groups files by size before doing any content comparison
- Eliminates files with unique sizes immediately
- Explains why this is cheap (metadata lookup, no file read needed)

**Full marks:** Correctly implements size grouping, only proceeds to hash for groups of 2+.
**Partial:** Groups by size but doesn't skip singleton groups.
**Minimal:** No size filtering, goes straight to hashing.

```python
# Step 1: Group by size
size_groups = defaultdict(list)
for path in all_files:
    size_groups[fs.get_size(path)].append(path)

# Only check groups with 2+ files
candidates = [group for group in size_groups.values() if len(group) >= 2]
```

---

### 5. Part 2: Hash-Based Comparison (15%)

- Within same-size groups, computes hash to find duplicates
- Uses the hash as a grouping key
- Explains trade-off: hashing reads the entire file but produces a small fingerprint

**Full marks:** Correct hash grouping within size groups. Understands the pipeline.
**Partial:** Uses hashing but doesn't combine with size filtering.
**Minimal:** No hash-based comparison.

```python
# Step 2: Within each size group, group by hash
for group in candidates:
    hash_groups = defaultdict(list)
    for path in group:
        hash_groups[fs.get_hash(path)].append(path)
    # Groups with 2+ files are duplicates
```

---

### 6. Part 2: Chunk-by-Chunk Comparison for Very Large Files (15%)

- For very large files, compares chunks sequentially
- Bails out on the first differing chunk (avoids reading entire file)
- Explains when this is better than hashing (comparing 2 files that differ early)
- Explains when hashing is better (comparing N files -- hash once, compare hashes)

**Full marks:** Implements chunk comparison, explains trade-offs clearly.
**Partial:** Mentions the idea but doesn't implement or has bugs.
**Minimal:** No discussion of chunk comparison.

```python
def files_equal_by_chunks(fs, path1, path2, chunk_size=4096):
    """Compare two files chunk by chunk. Bail out on first difference."""
    offset = 0
    while True:
        chunk1 = fs.read_chunk(path1, offset, chunk_size)
        chunk2 = fs.read_chunk(path2, offset, chunk_size)
        if chunk1 != chunk2:
            return False
        if len(chunk1) == 0:  # Both reached end of file
            return True
        offset += chunk_size
```

---

## Common Approaches

### Part 1: Standard HashMap Approach

```python
def find_duplicates(paths):
    content_to_files = defaultdict(list)
    for path_str in paths:
        parts = path_str.split()
        directory = parts[0]
        for file_entry in parts[1:]:
            paren_idx = file_entry.index("(")
            filename = file_entry[:paren_idx]
            content = file_entry[paren_idx + 1:-1]
            full_path = f"{directory}/{filename}"
            content_to_files[content].append(full_path)
    return [files for files in content_to_files.values() if len(files) >= 2]
```

### Part 2: Three-Stage Pipeline

```python
def find_duplicates_optimized(fs, root):
    all_files = fs.list_files(root)

    # Stage 1: Group by size
    size_groups = defaultdict(list)
    for path in all_files:
        size_groups[fs.get_size(path)].append(path)

    # Stage 2: Within same-size groups, group by hash
    duplicates = []
    for group in size_groups.values():
        if len(group) < 2:
            continue
        hash_groups = defaultdict(list)
        for path in group:
            hash_groups[fs.get_hash(path)].append(path)
        for hash_group in hash_groups.values():
            if len(hash_group) >= 2:
                duplicates.append(hash_group)

    return duplicates
```

### Part 2 (Advanced): With Chunk Comparison

For very large files, replace or augment the hash step:
1. Compare small initial chunks first (first 4KB) -- quick filter
2. If initial chunks match, compare full hash or do full chunk comparison
3. This avoids reading entire 10GB files when they differ in the first few bytes

---

## Red Flags

- Not handling the parentheses parsing correctly (off-by-one)
- Including singleton groups in output
- Part 2: Hashing all files without size filtering first
- Part 2: Not understanding when chunk comparison is beneficial
- Part 2: Loading entire file into memory for comparison

## Green Flags

- Clean parsing with clear variable names
- Explains the optimization pipeline clearly
- Discusses hash collision probability
- Mentions that `get_size` is an inode/metadata lookup (O(1), no disk read)
- Considers the case where N > 2 files have the same size (hash is better than N*(N-1)/2 pairwise chunk comparisons)

## Complexity Analysis

**Part 1:**
- Time: O(N * L) where N = number of path strings, L = average length
- Space: O(N * L) for the hashmap

**Part 2:**
- Time: O(F) for listing + O(F) for size grouping + O(D * S) for hashing duplicates
  - F = total files, D = files in candidate groups, S = average file size
- Space: O(F) for the groupings
- With chunk comparison: Best case O(C) per pair where C = chunk size (bail out early)
