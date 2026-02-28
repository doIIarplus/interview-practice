# Follow-Up Questions: Find Duplicate File in System (Question 20)

## 1. What are the time and space complexities of your solution?

**Expected discussion:**

**Part 1:**
- Time: O(N * L) where N is the number of path strings and L is the average length of each string. Parsing is linear, and hashmap operations are O(1) amortized.
- Space: O(F * C) where F is the total number of files and C is the average content length. The hashmap stores all file paths and uses content as keys.

**Part 2:**
- Time: O(F) for listing files + O(F) for size grouping + O(D * S) for hashing candidate duplicates, where D is the number of candidate files and S is average file size.
- Space: O(F) for grouping data structures.
- The key insight: size filtering is O(F) with no I/O beyond metadata, and it eliminates most files. Only a small subset D proceeds to the expensive hashing step.

---

## 2. How would you handle symbolic links (could create infinite loops)?

**Expected discussion:**
- Symbolic links can create cycles in the directory tree (A -> B -> A).
- **Detection:** Track visited inodes (not paths). Two paths can point to the same file via symlinks; the inode is the canonical identifier.
- **Strategy 1:** Resolve all symlinks to their real paths first (`os.path.realpath()`), then deduplicate by real path before processing.
- **Strategy 2:** Track visited directories by inode during traversal, skip already-visited directories.
- **Strategy 3:** Don't follow symlinks at all (`os.walk` with `followlinks=False`).
- Also consider: should two symlinks pointing to the same file be reported as "duplicates"? Depends on use case -- deduplication (yes) vs. cleanup (no, they share storage).

---

## 3. What if you need to find duplicates across a distributed file system (multiple machines)?

**Expected discussion:**
- **MapReduce approach:**
  - **Map phase:** Each machine computes `(size, path)` for its local files.
  - **Shuffle:** Group by size across machines.
  - **Reduce phase:** For each size group, compute hashes. Group by `(size, hash)`.
- **Distributed hashing:** Use consistent hashing to assign files to processing nodes based on their size (so same-size files end up on the same node for comparison).
- **Bandwidth optimization:** Only send file metadata (size, path) in the first phase. Only transfer file content (or hashes) when needed for comparison.
- **Bloom filters:** Use a distributed Bloom filter to quickly check if a file's hash has been seen on another machine.
- Real-world tools: HDFS + Spark, or cloud object storage with Lambda/serverless functions.

---

## 4. How would you make this work in real-time (detect duplicates as files are added)?

**Expected discussion:**
- **File system watcher:** Use `inotify` (Linux) or `FSEvents` (macOS) to detect new/modified files.
- **Incremental indexing:** Maintain a persistent index of `(size, hash) -> [paths]`. When a new file is added:
  1. Check if any existing file has the same size.
  2. If yes, compute hash and check for hash match.
  3. Update the index.
- **Database:** Store the index in a database (SQLite, PostgreSQL) for persistence and fast lookups.
- **Trade-off:** Pre-computing hashes for all files uses disk I/O upfront but makes future duplicate detection instant. Lazy hashing (only hash when a size collision occurs) saves I/O but may be slower for frequent additions.
- Consider: file modifications (hash invalidation), file deletions (index cleanup), file renames (path update but same inode).

---

## 5. What's the probability of a hash collision with MD5? SHA256? Is it safe to rely on hash alone?

**Expected discussion:**
- **MD5 (128-bit):** Probability of collision with N files is approximately N^2 / 2^129 (birthday paradox). With 1 billion files, probability is ~10^-20. However, MD5 is cryptographically broken -- **intentional** collisions can be constructed.
- **SHA-256 (256-bit):** Collision probability with N files is ~N^2 / 2^257. Effectively zero for any practical number of files. No known collision has ever been found.
- **Is hash alone safe?**
  - For deduplication: SHA-256 is safe in practice. The probability of an accidental collision is astronomically low.
  - For security-critical applications: Consider hash + size + byte comparison for paranoia.
  - Git uses SHA-1 (160-bit) for content addressing and has worked fine for billions of objects.
- **Practical advice:** SHA-256 alone is sufficient for almost all use cases. The cost of a false positive (treating different files as identical) is much higher than the cost of a byte-by-byte verification, so some systems do both.

---

## 6. How would you parallelize this for a large file system?

**Expected discussion:**
- **Stage 1 (listing/sizing):** Parallelize directory traversal across threads. Each thread walks a subtree and collects `(path, size)` pairs.
- **Stage 2 (hashing):** Use a thread pool to hash files in parallel. I/O-bound work benefits from threads (especially on SSDs where random reads are fast).
- **Stage 3 (comparison):** Chunk comparisons can be parallelized across file pairs.
- **Pipeline parallelism:** Start hashing files as soon as size groups are identified, don't wait for all sizes to be collected.
- **Memory management:** Process size groups in batches to avoid loading too many hashes into memory.
- **I/O scheduling:** On HDDs, sequential reads are much faster than random reads. Consider sorting files by disk location before reading.
- **Producer-consumer pattern:** One thread discovers files, another thread groups by size, another thread computes hashes. Connected by queues.
