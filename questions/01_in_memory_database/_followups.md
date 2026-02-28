# Follow-Up Questions: In-Memory Database

> **This file is hidden from the candidate.**

## Follow-Up 1: Thread Safety

**Question:** How would you make this database thread-safe?

**What to look for:**
- Mentions `threading.Lock` or `threading.RLock` for mutual exclusion
- Understands the difference between a lock per database vs. a lock per key (granularity tradeoffs)
- Discusses read-write locks (`threading.RLock` or a reader-writer pattern) for better read concurrency
- Bonus: mentions `concurrent.futures` or `queue`-based serialization patterns
- Bonus: discusses MVCC (multi-version concurrency control) as an alternative to locking, which pairs naturally with the timestamped design

**Red flags:**
- Suggests making everything immutable without discussing the performance implications
- Doesn't understand what a race condition is in this context
- Suggests "just use asyncio" (asyncio is single-threaded, doesn't solve shared-state concurrency)

---

## Follow-Up 2: Scaling to Millions of Keys

**Question:** What if this needed to handle millions of keys? What data structures would you change?

**What to look for:**
- For prefix scanning: suggests a trie or sorted data structure (e.g., `SortedDict` from `sortedcontainers`)
- For timestamp lookups: mentions binary search (`bisect`) if not already using it; discusses B-tree or skip list for on-disk variants
- Discusses memory overhead of storing full history for every field
- Mentions sharding/partitioning by key for horizontal scaling
- Discusses memory-mapped files or memory-efficient representations
- Bonus: mentions bloom filters for quick "does this key exist?" checks
- Bonus: discusses LSM trees (Log-Structured Merge Trees) as used by LevelDB/RocksDB

**Red flags:**
- No awareness of memory constraints
- Suggests "just add more RAM"
- Doesn't consider the scan/prefix operations becoming bottlenecks

---

## Follow-Up 3: Transaction Support

**Question:** How would you add transaction support (BEGIN, COMMIT, ROLLBACK)?

**What to look for:**
- Suggests a write-ahead log (WAL) or undo log approach
- Discusses keeping a "pending changes" buffer that gets merged on COMMIT
- Understands isolation levels: do reads within a transaction see uncommitted writes from other transactions?
- Mentions snapshot isolation as a natural fit with the timestamped design
- Discusses what happens if two transactions modify the same field (conflict detection)
- Bonus: mentions two-phase commit for distributed transactions

**Red flags:**
- No concept of isolation -- just "save and restore the whole database"
- Doesn't consider concurrent transactions
- Overly complex solutions for a single-user scenario

---

## Follow-Up 4: Persistence to Disk

**Question:** What if you needed to persist this to disk? How would you design the file format?

**What to look for:**
- Suggests append-only log file (simple, crash-safe with fsync)
- Discusses write-ahead logging (WAL) for crash recovery
- Mentions periodic snapshots + log replay for fast startup
- Considers the tradeoff between write performance (append-only) and read performance (indexed)
- Discusses file format options: JSON lines, binary format, Protocol Buffers, etc.
- Mentions compaction of the log file as analogous to Level 4's COMPACT operation
- Bonus: discusses mmap for memory-mapped I/O
- Bonus: mentions SSTable / LSM tree approach (as used by LevelDB, RocksDB, Cassandra)

**Red flags:**
- Suggests writing the entire database to disk on every operation
- No consideration of crash safety (partial writes)
- Suggests using SQLite or another existing database (misses the point of the question)
