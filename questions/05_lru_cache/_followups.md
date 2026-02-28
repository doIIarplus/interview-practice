# Follow-up Questions: LRU Cache with Persistence

---

## 1. What's the time complexity of your LRU operations? Can you make all operations O(1)?

**Expected discussion:**
- List-based approach:
  - `get` (cache hit): O(n) due to `list.remove(key)` for reordering
  - `put` (cache miss): O(1) amortized for append, O(n) for eviction via `del order[0]`
- `OrderedDict`-based approach:
  - `get`: O(1) -- `move_to_end()` is O(1) using internal doubly-linked list
  - `put`: O(1) -- insertion is O(1), `popitem(last=False)` is O(1)
- Manual O(1) approach: hash map + doubly-linked list
  - Hash map maps keys to linked list nodes
  - Doubly-linked list maintains order (head = LRU, tail = MRU)
  - All operations (get, put, evict) are O(1)

**Strong answer:** Implements or describes the hash map + doubly-linked list approach,
explaining how node removal and insertion at tail are both O(1) with the doubly-linked list.

---

## 2. Implement this using `collections.OrderedDict` -- what are the trade-offs?

**Expected discussion:**
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
```

**Trade-offs:**
- **Pros**: Simple, Pythonic, all operations O(1), well-tested stdlib
- **Cons**: Slight memory overhead (maintains both dict and linked list internally),
  less control over memory layout, Python-specific (not portable to other languages)
- vs `functools.lru_cache`: stdlib version is C-implemented and faster, but less flexible
  (fixed size, no persistence, no manual eviction)

---

## 3. How would you make this thread-safe for a multi-threaded application?

**Expected discussion:**
- **Basic approach**: Wrap all cache operations with a `threading.Lock`
  ```python
  import threading
  lock = threading.Lock()

  def wrapper(*args, **kwargs):
      key = make_key(args, kwargs)
      with lock:
          if key in cache:
              cache.move_to_end(key)
              return cache[key]
      # Compute outside the lock to avoid blocking other threads
      result = func(*args, **kwargs)
      with lock:
          cache[key] = result
          if len(cache) > max_size:
              cache.popitem(last=False)
      return result
  ```
- **Problem with above**: Two threads could compute the same key simultaneously (thundering herd)
- **Better**: Use a per-key lock or a "computing" sentinel to prevent duplicate computation
- **Read-write lock**: Multiple readers can access cache simultaneously; only writes need exclusive access
- **`functools.lru_cache`**: Already thread-safe in CPython due to the GIL, but not truly concurrent

**Strong answer:** Identifies the thundering herd problem and proposes a solution
(per-key locking, futures/promises, or a "computing" sentinel pattern).

---

## 4. What if you needed to persist the cache to a remote store (Redis) instead of disk?

**Expected discussion:**
- Redis naturally supports key-value storage with TTL and eviction policies
- Use Redis sorted sets for LRU ordering (score = timestamp of last access)
- Challenges:
  - **Serialization**: Need to serialize Python objects to bytes (pickle, msgpack, JSON)
  - **Key format**: Redis keys are strings, so need to serialize the cache key
  - **Atomicity**: Redis operations are atomic, but multi-key operations need transactions (MULTI/EXEC)
  - **Network latency**: Every cache lookup now involves a network round-trip
  - **Consistency**: Multiple processes sharing the cache need to coordinate
- Redis has built-in LRU eviction (`maxmemory-policy allkeys-lru`), so you might not
  need to implement LRU yourself
- Consider a two-level cache: in-memory L1 (fast, small) + Redis L2 (slower, larger, shared)

**Strong answer:** Mentions the latency trade-off and proposes a two-level cache architecture.

---

## 5. How would you add a TTL (time-to-live) to cache entries?

**Expected discussion:**
- Store `(value, expiry_timestamp)` tuples in the cache
- On `get`: check if current time > expiry; if so, treat as cache miss and evict
- On `put`: set expiry = current_time + ttl
- **Lazy vs eager expiration:**
  - Lazy: Check expiry only on access. Simple but stale entries consume memory.
  - Eager: Background thread periodically scans and removes expired entries.
  - Hybrid: Lazy checks + periodic cleanup sweep.
- Consider: Should accessing an expired entry refresh the TTL or treat it as a miss?
- Consider: Different TTLs per entry vs global TTL?

**Strong answer:**
```python
import time

def get(self, key):
    if key in self.cache:
        value, expiry = self.cache[key]
        if time.monotonic() > expiry:
            del self.cache[key]
            return None  # expired
        self.cache.move_to_end(key)
        return value
    return None
```

---

## 6. What would you change for a distributed cache shared across multiple processes?

**Expected discussion:**
- **Shared memory**: `multiprocessing.shared_memory` or memory-mapped files
  - Complex serialization, need locking across processes
- **External store**: Redis, Memcached, or a database
  - Simplest for multiple processes, well-understood semantics
- **Cache invalidation**: The "two hardest problems in computer science"
  - When one process updates a value, others need to know
  - Options: polling, pub/sub notifications, versioning
- **Consistency models:**
  - Strong consistency: Every read sees the latest write (requires coordination)
  - Eventual consistency: Reads may see stale data (simpler, faster)
  - For a cache, eventual consistency is usually acceptable
- **Partitioning**: For very large caches, shard across multiple Redis instances
  using consistent hashing

**Strong answer:** Discusses cache invalidation as the core challenge and proposes
a concrete architecture (e.g., local cache + Redis + pub/sub for invalidation).
