# Rubric: LRU Cache with Persistence

**Total: 100 points**

---

## Part 1: Bug Identification and Fixes (60 points)

### Bug 1: Cache key does not properly include kwargs values (15 points)

**The bug:**
```python
key = args + tuple(kwargs)
```
`tuple(kwargs)` produces a tuple of only the **keys** of the kwargs dict, discarding the
values entirely. For example:
```python
tuple({"greeting": "hello"})  # -> ("greeting",)
tuple({"greeting": "hi"})     # -> ("greeting",)  -- SAME KEY!
```

**The fix:**
```python
key = args + tuple(sorted(kwargs.items()))
```
This captures both kwarg keys and values. The `sorted()` ensures consistent ordering
regardless of the order kwargs are passed.

| Points | Criteria |
|--------|----------|
| 15 | Correctly identifies the bug AND provides a working fix with sorted items |
| 10 | Identifies the bug but fix is incomplete (e.g., no sorting) |
| 5  | Mentions kwargs might be a problem but doesn't fully understand it |
| 0  | Misses this bug entirely |

**Bonus discussion (no extra points, but positive signal):**
- What if kwarg values are unhashable (e.g., lists)? The key would fail as a dict key.
- Could use `repr()` or `pickle` for a more robust key, but with performance trade-offs.

---

### Bug 2: No LRU update on cache hit (20 points)

**The bug:**
```python
if key in cache:
    return cache[key]  # returns immediately without updating order!
```
When a cached entry is accessed, it should be moved to the "most recently used" position.
The buggy code returns immediately, leaving `order` unchanged. This means a frequently
accessed item can be evicted because it's still at the front of `order`.

**The fix:**
```python
if key in cache:
    order.remove(key)      # remove from current position
    order.append(key)      # add to end (most recent)
    return cache[key]
```
Or better, switch to `collections.OrderedDict`:
```python
if key in cache:
    cache.move_to_end(key)
    return cache[key]
```

| Points | Criteria |
|--------|----------|
| 20 | Correctly identifies the bug AND provides a working fix that moves the key to most-recent position |
| 15 | Identifies the bug but fix has subtle issues (e.g., doesn't handle duplicates in order list) |
| 5  | Mentions something about ordering but doesn't fully articulate the bug |
| 0  | Misses this bug entirely |

**This is the most important bug** -- it fundamentally breaks the LRU semantics.

---

### Bug 3: O(n) list operations (10 points)

**The issue:**
```python
del order[0]         # O(n) -- shifts all elements
order.remove(key)    # O(n) -- linear scan (if they fix bug 2)
```
Using a Python list for the LRU order gives O(n) eviction and O(n) access-reordering.

**The fix:** Use `collections.OrderedDict` which provides O(1) for all operations:
```python
from collections import OrderedDict
cache = OrderedDict()

# On hit:
cache.move_to_end(key)

# On insert:
cache[key] = result

# On eviction:
cache.popitem(last=False)  # removes oldest
```

| Points | Criteria |
|--------|----------|
| 10 | Recognizes the O(n) issue and proposes OrderedDict or doubly-linked list |
| 7  | Recognizes it's inefficient but doesn't propose a concrete O(1) solution |
| 3  | Mentions performance but doesn't connect it to the list operations |
| 0  | Doesn't address performance at all |

**Note:** This is less of a "bug" and more of a performance concern, but upgrading to
OrderedDict also makes bug 2 trivially fixable, so it shows deeper understanding.

---

### Bug 4 (bonus): Potential duplicate entries in order list (5 points)

**The issue:** If the candidate fixes bug 2 by adding `order.remove(key)` + `order.append(key)`
on cache hit, the original code on cache **miss** also does `order.append(key)`. If a key is
evicted and then re-inserted, this is fine. But if there's a code path where `append` is called
without proper cleanup, duplicates could accumulate.

More critically: the original code appends to `order` on every cache miss but never checks
if the key is somehow already present (it shouldn't be, but defensive coding matters).

| Points | Criteria |
|--------|----------|
| 5  | Identifies and discusses potential duplicate entries or integrity issues with the order list |
| 0  | Doesn't mention this |

---

### Overall Part 1 Assessment (10 points)

| Points | Criteria |
|--------|----------|
| 10 | Complete rewrite using OrderedDict with clean, correct implementation |
| 7  | Fixes all bugs but code is somewhat messy |
| 3  | Fixes some bugs, misses others |
| 0  | Minimal fixes, fundamental issues remain |

---

## Part 2: Persistence (40 points)

### Serialization Method (10 points)

| Points | Criteria |
|--------|----------|
| 10 | Uses json (for human-readable) or pickle (for flexibility) with clear rationale |
| 7  | Uses a serialization method that works but doesn't discuss trade-offs |
| 3  | Attempts serialization but it's partially broken |
| 0  | No serialization attempted |

**Discussion points:**
- `json`: Human-readable, but limited to JSON-serializable types. Keys must be strings.
- `pickle`: Handles arbitrary Python objects, but security concerns (arbitrary code execution).
- `shelve`: Built-in key-value store, but complex for this use case.
- Candidate should acknowledge the trade-offs.

---

### Preserving LRU Order (20 points)

| Points | Criteria |
|--------|----------|
| 20 | Save and load correctly preserve the exact LRU ordering; eviction after load behaves correctly |
| 15 | LRU order is mostly preserved but with minor issues |
| 10 | Data is saved/loaded but ordering is lost |
| 5  | Partial implementation |
| 0  | Not attempted |

**Key insight:** If using `OrderedDict`, iteration order is insertion order, so
`list(cache.items())` preserves the LRU order naturally. On load, insert items in the
same order to restore the LRU state.

If using a list + dict approach, both must be saved and restored in the correct order.

---

### Edge Case Handling (10 points)

| Points | Criteria |
|--------|----------|
| 10 | Handles non-serializable values gracefully, file not found, corrupt file |
| 7  | Handles most edge cases |
| 3  | Basic save/load works but crashes on edge cases |
| 0  | No edge case handling |

**Edge cases:**
- What if cache contains non-serializable values (functions, file handles)?
- What if the file doesn't exist on load?
- What if the file is corrupted or truncated?
- What if the saved cache has more entries than current max_size?
- What if the saved format is from an older version?

---

## Ideal Complete Solution

```python
from collections import OrderedDict
import json
from typing import Any, Callable


def memoize(max_size: int):
    def decorator(func: Callable) -> Callable:
        cache: OrderedDict = OrderedDict()

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            if len(cache) > max_size:
                cache.popitem(last=False)
            return result

        def save(filepath: str) -> None:
            data = {
                "max_size": max_size,
                "entries": [
                    {"key": repr(k), "value": v}
                    for k, v in cache.items()
                ],
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        def load(filepath: str) -> None:
            with open(filepath, "r") as f:
                data = json.load(f)
            cache.clear()
            for entry in data["entries"]:
                key = eval(entry["key"])  # Note: eval is dangerous in production
                cache[key] = entry["value"]

        wrapper.cache_info = lambda: {"size": len(cache), "max_size": max_size}
        wrapper.save = save
        wrapper.load = load
        return wrapper
    return decorator
```

---

## Red Flags

- Cannot identify the kwargs bug even with the failing test case in front of them
- Doesn't understand what "least recently used" means (confuses with "least frequently used")
- Adds `order.remove(key)` in the hit path but doesn't realize this is O(n)
- Uses `eval()` for deserialization without acknowledging the security risk
- Persistence doesn't preserve order at all

## Green Flags

- Immediately reaches for `OrderedDict` or mentions doubly-linked list + dict
- Discusses `functools.lru_cache` and how it's implemented in CPython
- Mentions thread safety concerns unprompted
- Considers the security implications of pickle/eval for persistence
- Tests their fix against the provided test cases mentally before writing code
