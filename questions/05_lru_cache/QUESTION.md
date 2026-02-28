# Question 05: LRU Cache with Persistence

## Part 1: Debug the Memoization Decorator

You are given a buggy implementation of a memoization decorator that uses an LRU
(Least Recently Used) cache. The decorator should cache function results based on
their arguments, evicting the least recently used entry when the cache exceeds its
capacity.

Here is the buggy implementation:

```python
def memoize(max_size: int):
    def decorator(func):
        cache = {}
        order = []

        def wrapper(*args, **kwargs):
            key = args + tuple(kwargs)
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            order.append(key)
            if len(cache) > max_size:
                old = order[0]
                del order[0]
                del cache[old]
            return result

        wrapper.cache_info = lambda: {"size": len(cache), "max_size": max_size}
        return wrapper
    return decorator
```

**Find and fix all the bugs.** There are at least 3 bugs.

### Test Cases That Expose Bugs

```python
@memoize(max_size=2)
def add(a, b):
    return a + b

# Test 1: Basic caching
assert add(1, 2) == 3
assert add(1, 2) == 3  # should be cached

# Test 2: LRU eviction order
add(3, 4)  # cache: {(1,2): 3, (3,4): 7}
add(1, 2)  # access (1,2) again -- should make it "most recently used"
add(5, 6)  # should evict (3,4), NOT (1,2)
assert add(1, 2) == 3  # should still be cached
# But with the buggy code, (1,2) might have been evicted instead!

# Test 3: kwargs handling
@memoize(max_size=3)
def greet(name, greeting="hello"):
    return f"{greeting}, {name}!"

result1 = greet("Alice", greeting="hello")
result2 = greet("Alice", greeting="hi")
# These should be cached under DIFFERENT keys
# But the buggy key generation may not distinguish them properly
```

## Part 2: Add Persistence

Extend your corrected cache with persistence capabilities:

1. **`save(filepath: str)`** -- Serialize the cache to disk, preserving LRU ordering.
2. **`load(filepath: str)`** -- Restore the cache from disk with correct LRU ordering.

Requirements:
- After `save()` and `load()`, the cache should behave identically to before saving.
- LRU ordering must be preserved (the least recently used item should still be the
  first to be evicted after loading).
- Handle cases where the saved data contains entries that can't be deserialized.

### Example Usage

```python
@memoize(max_size=100)
def expensive_computation(x: int) -> int:
    return x ** 2

# Populate cache
for i in range(50):
    expensive_computation(i)

# Save to disk
expensive_computation.save("cache.json")

# ... process restarts ...

# Restore cache
expensive_computation.load("cache.json")

# This should be a cache hit (no recomputation)
result = expensive_computation(25)
```

## Getting Started

See `starter.py` for the buggy implementation and test cases.
