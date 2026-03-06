"""
Question 05: LRU Cache with Persistence

Part 1: Find and fix the bugs in the memoize decorator below.
Part 2: Add save() and load() methods for persistence.

See QUESTION.md for full problem description.
"""

from __future__ import annotations

import functools
from typing import Any, Callable


# ---------------------------------------------------------------------------
# BUGGY IMPLEMENTATION -- Find and fix at least 3 bugs
# ---------------------------------------------------------------------------

def memoize(max_size: int):
    """
    A memoization decorator with LRU eviction.

    Caches function results based on their arguments. When the cache exceeds
    max_size, evicts the least recently used entry.

    Args:
        max_size: Maximum number of entries in the cache.

    Returns:
        A decorator that wraps a function with LRU caching.

    Usage:
        @memoize(max_size=128)
        def expensive_function(x, y):
            return x + y

        result = expensive_function(1, 2)  # computed
        result = expensive_function(1, 2)  # cached

        expensive_function.cache_info()    # {"size": 1, "max_size": 128}
    """
    def decorator(func: Callable) -> Callable:
        cache: dict[Any, Any] = {}
        order: list[Any] = []

        def wrapper(*args: Any, **kwargs: Any) -> Any:
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


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    @memoize(max_size=128)
    def expensive_function(x, y):
        return x + y

    result = expensive_function(1, 2)
    print(f"Result: {result}")
    print(f"Cache info: {expensive_function.cache_info()}")
