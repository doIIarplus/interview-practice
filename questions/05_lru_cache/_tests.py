"""Hidden tests for Question 05: LRU Cache with Persistence
Run: python questions/05_lru_cache/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import memoize


def test_basic_caching():
    """Test that basic caching works."""
    call_count = 0

    @memoize(max_size=3)
    def add(a: int, b: int) -> int:
        nonlocal call_count
        call_count += 1
        return a + b

    assert add(1, 2) == 3
    assert call_count == 1
    assert add(1, 2) == 3
    assert call_count == 1, f"Expected 1 call, got {call_count} (cache miss on repeat)"
    assert add(3, 4) == 7
    assert call_count == 2
    print("  test_basic_caching: PASSED")


def test_lru_eviction_order():
    """Test that accessing a cached entry protects it from eviction."""
    call_count = 0

    @memoize(max_size=2)
    def multiply(a: int, b: int) -> int:
        nonlocal call_count
        call_count += 1
        return a * b

    multiply(1, 2)
    multiply(3, 4)
    multiply(1, 2)   # access again — should move to most recent
    multiply(5, 6)   # should evict (3,4), NOT (1,2)

    call_count_before = call_count
    result = multiply(1, 2)
    assert result == 2
    assert call_count == call_count_before, (
        f"(1,2) was evicted even though it was recently accessed! "
        f"Calls went from {call_count_before} to {call_count}"
    )
    print("  test_lru_eviction_order: PASSED")


def test_kwargs_handling():
    """Test that kwargs are properly included in the cache key."""
    call_count = 0

    @memoize(max_size=5)
    def greet(name: str, greeting: str = "hello") -> str:
        nonlocal call_count
        call_count += 1
        return f"{greeting}, {name}!"

    result1 = greet("Alice", greeting="hello")
    assert result1 == "hello, Alice!"
    assert call_count == 1

    result2 = greet("Alice", greeting="hi")
    assert result2 == "hi, Alice!", (
        f"Expected 'hi, Alice!' but got '{result2}'"
    )
    assert call_count == 2, (
        f"Expected 2 calls but got {call_count}"
    )
    print("  test_kwargs_handling: PASSED")


def test_cache_info():
    """Test that cache_info reports correct size."""
    @memoize(max_size=3)
    def square(x: int) -> int:
        return x * x

    square(1)
    square(2)
    square(3)
    info = square.cache_info()
    assert info["size"] == 3, f"Expected size 3, got {info['size']}"
    assert info["max_size"] == 3

    square(4)
    info = square.cache_info()
    assert info["size"] == 3, f"Expected size 3 after eviction, got {info['size']}"
    print("  test_cache_info: PASSED")


def test_eviction_correctness():
    """Test that eviction removes the correct entry."""
    call_count = 0

    @memoize(max_size=2)
    def double(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    double(1)
    double(2)
    double(3)  # should evict 1

    call_count_before = call_count
    double(2)
    assert call_count == call_count_before, "2 was evicted but should still be cached"

    double(1)
    assert call_count == call_count_before + 1, "1 should have been evicted and recomputed"
    print("  test_eviction_correctness: PASSED")


def test_persistence():
    """Test save/load functionality (Part 2)."""
    import tempfile

    call_count = 0

    @memoize(max_size=5)
    def compute(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x ** 2

    for i in range(5):
        compute(i)
    assert call_count == 5

    filepath = os.path.join(tempfile.gettempdir(), "test_cache.json")

    try:
        compute.save(filepath)

        call_count2 = 0

        @memoize(max_size=5)
        def compute2(x: int) -> int:
            nonlocal call_count2
            call_count2 += 1
            return x ** 2

        compute2.load(filepath)

        for i in range(5):
            result = compute2(i)
            assert result == i ** 2
        assert call_count2 == 0, (
            f"Expected 0 calls after loading cache, got {call_count2}"
        )
        print("  test_persistence: PASSED")

    except AttributeError:
        print("  test_persistence: SKIPPED (save/load not implemented yet)")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def run_tests():
    print("=" * 60)
    print("LRU Cache Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Part 1: Basic Caching", test_basic_caching),
        ("Part 1: LRU Eviction Order", test_lru_eviction_order),
        ("Part 1: Kwargs Handling", test_kwargs_handling),
        ("Part 1: Cache Info", test_cache_info),
        ("Part 1: Eviction Correctness", test_eviction_correctness),
        ("Part 2: Persistence", test_persistence),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"Running: {name}")
        try:
            test_fn()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
