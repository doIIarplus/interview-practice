"""Hidden tests for Question 11: Memory Pool Allocator
Run: python questions/11_memory_pool_allocator/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import MemoryPool


def test_basic():
    """Test basic allocation and freeing."""
    pool = MemoryPool(1024)

    a = pool.alloc(256)
    b = pool.alloc(256)
    c = pool.alloc(256)

    assert a == 0, f"Expected 0, got {a}"
    assert b == 256, f"Expected 256, got {b}"
    assert c == 512, f"Expected 512, got {c}"

    pool.free(b)

    d = pool.alloc(128)
    assert d == 256, f"Expected 256, got {d}"

    stats = pool.stats()
    assert stats["used"] == 256 + 128 + 256, f"Used: {stats['used']}"
    assert stats["free"] == 1024 - 640, f"Free: {stats['free']}"

    print("[PASS] test_basic")


def test_coalescing():
    """Test that adjacent free blocks are coalesced."""
    pool = MemoryPool(1024)

    a = pool.alloc(256)
    b = pool.alloc(256)
    c = pool.alloc(256)

    pool.free(a)
    pool.free(b)

    d = pool.alloc(512)
    assert d == 0, f"Expected 0 after coalescing, got {d}"

    print("[PASS] test_coalescing")


def test_fragmentation():
    """Test fragmentation detection and defragmentation."""
    pool = MemoryPool(1024)

    blocks = []
    for i in range(8):
        blocks.append(pool.alloc(128))

    for i in range(0, 8, 2):
        pool.free(blocks[i])

    big = pool.alloc(256)
    assert big is None, "Should fail due to fragmentation"

    stats = pool.stats()
    assert stats["free"] == 512
    assert stats["fragmentation"] > 0, "Fragmentation should be > 0"

    mapping = pool.defragment()
    assert len(mapping) > 0, "Some blocks should have moved"

    stats_after = pool.stats()
    assert stats_after["fragmentation"] == 0.0, (
        f"Fragmentation should be 0 after defrag, got {stats_after['fragmentation']}"
    )

    big = pool.alloc(256)
    assert big is not None, "Should succeed after defragmentation"

    print("[PASS] test_fragmentation")


def test_aligned_allocation():
    """Test aligned allocation."""
    pool = MemoryPool(1024)

    a = pool.alloc(100)
    assert a == 0

    b = pool.alloc_aligned(200, 256)
    assert b == 256, f"Expected 256-byte aligned offset, got {b}"

    c = pool.alloc_aligned(100, 512)
    assert c == 512, f"Expected 512-byte aligned offset, got {c}"

    print("[PASS] test_aligned_allocation")


def test_error_handling():
    """Test error cases."""
    pool = MemoryPool(1024)

    try:
        pool.alloc(0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        pool.free(999)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    a = pool.alloc(256)
    pool.free(a)
    try:
        pool.free(a)
        assert False, "Double free should raise ValueError"
    except ValueError:
        pass

    print("[PASS] test_error_handling")


def test_full_pool():
    """Test behavior when pool is completely full."""
    pool = MemoryPool(256)

    a = pool.alloc(256)
    assert a == 0

    b = pool.alloc(1)
    assert b is None, "Pool is full, should return None"

    stats = pool.stats()
    assert stats["used"] == 256
    assert stats["free"] == 0
    assert stats["fragmentation"] == 0.0

    print("[PASS] test_full_pool")


def run_tests():
    print("=" * 60)
    print("Memory Pool Allocator — Hidden Tests")
    print("=" * 60 + "\n")

    test_basic()
    test_coalescing()
    test_fragmentation()
    test_aligned_allocation()
    test_error_handling()
    test_full_pool()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
