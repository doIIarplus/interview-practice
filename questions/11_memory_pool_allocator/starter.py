"""
Question 11: Memory Pool Allocator
====================================

Implement a memory pool allocator that manages a fixed-size contiguous block of
memory. This simulates how GPU memory allocators work — you have a fixed amount
of GPU memory and need to efficiently allocate and free blocks for tensors of
various sizes.

Implement the MemoryPool class below.
"""

from __future__ import annotations


class MemoryPool:
    """A simple memory pool allocator simulating GPU memory management.

    Manages a contiguous block of memory from offset 0 to total_size - 1.
    Uses a first-fit allocation strategy and coalesces adjacent free blocks
    on free().
    """

    def __init__(self, total_size: int) -> None:
        """Initialize a memory pool of total_size bytes.

        Args:
            total_size: Size of the memory pool in bytes. Must be positive.

        Raises:
            ValueError: If total_size is not positive.
        """
        pass

    def alloc(self, size: int) -> int | None:
        """Allocate a contiguous block of size bytes using first-fit strategy.

        Scans free blocks from lowest to highest offset and uses the first
        block that is large enough.

        Args:
            size: Number of bytes to allocate. Must be positive.

        Returns:
            The starting offset of the allocated block, or None if no
            contiguous block of sufficient size is available.

        Raises:
            ValueError: If size is not positive.
        """
        pass

    def free(self, offset: int) -> None:
        """Free the block at the given offset.

        After freeing, adjacent free blocks are coalesced (merged) into a
        single larger free block.

        Args:
            offset: The starting offset of the block to free. Must be the
                     start of a currently allocated block.

        Raises:
            ValueError: If offset does not correspond to the start of an
                        allocated block (including double-free).
        """
        pass

    def alloc_aligned(self, size: int, alignment: int) -> int | None:
        """Allocate a block aligned to the given alignment boundary.

        The returned offset will be a multiple of alignment. The alignment
        must be a positive power of 2.

        Args:
            size: Number of bytes to allocate. Must be positive.
            alignment: Alignment boundary in bytes. Must be a positive power of 2.

        Returns:
            The starting offset of the aligned allocated block, or None if
            no suitable block is available.

        Raises:
            ValueError: If size is not positive or alignment is not a power of 2.
        """
        pass

    def stats(self) -> dict:
        """Return memory pool statistics.

        Returns:
            A dict with keys:
                - "total": Total pool size in bytes
                - "used": Total bytes currently allocated
                - "free": Total bytes currently free
                - "num_blocks": Number of currently allocated blocks
                - "fragmentation": 1 - (largest_free_block / total_free),
                  or 0.0 if no free space exists
        """
        pass

    def defragment(self) -> dict[int, int]:
        """Compact all allocated blocks to the beginning of the pool.

        Moves all allocated blocks to be contiguous starting from offset 0,
        eliminating all fragmentation. The relative order of allocated blocks
        is preserved.

        Returns:
            A mapping of {old_offset: new_offset} for every allocated block
            that moved. Blocks that did not move are not included.
        """
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic() -> None:
    """Test basic allocation and freeing."""
    pool = MemoryPool(1024)

    # Allocate three blocks
    a = pool.alloc(256)  # Should return 0
    b = pool.alloc(256)  # Should return 256
    c = pool.alloc(256)  # Should return 512

    assert a == 0, f"Expected 0, got {a}"
    assert b == 256, f"Expected 256, got {b}"
    assert c == 512, f"Expected 512, got {c}"

    # Free middle block
    pool.free(b)

    # Allocate smaller block — should fit in freed space (first fit)
    d = pool.alloc(128)  # Should return 256
    assert d == 256, f"Expected 256, got {d}"

    # Check stats
    stats = pool.stats()
    assert stats["used"] == 256 + 128 + 256, f"Used: {stats['used']}"
    assert stats["free"] == 1024 - 640, f"Free: {stats['free']}"

    print("[PASS] test_basic")


def test_coalescing() -> None:
    """Test that adjacent free blocks are coalesced."""
    pool = MemoryPool(1024)

    a = pool.alloc(256)
    b = pool.alloc(256)
    c = pool.alloc(256)

    # Free a and b — they should coalesce into a single 512-byte block
    pool.free(a)
    pool.free(b)

    # Now we should be able to allocate 512 bytes at offset 0
    d = pool.alloc(512)
    assert d == 0, f"Expected 0 after coalescing, got {d}"

    print("[PASS] test_coalescing")


def test_fragmentation() -> None:
    """Test fragmentation detection and defragmentation."""
    pool = MemoryPool(1024)

    # Allocate 8 blocks of 128 bytes each (fills the pool)
    blocks = []
    for i in range(8):
        blocks.append(pool.alloc(128))

    # Free every other block — creates fragmentation
    for i in range(0, 8, 2):
        pool.free(blocks[i])

    # Can't allocate 256 bytes even though 512 bytes are free (fragmented)
    big = pool.alloc(256)
    assert big is None, "Should fail due to fragmentation"

    # Check fragmentation metric
    stats = pool.stats()
    assert stats["free"] == 512
    assert stats["fragmentation"] > 0, "Fragmentation should be > 0"

    # Defragment
    mapping = pool.defragment()
    assert len(mapping) > 0, "Some blocks should have moved"

    # After defragmentation, fragmentation should be 0
    stats_after = pool.stats()
    assert stats_after["fragmentation"] == 0.0, (
        f"Fragmentation should be 0 after defrag, got {stats_after['fragmentation']}"
    )

    # Now should be able to allocate 256 bytes
    big = pool.alloc(256)
    assert big is not None, "Should succeed after defragmentation"

    print("[PASS] test_fragmentation")


def test_aligned_allocation() -> None:
    """Test aligned allocation."""
    pool = MemoryPool(1024)

    a = pool.alloc(100)  # Returns 0
    assert a == 0

    # Allocate aligned to 256 bytes — should skip to offset 256
    b = pool.alloc_aligned(200, 256)
    assert b == 256, f"Expected 256-byte aligned offset, got {b}"

    # Allocate aligned to 512 bytes
    c = pool.alloc_aligned(100, 512)
    assert c == 512, f"Expected 512-byte aligned offset, got {c}"

    print("[PASS] test_aligned_allocation")


def test_error_handling() -> None:
    """Test error cases."""
    pool = MemoryPool(1024)

    # Allocate zero bytes should raise
    try:
        pool.alloc(0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Free invalid offset should raise
    try:
        pool.free(999)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Double free should raise
    a = pool.alloc(256)
    pool.free(a)
    try:
        pool.free(a)
        assert False, "Double free should raise ValueError"
    except ValueError:
        pass

    print("[PASS] test_error_handling")


def test_full_pool() -> None:
    """Test behavior when pool is completely full."""
    pool = MemoryPool(256)

    a = pool.alloc(256)
    assert a == 0

    # Pool is full
    b = pool.alloc(1)
    assert b is None, "Pool is full, should return None"

    stats = pool.stats()
    assert stats["used"] == 256
    assert stats["free"] == 0
    assert stats["fragmentation"] == 0.0

    print("[PASS] test_full_pool")


if __name__ == "__main__":
    test_basic()
    test_coalescing()
    test_fragmentation()
    test_aligned_allocation()
    test_error_handling()
    test_full_pool()
    print("\nAll tests passed!")
