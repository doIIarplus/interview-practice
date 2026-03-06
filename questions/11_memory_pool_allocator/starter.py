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

# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    pool = MemoryPool(1024)
    a = pool.alloc(256)
    b = pool.alloc(256)
    print(f"Allocated at: {a}, {b}")
    print(f"Stats: {pool.stats()}")
