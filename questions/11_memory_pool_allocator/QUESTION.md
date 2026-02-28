# Question 11: Memory Pool Allocator

## Difficulty: Hard
## Time: 60 minutes
## Category: Memory Management / Systems Programming

---

## Background

GPU memory management is a critical component of ML inference systems. Unlike CPU
memory, GPU memory is a fixed, contiguous resource that cannot be swapped to disk.
Efficient allocation and deallocation of GPU memory for tensors of varying sizes
directly impacts inference throughput and latency.

This question asks you to implement a simplified memory pool allocator that captures
the essential challenges of GPU memory management: contiguous allocation, fragmentation,
alignment requirements, and defragmentation.

---

## Task

Implement a `MemoryPool` class that manages a fixed-size contiguous block of memory.

### Methods

#### `__init__(self, total_size: int)`

Initialize a memory pool of `total_size` bytes. The pool represents byte offsets
from 0 to `total_size - 1`.

#### `alloc(self, size: int) -> int | None`

Allocate a contiguous block of `size` bytes using a **first-fit** strategy: scan
free blocks from lowest offset to highest and use the first one that fits.

Return the starting offset of the allocated block, or `None` if no contiguous block
of sufficient size is available.

**Example:**
```python
pool = MemoryPool(1024)
a = pool.alloc(256)   # Returns 0
b = pool.alloc(256)   # Returns 256
c = pool.alloc(256)   # Returns 512
d = pool.alloc(512)   # Returns None (only 256 bytes free)
```

#### `free(self, offset: int) -> None`

Free the block starting at the given offset. After freeing, adjacent free blocks
should be **coalesced** (merged) into a single larger free block.

**Example:**
```python
pool = MemoryPool(1024)
a = pool.alloc(256)   # 0
b = pool.alloc(256)   # 256
c = pool.alloc(256)   # 512

pool.free(b)          # Free middle block
pool.free(a)          # Free first block — coalesces with freed 'b' into [0, 512)

d = pool.alloc(512)   # Returns 0 (coalesced block is large enough)
```

#### `alloc_aligned(self, size: int, alignment: int) -> int | None`

Allocate a block of `size` bytes where the starting offset is a multiple of
`alignment`. This is important for GPU memory access patterns where tensors must
be aligned to 256-byte or 512-byte boundaries.

**Example:**
```python
pool = MemoryPool(1024)
a = pool.alloc(100)         # Returns 0
b = pool.alloc_aligned(200, 256)  # Returns 256 (next 256-byte boundary)
# Note: bytes 100-255 become internal fragmentation
```

#### `stats(self) -> dict`

Return statistics about the memory pool:

```python
{
    "total": int,           # Total pool size
    "used": int,            # Total bytes in allocated blocks
    "free": int,            # Total bytes in free blocks
    "num_blocks": int,      # Number of allocated blocks
    "fragmentation": float  # 1 - (largest_free_block / total_free)
                            # 0 if nothing is free
}
```

**Fragmentation** measures how scattered the free space is. If all free space is
in one contiguous block, fragmentation is 0. If free space is split into many small
blocks, fragmentation approaches 1.

**Example:**
```python
pool = MemoryPool(1024)
pool.alloc(256)
pool.alloc(256)
pool.alloc(256)
stats = pool.stats()
# stats == {"total": 1024, "used": 768, "free": 256,
#           "num_blocks": 3, "fragmentation": 0.0}
# fragmentation is 0 because all free space is one block
```

#### `defragment(self) -> dict[int, int]`

Compact all allocated blocks toward the beginning of the pool, eliminating
fragmentation. Return a mapping of `{old_offset: new_offset}` so that callers
can update their pointers/references.

**Example:**
```python
pool = MemoryPool(1024)
a = pool.alloc(128)   # 0
b = pool.alloc(128)   # 128
c = pool.alloc(128)   # 256
d = pool.alloc(128)   # 384

pool.free(a)          # Free block at 0
pool.free(c)          # Free block at 256

# Memory layout: [FREE 128][USED 128][FREE 128][USED 128][FREE 512]

mapping = pool.defragment()
# mapping == {128: 0, 384: 128}
# Memory layout after: [USED 128][USED 128][FREE 768]
```

---

## Important Details

- **Coalescing**: When a block is freed, if the adjacent blocks (before and/or after)
  are also free, they must be merged into a single free block. This prevents
  fragmentation from growing unboundedly.

- **Error handling**: `free()` with an invalid offset (not the start of an allocated
  block) should raise a `ValueError`. Double-free should also raise an error.

- **Allocation sizes**: All sizes are positive integers. `alloc(0)` should raise
  a `ValueError`.

---

## Starter Code

See `starter.py` for the class skeleton and test cases.

---

## Evaluation Criteria

- Correct first-fit allocation with proper block tracking
- Free with coalescing of adjacent free blocks (critical)
- Aligned allocation handling
- Accurate statistics including fragmentation metric
- Correct defragmentation with offset mapping
- Appropriate data structure choice for tracking blocks
