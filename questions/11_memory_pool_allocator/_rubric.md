# Rubric: Memory Pool Allocator

**Total: 100 points**

---

## 1. Correct First-Fit Allocation (20 points)

### Full marks (20):
- Maintains a sorted list of free blocks (offset, size)
- Scans from lowest offset to highest
- Splits a free block when allocation is smaller than the block
- Returns the correct starting offset
- Handles pool-full case (returns None)
- Rejects size <= 0 with ValueError

### Partial credit (10-15):
- Correct allocation but uses an inefficient data structure
- Works but doesn't split free blocks properly (wastes remaining space)

### Minimal credit (1-9):
- Returns offsets but doesn't track free/allocated correctly
- Always allocates at the end (ignores freed blocks)

### Good data structure approach:
```python
# Option 1: Sorted list of free blocks
self.free_blocks = [(0, total_size)]  # (offset, size)
self.allocated = {}  # offset -> size

# Option 2: Sorted list of all blocks with status
self.blocks = [(0, total_size, False)]  # (offset, size, is_allocated)
```

---

## 2. Free with Coalescing of Adjacent Free Blocks (20 points)

### Full marks (20):
- Correctly removes the block from allocated tracking
- Checks for adjacent free block BEFORE the freed block
- Checks for adjacent free block AFTER the freed block
- Merges all adjacent free blocks into one
- Handles freeing first block, last block, and middle blocks
- Raises ValueError for invalid offset or double-free

### Partial credit (10-15):
- Frees correctly but doesn't coalesce
- Coalesces in one direction but not both

### Minimal credit (1-9):
- Basic free without coalescing
- Doesn't validate the offset

### Critical implementation:
```python
def free(self, offset):
    if offset not in self.allocated:
        raise ValueError(f"Invalid offset: {offset}")
    size = self.allocated.pop(offset)

    # Create new free block
    new_start = offset
    new_size = size

    # Try to coalesce with previous free block
    for i, (fstart, fsize) in enumerate(self.free_blocks):
        if fstart + fsize == new_start:
            new_start = fstart
            new_size += fsize
            self.free_blocks.pop(i)
            break

    # Try to coalesce with next free block
    for i, (fstart, fsize) in enumerate(self.free_blocks):
        if new_start + new_size == fstart:
            new_size += fsize
            self.free_blocks.pop(i)
            break

    # Insert coalesced block in sorted order
    # ... insert at correct position ...
```

**This is the single most important part of the question.** Candidates who don't
coalesce on free() will produce increasingly fragmented pools, which is the #1
bug in real memory allocators.

---

## 3. Aligned Allocation (15 points)

### Full marks (15):
- Correctly computes the aligned offset within a free block
- Handles "wasted" space before the aligned offset (creates a small free block
  for the gap, or accounts for it properly)
- Validates alignment is a power of 2
- Handles case where alignment padding makes the block too small

### Partial credit (8-12):
- Correct alignment but wastes the space before the aligned offset
  (doesn't track it as a separate free block)
- Correct alignment but doesn't validate alignment parameter

### Minimal credit (1-7):
- Incorrect alignment computation
- Doesn't handle the gap before the aligned offset

### Key formula:
```python
# Align up to next multiple of alignment
aligned_offset = (offset + alignment - 1) & ~(alignment - 1)
# Or equivalently:
aligned_offset = ((offset + alignment - 1) // alignment) * alignment
```

### Space before alignment:
When a free block starts at offset 100 and alignment is 256, the aligned offset
is 256. The space from 100-255 (156 bytes) should remain as a free block, and
the allocation starts at 256. The candidate needs to handle this split.

---

## 4. Stats Computation Including Fragmentation (10 points)

### Full marks (10):
- total = pool size (constant)
- used = sum of all allocated block sizes
- free = total - used
- num_blocks = count of allocated blocks
- fragmentation = 1 - (largest_free_block / total_free), 0 if total_free == 0
- All values are consistent with each other

### Partial credit (5-8):
- Most stats correct but fragmentation formula is wrong
- Doesn't handle edge case of no free space

### Minimal credit (1-4):
- Only some stats computed correctly

---

## 5. Defragmentation with Correct Offset Mapping (20 points)

### Full marks (20):
- Compacts all allocated blocks to the beginning of the pool
- Preserves the relative order of allocated blocks
- Returns correct {old_offset: new_offset} mapping
- Only includes blocks that actually moved in the mapping
- After defrag, all free space is one contiguous block at the end
- Internal state (free blocks, allocated blocks) is consistent after defrag

### Partial credit (10-15):
- Correct compaction but mapping is incomplete or includes non-moved blocks
- Correct compaction but internal state is inconsistent

### Minimal credit (1-9):
- Partial compaction or incorrect state after defrag

### Key algorithm:
```python
def defragment(self):
    # Get all allocated blocks sorted by offset
    sorted_blocks = sorted(self.allocated.items())  # [(offset, size), ...]

    mapping = {}
    current_offset = 0
    new_allocated = {}

    for old_offset, size in sorted_blocks:
        if old_offset != current_offset:
            mapping[old_offset] = current_offset
        new_allocated[current_offset] = size
        current_offset += size

    # Update state
    self.allocated = new_allocated
    if current_offset < self.total_size:
        self.free_blocks = [(current_offset, self.total_size - current_offset)]
    else:
        self.free_blocks = []

    return mapping
```

---

## 6. Data Structure Choice (15 points)

### Full marks (15):
- Uses an appropriate sorted data structure for free blocks
- O(n) allocation (n = number of free blocks) — acceptable
- O(1) or O(log n) lookup for allocated blocks (dict)
- Clearly separates tracking of free vs allocated blocks
- Explains trade-offs of their chosen approach

### Partial credit (8-12):
- Functional but suboptimal data structure (e.g., unsorted lists requiring sort on each alloc)
- Dict for allocated, list for free — works but coalescing is awkward

### Minimal credit (1-7):
- Uses a bitmap (works but O(n) in pool size, not block count)
- No clear data structure for free blocks

### Impressive approaches:
- **Sorted list of free blocks + dict of allocated**: Good balance of simplicity and performance
- **Skip list or balanced BST for free blocks**: O(log n) allocation
- **Buddy allocator**: If they propose this as an optimization, excellent systems knowledge
- **Free list per size class**: What PyTorch's caching allocator actually does

---

## Red Flags (Automatic Deductions)

- **-15 points**: No coalescing on free() (the most critical feature)
- **-10 points**: Bitmap approach that's O(total_size) for allocation
- **-5 points**: No error handling for invalid free() / double-free
- **-5 points**: Defragment doesn't update internal state consistently
- **-5 points**: alloc(0) doesn't raise an error

---

## Exceptional Answers (Bonus Discussion Points)

- Discusses buddy allocator and why it's used in kernel/GPU memory management
- Mentions that real GPU allocators (like PyTorch caching allocator) use size classes
  to avoid fragmentation
- Discusses the trade-off between internal fragmentation (wasted space within blocks
  due to alignment/size rounding) and external fragmentation (scattered free blocks)
- Mentions that defragmentation is expensive and real systems try to avoid it
  through smart allocation strategies
- Proposes thread-safe version with appropriate locking granularity
