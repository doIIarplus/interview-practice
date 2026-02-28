# Follow-up Questions: Memory Pool Allocator

---

## 1. What data structure would make allocation faster?

**What we're looking for:**
- **Free lists**: Maintain linked lists of free blocks, potentially organized by size
- **Buddy system**: Split memory into powers-of-2 sized blocks, O(log n) alloc/free
- **Slab allocator**: Pre-allocate pools of fixed-size objects for common sizes
- **Segregated free lists**: Multiple free lists, one per size class
- **Skip list or balanced BST**: O(log n) first-fit search

**Strong answer includes:**
- Explanation that first-fit with a sorted list is O(n) in the number of free blocks
- Buddy system gives O(log n) but suffers from internal fragmentation (rounding up
  to power of 2)
- Real systems use a hybrid: size classes for small allocations, general allocator
  for large ones

---

## 2. How does the buddy allocator work and why is it popular for GPU memory?

**What we're looking for:**
- Memory is divided into blocks of power-of-2 sizes
- To allocate size n: find smallest power-of-2 block >= n, split larger blocks if needed
- To free: merge ("buddy") adjacent blocks back together if both are free
- Each block's buddy is found by flipping a bit in the address
- O(log n) allocation and deallocation
- Used in Linux kernel (page allocator) and some GPU memory managers

**Strong answer includes:**
- Trade-off: internal fragmentation (allocating 257 bytes requires a 512-byte block)
  vs simplicity and speed of coalescing
- Buddy coalescing is simpler than general coalescing because the buddy is
  deterministic (XOR the address with the block size)
- GPU memory often allocates large, power-of-2-aligned tensors, making buddy
  allocation a natural fit

---

## 3. What are the trade-offs between first-fit, best-fit, and worst-fit?

**What we're looking for:**
- **First-fit**: Fast (stops at first suitable block), tends to fragment the beginning
  of the pool, generally good performance in practice
- **Best-fit**: Finds smallest suitable block, minimizes wasted space per allocation,
  but creates many tiny unusable fragments and is O(n)
- **Worst-fit**: Uses largest available block, idea is to leave large remaining chunks,
  but in practice leads to rapid fragmentation of large blocks
- **Next-fit**: Like first-fit but starts where the last search ended, distributes
  allocations more evenly

**Strong answer includes:**
- In practice, first-fit and next-fit are most commonly used due to speed
- Best-fit is theoretically better for fragmentation but the tiny leftover fragments
  ("slivers") it creates are often useless
- Research shows first-fit performs surprisingly well in most workloads
- For GPU tensor allocation, size-class allocators dominate because tensor sizes
  tend to be repeated (same model architecture reuses the same shapes)

---

## 4. How would you handle concurrent allocations from multiple threads?

**What we're looking for:**
- **Mutex/lock**: Simple but serializes all allocations, creating contention
- **Reader-writer lock**: Doesn't help much since alloc/free both write
- **Per-size-class locks**: If using segregated free lists, lock each list independently
- **Lock-free approaches**: CAS-based free list manipulation
- **Thread-local caches**: Each thread has its own small pool; only contend on the
  global pool when the thread cache is exhausted (like tcmalloc/jemalloc)
- **Arena per thread**: Each thread allocates from its own arena

**Strong answer includes:**
- Mentions that CUDA operations are serialized per stream, so GPU memory allocation
  contention comes from multiple CPU threads submitting to different streams
- PyTorch's caching allocator uses a mutex and handles this by being very fast
  (the cache hit path is just a dict lookup + pop)
- jemalloc and tcmalloc approaches: thread-local caches to minimize contention

---

## 5. Real GPU memory allocators use memory pools with size classes. Why?

**What we're looking for:**
- ML workloads have highly repetitive allocation patterns (same tensor shapes used
  repeatedly during forward/backward passes)
- Size classes (e.g., 512B, 1KB, 2KB, 4KB, ..., 1GB) allow O(1) allocation by
  maintaining a free list per size class
- Round up to the next size class: small internal fragmentation, but eliminates
  external fragmentation within a class
- PyTorch's caching allocator: does NOT return memory to CUDA when freed; keeps it
  in a cache indexed by size, so the next allocation of the same size is instant

**Strong answer includes:**
- The caching behavior means `torch.cuda.empty_cache()` exists to actually return
  memory to CUDA when needed
- Size classes work because tensor shapes are determined by model architecture
  and batch size, which are constant during training
- This is why you see "CUDA out of memory" even when nvidia-smi shows free memory:
  the caching allocator may be holding fragmented blocks

---

## 6. How does memory fragmentation affect ML training workloads specifically?

**What we're looking for:**
- During training, tensors for activations, gradients, and optimizer states are
  allocated and freed repeatedly each iteration
- If fragmentation builds up, large tensor allocations fail even with sufficient
  total free memory
- Gradient checkpointing changes allocation patterns (fewer large activations
  held simultaneously, but more frequent alloc/free)
- Mixed precision training: half-precision tensors are smaller but more numerous
- Dynamic shapes (variable sequence lengths) cause more fragmentation than
  static shapes

**Strong answer includes:**
- `torch.cuda.memory_stats()` shows fragmentation metrics
- Best practices: pre-allocate tensors, use in-place operations, avoid Python-level
  tensor creation in the training loop
- Memory-efficient attention (FlashAttention) reduces peak memory by fusing
  operations, reducing the number of intermediate tensors

---

## 7. What is CUDA's Unified Memory and how does it change the allocation model?

**What we're looking for:**
- Unified Memory provides a single address space accessible from both CPU and GPU
- The CUDA runtime automatically migrates pages between CPU and GPU memory
- Simplifies programming: no manual cudaMemcpy needed
- But: page faults for migration add latency; performance is worse than manual
  management for performance-critical code
- Useful for prototyping or workloads where data access patterns are unpredictable

**Strong answer includes:**
- Page migration granularity matters (typically 4KB or 64KB)
- Prefetch hints (`cudaMemPrefetchAsync`) can mitigate page fault overhead
- For ML inference at scale, Unified Memory overhead is usually unacceptable;
  manual memory management (via allocators like the one in this question) is preferred
- CUDA Managed Memory with `cudaMallocManaged` vs explicit `cudaMalloc`
