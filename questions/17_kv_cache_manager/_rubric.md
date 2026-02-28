# Rubric: KV-Cache Manager

**Total: 100 points**

---

## 1. Correct Memory-Per-Token Calculation (10 points)

| Points | Criteria |
|--------|----------|
| 10 | `bytes_per_token = num_layers * num_kv_heads * head_dim * dtype_bytes * 2` is computed correctly. The `* 2` for K + V is explicitly understood. `max_tokens = max_gpu_memory_bytes // bytes_per_token` uses integer division. |
| 7 | Formula is correct but the candidate doesn't clearly explain the `* 2` factor. |
| 3 | Formula is close but wrong (e.g., missing `* 2`, or missing `num_layers`). |
| 0 | Not computed or fundamentally wrong. |

### Key insight to verify
- With `ModelConfig(32, 8, 128, 2)`: `bytes_per_token = 32 * 8 * 128 * 2 * 2 = 131,072 = 128 KB`.
- An 8 GB budget holds `65,536` tokens total across all requests.
- Strong candidates will note this means ~16 requests at 4K context length, illustrating why KV-cache is the bottleneck.

---

## 2. Basic Allocation with Capacity Tracking (15 points)

| Points | Criteria |
|--------|----------|
| 15 | `allocate` correctly reserves `(prompt_tokens + max_gen_tokens) * bytes_per_token`, checks capacity, rejects duplicates, returns bool. `free` releases exactly the reserved amount. `stats` returns all fields with correct values. Memory accounting is consistent at all times. |
| 12 | Allocation and freeing work but one stat field is wrong or missing. |
| 8 | Basic allocation works but capacity checking or duplicate rejection is broken. |
| 4 | Allocates but memory tracking is inaccurate. |
| 0 | Not implemented. |

### Things to check
- Does `allocate` return `False` (not raise an exception) on OOM?
- Does `allocate` reject duplicate `request_id`s?
- Does `free` raise `KeyError` on unknown `request_id`?
- Is `used_memory_bytes` exactly `sum_of_reserved_tokens * bytes_per_token`?
- Is `utilization_pct` computed as `used / total * 100`?

---

## 3. Append/Get KV Correctness (10 points)

| Points | Criteria |
|--------|----------|
| 10 | `append_token` stores KV vectors per request per layer. `get_kv` returns them in order. Works for multiple tokens and multiple layers independently. Raises `KeyError` for unknown requests. |
| 7 | Works but doesn't validate vector lengths or doesn't raise on unknown requests. |
| 4 | Stores data but retrieval is incorrect (wrong order, wrong layer isolation). |
| 0 | Not implemented. |

### Expected behavior
```python
cache.append_token("req1", layer=0, key=[1.0, 2.0, ...], value=[3.0, 4.0, ...])
cache.append_token("req1", layer=0, key=[5.0, 6.0, ...], value=[7.0, 8.0, ...])
keys, vals = cache.get_kv("req1", layer=0)
assert len(keys) == 2  # two tokens cached for layer 0
keys1, vals1 = cache.get_kv("req1", layer=1)
assert len(keys1) == 0  # no tokens cached for layer 1
```

---

## 4. Paged Allocation with On-Demand Page Growth (20 points)

| Points | Criteria |
|--------|----------|
| 20 | Pages correctly sized. `allocate_paged` only allocates `ceil(prompt_tokens / page_size)` pages per layer. `append_token_paged` allocates new pages on demand when current page fills. Free list management is correct. Freed pages return to the free list. |
| 15 | Core paging works but page size math has an off-by-one, or freed pages don't return to the pool. |
| 10 | Pages are allocated but on-demand growth doesn't work correctly. |
| 5 | Attempts paging but memory accounting is wrong. |
| 0 | Not implemented. |

### Key details
- Page size: each physical page stores `page_size` tokens for one layer.
- Bytes per page: `page_size * num_kv_heads * head_dim * dtype_bytes * 2`.
- A request with `prompt_tokens=5` and `page_size=4` needs `ceil(5/4) = 2` pages **per layer**, so `2 * num_layers` total physical pages.
- The paged approach is better than pre-allocation because it doesn't reserve space for `max_gen_tokens` upfront.

### Acceptable variations
- Some candidates may implement pages as spanning all layers (one page = `page_size` tokens for all layers). This is acceptable if the memory math is consistent, but per-layer pages are more standard (matches vLLM).

---

## 5. Memory Efficiency Metric (10 points)

| Points | Criteria |
|--------|----------|
| 10 | `memory_efficiency()` correctly computes `total_stored_tokens / (num_allocated_pages * page_size)`. Handles the case where no pages are allocated (returns 0.0 or 1.0 with justification). Value is always in [0, 1]. |
| 7 | Formula is correct but edge case handling is missing. |
| 4 | Returns some efficiency metric but the formula is wrong. |
| 0 | Not implemented. |

### Example
- 5 tokens stored across 2 pages (page_size=4): efficiency = 5 / 8 = 0.625.
- This means 37.5% of allocated slots are wasted (internal fragmentation).
- Strong candidates will note that larger page sizes increase fragmentation but reduce page table overhead, and smaller page sizes are the opposite.

---

## 6. Prefix Caching with Page Sharing (20 points)

| Points | Criteria |
|--------|----------|
| 20 | Prefix pages are shared across requests with the same `prefix_hash`. Reference counting prevents premature deallocation. New prefixes are registered for future sharing. Memory savings are real (second request with same prefix allocates fewer pages). |
| 15 | Sharing works but reference counting is broken (pages freed while still in use, or never freed). |
| 10 | Prefix is cached but not actually shared (each request gets its own copy). |
| 5 | Attempts prefix caching but fundamentally broken. |
| 0 | Not implemented. |

### Key behaviors to verify
1. First request with `prefix_hash="sys_v1"`: allocates pages for prefix + unique tokens. Prefix pages are registered.
2. Second request with `prefix_hash="sys_v1"`: only allocates pages for unique tokens. Prefix pages are shared (reference count incremented).
3. Free first request: prefix pages NOT freed (ref count > 0).
4. Free second request: prefix pages freed (ref count = 0), OR kept for future reuse.

### Copy-on-write (bonus)
- If a request somehow needs to modify a shared page, it should copy the page first and modify the copy.
- In practice this doesn't happen in inference (KV-cache is append-only), but handling it shows sophistication.
- Worth 2-3 bonus points if implemented correctly.

---

## 7. Edge Cases (10 points)

| Points | Criteria |
|--------|----------|
| 10 | Handles all of: (a) OOM returns False, doesn't corrupt state; (b) free and reallocate works correctly; (c) many concurrent requests; (d) request with 0 prompt tokens; (e) request with 0 max_gen_tokens; (f) freeing unknown request raises error. |
| 7 | Most edge cases handled but one or two break. |
| 4 | Basic happy path works but edge cases crash or corrupt state. |
| 0 | No edge case handling. |

### Critical edge case: OOM should not corrupt state
```python
cache.allocate("req1", 50000, 10000)
# Fails:
cache.allocate("req2", 50000, 10000)  # returns False
# State should be unchanged — req1 still valid, no partial allocation for req2
```

---

## 8. Code Clarity and API Design (5 points)

| Points | Criteria |
|--------|----------|
| 5 | Clean separation between basic/paged/prefix modes. Good variable names (not `d`, `x`, `tmp`). Docstrings or comments explain non-obvious decisions. Consistent error handling. |
| 3 | Readable but some confusion between modes or inconsistent patterns. |
| 1 | Works but hard to follow. |
| 0 | Incomprehensible. |

---

## Grading Thresholds

| Grade | Points | Notes |
|-------|--------|-------|
| Strong Hire | 85-100 | All three parts work correctly. Memory accounting is exact. Prefix sharing saves memory. Edge cases handled. |
| Hire | 65-84 | Parts 1 and 2 work well. Prefix caching may have minor issues. Good understanding of the problem space. |
| Lean Hire | 45-64 | Part 1 works. Paged allocation has issues but the concept is understood. Prefix caching incomplete. |
| No Hire | < 45 | Cannot get basic allocation right, or memory math is fundamentally wrong. |

---

## Red Flags
- Using floating-point for memory accounting (leads to rounding errors).
- Not understanding why pre-allocation wastes memory (missing the core motivation for paged attention).
- Allocating contiguous memory for all layers together instead of per-layer pages.
- Not handling the case where `free()` should make memory available for new allocations.
- Treating the KV-cache as if it stores model weights (confusion about what's cached).

## Green Flags
- Immediately calculates bytes_per_token and reasons about how many requests fit in memory.
- Mentions vLLM and PagedAttention by name, understands the analogy to OS virtual memory.
- Discusses the trade-off between page size and fragmentation.
- Notes that GQA (fewer KV heads) reduces cache size proportionally.
- Considers what happens at long context lengths (e.g., 128K tokens per request = 16 GB just for KV-cache).
- Mentions that in practice, the KV-cache is stored in contiguous GPU memory and pages are logical, not physical.
