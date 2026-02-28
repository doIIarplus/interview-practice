# Question 17: KV-Cache Manager

## Background

In autoregressive LLM inference, each newly generated token must attend to **all previous tokens**. The self-attention mechanism requires Key and Value tensors for every token at every layer. Recomputing these from scratch on each step would be prohibitively expensive, so they are **cached** -- this is the KV-cache.

The KV-cache is the single largest memory consumer during LLM inference. For a 70B-parameter model serving a request with 4096 tokens of context, the KV-cache alone can consume **several gigabytes**. When serving hundreds of concurrent requests, efficient KV-cache management becomes the primary bottleneck for serving throughput.

You are building a **KV-cache manager** for a multi-tenant LLM serving system.

---

## Model Configuration

```python
@dataclass
class ModelConfig:
    num_layers: int = 32
    num_kv_heads: int = 8    # GQA: fewer KV heads than query heads
    head_dim: int = 128
    dtype_bytes: int = 2     # float16 = 2 bytes per element
```

The memory cost **per token** across all layers is:

```
bytes_per_token = num_layers * num_kv_heads * head_dim * dtype_bytes * 2
```

The `* 2` accounts for both the Key and Value tensors.

---

## Part 1: Basic KV-Cache

Implement the `KVCacheManager` class with the following methods:

### 1. `__init__(self, config: ModelConfig, max_gpu_memory_bytes: int)`

- Store the configuration.
- Calculate `bytes_per_token` using the formula above.
- Calculate the maximum number of tokens that can be cached given the GPU memory budget.
- Initialize tracking structures for active requests.

### 2. `allocate(self, request_id: str, prompt_tokens: int, max_gen_tokens: int) -> bool`

- Reserve cache space for a new request. The reservation must cover `prompt_tokens + max_gen_tokens` tokens (worst case: the model generates the maximum number of tokens).
- Return `True` if allocation succeeds, `False` if there is insufficient memory.
- Reject duplicate request IDs.

### 3. `append_token(self, request_id: str, layer: int, key: list[float], value: list[float])`

- Append a new KV entry for one layer of one token for the given request.
- The `key` and `value` lists each have length `num_kv_heads * head_dim`.
- Raise an error if the request does not exist.

### 4. `get_kv(self, request_id: str, layer: int) -> tuple[list[list[float]], list[list[float]]]`

- Return `(all_keys, all_values)` for the given request and layer.
- `all_keys` is a list of key vectors (one per cached token).
- `all_values` is a list of value vectors (one per cached token).

### 5. `free(self, request_id: str)`

- Release all memory associated with a completed request.
- Raise an error if the request does not exist.

### 6. `stats() -> dict`

Return:
```python
{
    "total_memory_bytes": int,         # total GPU memory budget
    "used_memory_bytes": int,          # currently allocated
    "num_active_requests": int,
    "total_cached_tokens": int,        # actual tokens stored (not reserved)
    "utilization_pct": float,          # used / total * 100
}
```

---

## Part 2: Paged Attention

The basic approach **wastes memory** because it pre-allocates for `max_gen_tokens` even though most requests generate far fewer tokens. Implement a **paged KV-cache** inspired by [vLLM's PagedAttention](https://arxiv.org/abs/2309.06180).

### Concept

Instead of contiguous pre-allocated buffers, memory is managed as fixed-size **pages**:

- Each page holds `page_size` tokens (e.g., 16 tokens per page).
- Pages are allocated **on demand** as tokens are generated.
- A **page table** maps `(request_id, layer, logical_page_index)` to a physical page.
- Free pages are tracked in a **free list**.

### 7. `__init__` for Paged Mode

The constructor should accept an additional `page_size: int = 16` parameter.
- Divide total memory into pages. Each page holds `page_size` tokens for **one layer** of **one KV head group**.
- Actually, for simplicity, each page stores `page_size` tokens worth of K and V data for **all heads of one layer**: `page_size * num_kv_heads * head_dim * dtype_bytes * 2` bytes per page.
- Maintain a free list of physical pages.

### 8. `allocate_paged(self, request_id: str, prompt_tokens: int) -> bool`

- Only allocate enough pages for the prompt tokens (not for future generation).
- New pages are allocated lazily as tokens are generated.
- Return `False` if not enough free pages for the prompt.

### 9. `append_token_paged(self, request_id: str, layer: int, key: list[float], value: list[float])`

- Append to the current (last) page for this request and layer.
- If the current page is full, allocate a new page from the free list.
- Raise an error if no free pages are available.

### 10. `memory_efficiency(self) -> float`

Return the ratio of **actually used token slots** to **total allocated token slots**:

```
efficiency = tokens_actually_stored / (num_allocated_pages * page_size)
```

This measures internal fragmentation. A value close to 1.0 means little wasted space.

---

## Part 3: Prefix Caching

Many LLM serving scenarios involve a **shared system prompt** prepended to every user request. Computing and storing the KV-cache for this prefix repeatedly is wasteful.

### 11. `enable_prefix_cache(self)`

Enable prefix caching mode. After this call, allocations can specify a shared prefix.

### 12. `allocate_with_prefix(self, request_id: str, prefix_hash: str, prefix_tokens: int, unique_tokens: int) -> bool`

- If `prefix_hash` has been seen before, **reuse** the existing KV-cache pages for the prefix portion (shared, read-only).
- Only allocate new pages for `unique_tokens`.
- If `prefix_hash` is new, allocate pages for both prefix and unique tokens, and register the prefix pages for future sharing.
- Return `False` if insufficient memory.
- Use **copy-on-write** semantics: if a request needs to modify a shared prefix page (this shouldn't normally happen in inference, but handle it gracefully), copy the page first.

---

## Example

```python
config = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)

# Memory per token = 32 * 8 * 128 * 2 * 2 = 131,072 bytes = 128 KB
bytes_per_token = 32 * 8 * 128 * 2 * 2
assert bytes_per_token == 131072

# 8 GB of GPU memory for KV-cache
cache = KVCacheManager(config, max_gpu_memory_bytes=8 * 1024**3)

# Maximum tokens = 8 GB / 128 KB = 65,536 tokens
max_tokens = (8 * 1024**3) // 131072
assert max_tokens == 65536

# Allocate two requests
assert cache.allocate("req1", prompt_tokens=1000, max_gen_tokens=500)   # reserves 1500 tokens
assert cache.allocate("req2", prompt_tokens=2000, max_gen_tokens=1000)  # reserves 3000 tokens

s = cache.stats()
assert s["num_active_requests"] == 2
assert s["used_memory_bytes"] == (1500 + 3000) * 131072

# Free a request and reallocate
cache.free("req1")
s = cache.stats()
assert s["num_active_requests"] == 1
```

---

## Constraints

- All memory accounting should be exact (no floating-point memory sizes).
- The `key` and `value` vectors passed to `append_token` have length `num_kv_heads * head_dim`.
- For Part 2 (paged), you may assume `page_size` evenly divides typical prompt lengths, but handle the case where it doesn't (partial pages).
- Thread safety is not required (single-threaded).
- Focus on correct memory accounting and API semantics, not on actual GPU memory allocation.
