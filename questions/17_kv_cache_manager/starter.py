"""
Question 17: KV-Cache Manager

Manage the Key-Value cache for a multi-tenant LLM serving system.
Covers basic allocation, paged attention, and prefix caching.

See QUESTION.md for full problem description.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the LLM model's KV-cache dimensions."""
    num_layers: int = 32
    num_kv_heads: int = 8    # GQA: fewer KV heads than query heads
    head_dim: int = 128
    dtype_bytes: int = 2     # float16 = 2 bytes per element

    @property
    def bytes_per_token(self) -> int:
        """Memory cost per token across all layers (both K and V)."""
        return self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes * 2


# ---------------------------------------------------------------------------
# Part 1: Basic KV-Cache Manager
# ---------------------------------------------------------------------------

class KVCacheManager:
    """
    Manages KV-cache memory for multiple concurrent LLM inference requests.

    Part 1: Basic pre-allocated cache.
    Part 2: Paged cache with on-demand allocation.
    Part 3: Prefix caching with page sharing.
    """

    def __init__(
        self,
        config: ModelConfig,
        max_gpu_memory_bytes: int,
        page_size: int = 16,
    ) -> None:
        """
        Initialize the KV-cache manager.

        Args:
            config: Model configuration (layers, heads, dims, dtype).
            max_gpu_memory_bytes: Total GPU memory budget for KV-cache.
            page_size: Tokens per page (used in Part 2 paged mode).
        """
        self.config = config
        self.max_gpu_memory_bytes = max_gpu_memory_bytes
        self.page_size = page_size
        self.bytes_per_token = config.bytes_per_token
        self.max_tokens = max_gpu_memory_bytes // self.bytes_per_token

        # TODO: Initialize tracking structures for Part 1 (basic allocation)
        # - Active requests and their reserved token counts
        # - KV storage per request per layer

        # TODO: Initialize structures for Part 2 (paged allocation)
        # - Physical pages and free list
        # - Page tables mapping (request, layer, page_index) -> physical page

        # TODO: Initialize structures for Part 3 (prefix caching)
        # - Prefix registry mapping prefix_hash -> list of page IDs
        # - Reference counts for shared pages

    # -------------------------------------------------------------------
    # Part 1: Basic (pre-allocated) API
    # -------------------------------------------------------------------

    def allocate(self, request_id: str, prompt_tokens: int, max_gen_tokens: int) -> bool:
        """
        Reserve cache space for a new request (pre-allocated mode).

        Reserves enough space for prompt_tokens + max_gen_tokens.
        Returns False if insufficient memory. Rejects duplicate request IDs.
        """
        # TODO: Implement
        pass

    def append_token(self, request_id: str, layer: int, key: list[float], value: list[float]) -> None:
        """
        Append a KV entry for one layer of one new token.

        Args:
            request_id: Active request identifier.
            layer: Layer index (0 to num_layers - 1).
            key: Key vector of length num_kv_heads * head_dim.
            value: Value vector of length num_kv_heads * head_dim.

        Raises:
            KeyError: If request_id does not exist.
        """
        # TODO: Implement
        pass

    def get_kv(self, request_id: str, layer: int) -> tuple[list[list[float]], list[list[float]]]:
        """
        Retrieve all cached keys and values for a request at a given layer.

        Returns:
            (all_keys, all_values) where each is a list of vectors,
            one per cached token.

        Raises:
            KeyError: If request_id does not exist.
        """
        # TODO: Implement
        pass

    def free(self, request_id: str) -> None:
        """
        Release all memory for a completed request.

        Raises:
            KeyError: If request_id does not exist.
        """
        # TODO: Implement
        pass

    def stats(self) -> dict:
        """
        Return current cache statistics.

        Returns:
            {
                "total_memory_bytes": int,
                "used_memory_bytes": int,
                "num_active_requests": int,
                "total_cached_tokens": int,
                "utilization_pct": float,
            }
        """
        # TODO: Implement
        pass

    # -------------------------------------------------------------------
    # Part 2: Paged KV-Cache API
    # -------------------------------------------------------------------

    def allocate_paged(self, request_id: str, prompt_tokens: int) -> bool:
        """
        Allocate pages for a new request (paged mode).

        Only allocates pages needed for the prompt. Additional pages
        are allocated on demand during token generation.

        Returns False if insufficient free pages for the prompt.
        """
        # TODO: Implement
        pass

    def append_token_paged(self, request_id: str, layer: int, key: list[float], value: list[float]) -> None:
        """
        Append a KV entry using paged storage.

        If the current page is full, allocates a new page from the free list.

        Raises:
            KeyError: If request_id does not exist.
            MemoryError: If no free pages available.
        """
        # TODO: Implement
        pass

    def memory_efficiency(self) -> float:
        """
        Ratio of actually-used token slots to total allocated token slots.

        Returns a float in [0.0, 1.0]. Measures internal fragmentation.
        1.0 = no wasted slots in any page.
        """
        # TODO: Implement
        pass

    # -------------------------------------------------------------------
    # Part 3: Prefix Caching
    # -------------------------------------------------------------------

    def enable_prefix_cache(self) -> None:
        """Enable prefix caching mode for shared system prompts."""
        # TODO: Implement
        pass

    def allocate_with_prefix(
        self,
        request_id: str,
        prefix_hash: str,
        prefix_tokens: int,
        unique_tokens: int,
    ) -> bool:
        """
        Allocate with a shared prefix (paged mode required).

        If prefix_hash was seen before, reuse those pages (read-only, shared).
        Otherwise, allocate pages for prefix + unique and register the prefix.

        Returns False if insufficient memory.
        """
        # TODO: Implement
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bytes_per_token():
    """Verify the per-token memory calculation."""
    config = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)
    assert config.bytes_per_token == 32 * 8 * 128 * 2 * 2  # 131,072
    assert config.bytes_per_token == 131072
    print("[PASS] test_bytes_per_token")


def test_basic_allocation():
    """Test basic pre-allocated cache: allocate, stats, free."""
    config = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)
    cache = KVCacheManager(config, max_gpu_memory_bytes=8 * 1024**3)

    # Max tokens = 8 GB / 128 KB = 65536
    assert cache.max_tokens == 65536

    # Allocate two requests
    assert cache.allocate("req1", prompt_tokens=1000, max_gen_tokens=500)
    assert cache.allocate("req2", prompt_tokens=2000, max_gen_tokens=1000)

    s = cache.stats()
    assert s["num_active_requests"] == 2
    assert s["used_memory_bytes"] == (1500 + 3000) * 131072

    # Reject duplicate
    assert not cache.allocate("req1", prompt_tokens=100, max_gen_tokens=100)

    # Free and verify
    cache.free("req1")
    s = cache.stats()
    assert s["num_active_requests"] == 1
    assert s["used_memory_bytes"] == 3000 * 131072

    print("[PASS] test_basic_allocation")


def test_basic_append_and_get():
    """Test appending and retrieving KV entries."""
    config = ModelConfig(num_layers=2, num_kv_heads=2, head_dim=4, dtype_bytes=2)
    cache = KVCacheManager(config, max_gpu_memory_bytes=1024 * 1024)
    cache.allocate("req1", prompt_tokens=10, max_gen_tokens=10)

    kv_size = config.num_kv_heads * config.head_dim  # 2 * 4 = 8
    key1 = [1.0] * kv_size
    val1 = [2.0] * kv_size
    key2 = [3.0] * kv_size
    val2 = [4.0] * kv_size

    cache.append_token("req1", layer=0, key=key1, value=val1)
    cache.append_token("req1", layer=0, key=key2, value=val2)

    keys, values = cache.get_kv("req1", layer=0)
    assert len(keys) == 2, f"Expected 2 keys, got {len(keys)}"
    assert len(values) == 2
    assert keys[0] == key1
    assert values[1] == val2

    # Layer 1 should be empty
    keys1, values1 = cache.get_kv("req1", layer=1)
    assert len(keys1) == 0

    print("[PASS] test_basic_append_and_get")


def test_oom_rejection():
    """Test that allocation fails when memory is exhausted."""
    config = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)
    # Only enough for 100 tokens
    cache = KVCacheManager(config, max_gpu_memory_bytes=100 * 131072)

    assert cache.allocate("req1", prompt_tokens=50, max_gen_tokens=40)  # 90 tokens
    assert not cache.allocate("req2", prompt_tokens=20, max_gen_tokens=20)  # 40 > 10 remaining

    # Free and retry
    cache.free("req1")
    assert cache.allocate("req2", prompt_tokens=20, max_gen_tokens=20)  # now fits

    print("[PASS] test_oom_rejection")


def test_paged_allocation():
    """Test paged allocation: on-demand page growth."""
    config = ModelConfig(num_layers=2, num_kv_heads=2, head_dim=4, dtype_bytes=2)
    # bytes_per_token = 2 * 2 * 4 * 2 * 2 = 64
    page_size = 4
    # bytes_per_page = page_size * bytes_per_token_per_layer
    #   but pages are per-layer, so bytes_per_page = page_size * num_kv_heads * head_dim * dtype_bytes * 2
    #   = 4 * 2 * 4 * 2 * 2 = 128 bytes per page per layer
    # We need pages for each layer, so total per logical page = 128 * 2 layers = 256
    # Actually, each physical page is for one layer. See QUESTION.md.

    cache = KVCacheManager(config, max_gpu_memory_bytes=4096, page_size=page_size)

    # Allocate for a prompt of 5 tokens (needs 2 pages per layer: ceil(5/4)=2)
    assert cache.allocate_paged("req1", prompt_tokens=5)

    kv_size = config.num_kv_heads * config.head_dim  # 8

    # Append 5 prompt tokens to layer 0
    for i in range(5):
        cache.append_token_paged("req1", layer=0, key=[float(i)] * kv_size, value=[float(i)] * kv_size)

    # Efficiency: 5 tokens stored, 2 pages * 4 slots = 8 allocated slots (layer 0 only so far)
    eff = cache.memory_efficiency()
    assert 0.0 < eff <= 1.0, f"Efficiency should be in (0, 1], got {eff}"

    print("[PASS] test_paged_allocation")


def test_prefix_caching():
    """Test prefix caching: shared pages for common system prompts."""
    config = ModelConfig(num_layers=2, num_kv_heads=2, head_dim=4, dtype_bytes=2)
    cache = KVCacheManager(config, max_gpu_memory_bytes=65536, page_size=4)
    cache.enable_prefix_cache()

    # First request with a prefix
    assert cache.allocate_with_prefix("req1", prefix_hash="system_v1", prefix_tokens=8, unique_tokens=4)

    # Second request with same prefix — should reuse prefix pages
    assert cache.allocate_with_prefix("req2", prefix_hash="system_v1", prefix_tokens=8, unique_tokens=6)

    s = cache.stats()
    assert s["num_active_requests"] == 2

    # Free req1; prefix pages should remain (req2 still using them)
    cache.free("req1")
    s = cache.stats()
    assert s["num_active_requests"] == 1

    print("[PASS] test_prefix_caching")


def test_free_and_reallocate():
    """Test that freed memory can be reused."""
    config = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)
    cache = KVCacheManager(config, max_gpu_memory_bytes=8 * 1024**3)

    # Fill up most of memory
    assert cache.allocate("big_req", prompt_tokens=60000, max_gen_tokens=5000)
    s1 = cache.stats()
    assert s1["utilization_pct"] > 90.0

    # Cannot fit another large request
    assert not cache.allocate("another", prompt_tokens=1000, max_gen_tokens=500)

    # Free and reallocate
    cache.free("big_req")
    assert cache.allocate("another", prompt_tokens=1000, max_gen_tokens=500)

    print("[PASS] test_free_and_reallocate")


def test_multiple_concurrent_requests():
    """Test many concurrent requests with different sizes."""
    config = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)
    cache = KVCacheManager(config, max_gpu_memory_bytes=8 * 1024**3)

    allocated = 0
    for i in range(100):
        prompt = 200
        gen = 100
        total = prompt + gen
        if cache.allocate(f"req_{i}", prompt_tokens=prompt, max_gen_tokens=gen):
            allocated += 1

    assert allocated > 0
    s = cache.stats()
    assert s["num_active_requests"] == allocated

    # Free half
    for i in range(allocated // 2):
        cache.free(f"req_{i}")

    s = cache.stats()
    assert s["num_active_requests"] == allocated - allocated // 2

    print(f"[PASS] test_multiple_concurrent_requests (allocated {allocated} of 100)")


if __name__ == "__main__":
    test_bytes_per_token()
    test_basic_allocation()
    test_basic_append_and_get()
    test_oom_rejection()
    test_paged_allocation()
    test_prefix_caching()
    test_free_and_reallocate()
    test_multiple_concurrent_requests()
    print("\nAll tests passed!")
