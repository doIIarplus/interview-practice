"""Hidden tests for Question 17: KV-Cache Manager
Run: python questions/17_kv_cache_manager/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import ModelConfig, KVCacheManager


def test_bytes_per_token():
    """Verify the per-token memory calculation."""
    config = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)
    assert config.bytes_per_token == 32 * 8 * 128 * 2 * 2
    assert config.bytes_per_token == 131072
    print("[PASS] test_bytes_per_token")


def test_basic_allocation():
    """Test basic pre-allocated cache: allocate, stats, free."""
    config = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)
    cache = KVCacheManager(config, max_gpu_memory_bytes=8 * 1024**3)
    assert cache.max_tokens == 65536

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

    kv_size = config.num_kv_heads * config.head_dim
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

    keys1, values1 = cache.get_kv("req1", layer=1)
    assert len(keys1) == 0
    print("[PASS] test_basic_append_and_get")


def test_oom_rejection():
    """Test that allocation fails when memory is exhausted."""
    config = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)
    cache = KVCacheManager(config, max_gpu_memory_bytes=100 * 131072)

    assert cache.allocate("req1", prompt_tokens=50, max_gen_tokens=40)
    assert not cache.allocate("req2", prompt_tokens=20, max_gen_tokens=20)

    cache.free("req1")
    assert cache.allocate("req2", prompt_tokens=20, max_gen_tokens=20)
    print("[PASS] test_oom_rejection")


def test_paged_allocation():
    """Test paged allocation: on-demand page growth."""
    config = ModelConfig(num_layers=2, num_kv_heads=2, head_dim=4, dtype_bytes=2)
    page_size = 4
    cache = KVCacheManager(config, max_gpu_memory_bytes=4096, page_size=page_size)

    assert cache.allocate_paged("req1", prompt_tokens=5)

    kv_size = config.num_kv_heads * config.head_dim
    for i in range(5):
        cache.append_token_paged("req1", layer=0, key=[float(i)] * kv_size, value=[float(i)] * kv_size)

    eff = cache.memory_efficiency()
    assert 0.0 < eff <= 1.0, f"Efficiency should be in (0, 1], got {eff}"
    print("[PASS] test_paged_allocation")


def test_prefix_caching():
    """Test prefix caching: shared pages for common system prompts."""
    config = ModelConfig(num_layers=2, num_kv_heads=2, head_dim=4, dtype_bytes=2)
    cache = KVCacheManager(config, max_gpu_memory_bytes=65536, page_size=4)
    cache.enable_prefix_cache()

    assert cache.allocate_with_prefix("req1", prefix_hash="system_v1", prefix_tokens=8, unique_tokens=4)
    assert cache.allocate_with_prefix("req2", prefix_hash="system_v1", prefix_tokens=8, unique_tokens=6)

    s = cache.stats()
    assert s["num_active_requests"] == 2

    cache.free("req1")
    s = cache.stats()
    assert s["num_active_requests"] == 1
    print("[PASS] test_prefix_caching")


def test_free_and_reallocate():
    """Test that freed memory can be reused."""
    config = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)
    cache = KVCacheManager(config, max_gpu_memory_bytes=8 * 1024**3)

    assert cache.allocate("big_req", prompt_tokens=60000, max_gen_tokens=5000)
    s1 = cache.stats()
    assert s1["utilization_pct"] > 90.0

    assert not cache.allocate("another", prompt_tokens=1000, max_gen_tokens=500)

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
        if cache.allocate(f"req_{i}", prompt_tokens=prompt, max_gen_tokens=gen):
            allocated += 1

    assert allocated > 0
    s = cache.stats()
    assert s["num_active_requests"] == allocated

    for i in range(allocated // 2):
        cache.free(f"req_{i}")

    s = cache.stats()
    assert s["num_active_requests"] == allocated - allocated // 2
    print(f"[PASS] test_multiple_concurrent_requests (allocated {allocated} of 100)")


def run_tests():
    print("Running KV-Cache Manager tests...\n")
    test_bytes_per_token()
    test_basic_allocation()
    test_basic_append_and_get()
    test_oom_rejection()
    test_paged_allocation()
    test_prefix_caching()
    test_free_and_reallocate()
    test_multiple_concurrent_requests()
    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
