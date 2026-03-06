"""Hidden tests for Question 12: Low-Latency Token Sampler
Run: python questions/12_low_latency_sampler/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
from starter import TokenSampler, generate_logits


def test_softmax_basic():
    """Test basic softmax properties."""
    sampler = TokenSampler()
    probs = sampler.softmax([2.0, 1.0, 0.1])
    assert abs(sum(probs) - 1.0) < 1e-6, f"Softmax doesn't sum to 1: {sum(probs)}"
    assert probs[0] > probs[1] > probs[2], "Softmax order wrong"
    print("[PASS] softmax basic")


def test_softmax_stability():
    """Test softmax numerical stability with large logits."""
    sampler = TokenSampler()
    probs = sampler.softmax([1000.0, 999.0, 998.0])
    assert abs(sum(probs) - 1.0) < 1e-6, "Softmax unstable with large logits"
    assert probs[0] > probs[1] > probs[2]
    print("[PASS] softmax numerical stability")


def test_greedy():
    """Test greedy sampling."""
    sampler = TokenSampler()
    assert sampler.sample_greedy([1.0, 3.0, 2.0]) == 1
    assert sampler.sample_greedy([5.0, 1.0, 1.0]) == 0
    print("[PASS] greedy")


def test_temperature_zero():
    """Test temperature=0 (should be greedy)."""
    sampler = TokenSampler()
    for _ in range(10):
        assert sampler.sample_temperature([1.0, 3.0, 2.0], 0.0) == 1
    print("[PASS] temperature=0 (greedy)")


def test_top_k():
    """Test top-k returns only top-k tokens."""
    sampler = TokenSampler()
    logits = [10.0, 1.0, 1.0, 1.0, 1.0]
    results = set()
    for _ in range(100):
        results.add(sampler.sample_top_k(logits, k=2))
    assert results.issubset({0, 1}), f"Top-k returned tokens outside top-2: {results}"
    print("[PASS] top-k filtering")


def test_top_p():
    """Test top-p with very low p."""
    sampler = TokenSampler()
    top_results = [sampler.sample_top_p([10.0, 1.0, 1.0], p=0.01) for _ in range(100)]
    assert top_results.count(0) == 100, "Top-p with p=0.01 should almost always return top token"
    print("[PASS] top-p with low p")


def test_min_p():
    """Test min-p filters low probability tokens."""
    sampler = TokenSampler()
    logits = [10.0, 5.0, -10.0, -10.0, -10.0]
    results = set()
    for _ in range(100):
        results.add(sampler.sample_min_p(logits, min_p=0.01))
    assert results.issubset({0, 1}), f"Min-p didn't filter low prob tokens: {results}"
    print("[PASS] min-p filtering")


def test_benchmark():
    """Benchmark all sampling methods."""
    sampler = TokenSampler()
    vocab_size = 50000
    n_iterations = 100
    logits = generate_logits(vocab_size)

    print(f"\nBenchmark: vocab_size={vocab_size}, iterations={n_iterations}")
    print("-" * 65)

    methods = [
        ("greedy", lambda: sampler.sample_greedy(logits)),
        ("temperature(0.8)", lambda: sampler.sample_temperature(logits, 0.8)),
        ("top_k(50)", lambda: sampler.sample_top_k(logits, 50)),
        ("top_p(0.9)", lambda: sampler.sample_top_p(logits, 0.9)),
        ("min_p(0.05)", lambda: sampler.sample_min_p(logits, 0.05)),
    ]

    for name, fn in methods:
        for _ in range(5):
            fn()
        start = time.perf_counter()
        for _ in range(n_iterations):
            fn()
        elapsed = time.perf_counter() - start
        ms_per_call = elapsed / n_iterations * 1000
        print(f"  {name:25s}: {ms_per_call:8.3f} ms/call")

    print("-" * 65)
    print("[PASS] benchmark")


def run_tests():
    print("=" * 60)
    print("Low-Latency Token Sampler — Hidden Tests")
    print("=" * 60 + "\n")

    test_softmax_basic()
    test_softmax_stability()
    test_greedy()
    test_temperature_zero()
    test_top_k()
    test_top_p()
    test_min_p()
    test_benchmark()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
