"""
Question 12: Low-Latency Token Sampler
========================================

Implement token sampling strategies for a language model inference engine.
Given a probability distribution over a vocabulary of tokens, implement
various sampling strategies optimized for low latency.

Use only the Python standard library (no numpy, no torch).

Implement the TokenSampler class below.
"""

from __future__ import annotations

import math
import random
import time


class TokenSampler:
    """Token sampling strategies for language model inference.

    All methods operate on logits (raw, unnormalized model outputs).
    Vocabulary sizes up to 128,000 tokens should be handled efficiently.
    """

    def softmax(self, logits: list[float]) -> list[float]:
        """Convert logits to a probability distribution.

        Must be numerically stable: subtract the maximum logit before
        exponentiating to prevent overflow.

        Args:
            logits: Raw model output scores, one per vocabulary token.

        Returns:
            A list of probabilities that sum to 1.0.
        """
        pass

    def sample_greedy(self, logits: list[float]) -> int:
        """Return the index of the maximum logit.

        Args:
            logits: Raw model output scores.

        Returns:
            The index (token ID) with the highest logit value.
        """
        pass

    def sample_temperature(self, logits: list[float], temperature: float) -> int:
        """Sample with temperature scaling.

        Divides logits by temperature before converting to probabilities
        and sampling. Temperature=0 should behave like greedy (no division
        by zero).

        Args:
            logits: Raw model output scores.
            temperature: Scaling factor. Higher = more random, lower = more
                         deterministic. Must be >= 0.

        Returns:
            A sampled token index.
        """
        pass

    def sample_top_k(
        self, logits: list[float], k: int, temperature: float = 1.0
    ) -> int:
        """Top-k sampling.

        Only consider the k tokens with the highest logits. Set all other
        logits to negative infinity, apply temperature, then sample from
        the resulting distribution.

        Args:
            logits: Raw model output scores.
            k: Number of top tokens to consider. Must be >= 1.
            temperature: Temperature for scaling (applied after filtering).

        Returns:
            A sampled token index (will be one of the top-k tokens).
        """
        pass

    def sample_top_p(
        self, logits: list[float], p: float, temperature: float = 1.0
    ) -> int:
        """Nucleus (top-p) sampling.

        Apply temperature, convert to probabilities, sort descending, and
        include tokens until cumulative probability >= p. Sample from this
        reduced set.

        At least one token (the highest probability) is always included.

        Args:
            logits: Raw model output scores.
            p: Cumulative probability threshold (0 < p <= 1).
            temperature: Temperature for scaling.

        Returns:
            A sampled token index.
        """
        pass

    def sample_min_p(
        self, logits: list[float], min_p: float, temperature: float = 1.0
    ) -> int:
        """Min-p sampling.

        Convert to probabilities, then filter out tokens whose probability
        is less than min_p * max_probability. Sample from the remaining
        tokens.

        At least one token (the highest probability) is always included.

        Args:
            logits: Raw model output scores.
            min_p: Minimum probability threshold as a fraction of the
                   maximum probability (0 < min_p <= 1).
            temperature: Temperature for scaling.

        Returns:
            A sampled token index.
        """
        pass


def generate_logits(vocab_size: int = 50000) -> list[float]:
    """Generate a realistic-looking logit distribution.

    Simulates a Zipf-like distribution where a small number of tokens
    have high logits and most tokens have low logits.

    Args:
        vocab_size: Number of tokens in the vocabulary.

    Returns:
        A list of logit values.
    """
    logits = []
    for i in range(vocab_size):
        base = random.gauss(-5, 2)
        if i < 100:
            # Top tokens are more likely
            base += random.gauss(5, 1)
        elif i < 1000:
            base += random.gauss(2, 1)
        logits.append(base)
    return logits


def test_correctness() -> None:
    """Basic correctness tests for all sampling methods."""
    sampler = TokenSampler()

    # Test softmax
    probs = sampler.softmax([2.0, 1.0, 0.1])
    assert abs(sum(probs) - 1.0) < 1e-6, f"Softmax doesn't sum to 1: {sum(probs)}"
    assert probs[0] > probs[1] > probs[2], "Softmax order wrong"
    print("[PASS] softmax basic")

    # Test softmax numerical stability with large logits
    probs_large = sampler.softmax([1000.0, 999.0, 998.0])
    assert abs(sum(probs_large) - 1.0) < 1e-6, "Softmax unstable with large logits"
    assert probs_large[0] > probs_large[1] > probs_large[2]
    print("[PASS] softmax numerical stability")

    # Test greedy
    assert sampler.sample_greedy([1.0, 3.0, 2.0]) == 1
    assert sampler.sample_greedy([5.0, 1.0, 1.0]) == 0
    print("[PASS] greedy")

    # Test temperature=0 (should be greedy)
    for _ in range(10):
        assert sampler.sample_temperature([1.0, 3.0, 2.0], 0.0) == 1
    print("[PASS] temperature=0 (greedy)")

    # Test top-k returns only top-k tokens
    logits = [10.0, 1.0, 1.0, 1.0, 1.0]
    results = set()
    for _ in range(100):
        results.add(sampler.sample_top_k(logits, k=2))
    assert results.issubset({0, 1}), f"Top-k returned tokens outside top-2: {results}"
    print("[PASS] top-k filtering")

    # Test top-p with very low p (should mostly return top token)
    top_results = [sampler.sample_top_p([10.0, 1.0, 1.0], p=0.01) for _ in range(100)]
    assert top_results.count(0) == 100, "Top-p with p=0.01 should almost always return top token"
    print("[PASS] top-p with low p")

    # Test min-p filters low probability tokens
    logits = [10.0, 5.0, -10.0, -10.0, -10.0]
    results = set()
    for _ in range(100):
        results.add(sampler.sample_min_p(logits, min_p=0.01))
    # Tokens 2-4 should have negligible probability
    assert results.issubset({0, 1}), f"Min-p didn't filter low prob tokens: {results}"
    print("[PASS] min-p filtering")

    print("\nAll correctness tests passed!")


def benchmark_sampler(vocab_size: int = 50000, n_iterations: int = 1000) -> None:
    """Benchmark all sampling methods.

    Args:
        vocab_size: Size of the vocabulary to test with.
        n_iterations: Number of iterations for timing.
    """
    sampler = TokenSampler()
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
        # Warm up
        for _ in range(10):
            fn()

        start = time.perf_counter()
        for _ in range(n_iterations):
            fn()
        elapsed = time.perf_counter() - start

        ms_per_call = elapsed / n_iterations * 1000
        calls_per_sec = n_iterations / elapsed
        print(f"  {name:25s}: {ms_per_call:8.3f} ms/call  ({calls_per_sec:,.0f} calls/sec)")

    print("-" * 65)


if __name__ == "__main__":
    test_correctness()
    benchmark_sampler()
