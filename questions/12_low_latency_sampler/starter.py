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


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    sampler = TokenSampler()
    logits = [2.0, 1.0, 0.5, -1.0]
    print(f"Greedy: {sampler.sample_greedy(logits)}")
    print(f"Softmax: {sampler.softmax(logits)}")
